"""
15_traffic_gen.py의 ground truth와 16_realtime_ids.py의 예측 로그를 대조해서
실제 스트리밍 환경에서의 공격 탐지율, 오탐률, 탐지 지연시간을 계산.

사용법: python3 17_evaluate_realtime.py
(15, 16번을 같은 시간에 각각 다른 터미널에서 동시 실행한 뒤 실행)
"""
import csv, os
import pandas as pd
import numpy as np

HERE = os.path.dirname(__file__)
gt = pd.read_csv(os.path.join(HERE, "..", "results", "realtime_ground_truth.csv"))
pred = pd.read_csv(os.path.join(HERE, "..", "results", "realtime_predictions.csv"))

print(f"Ground truth 공격 버스트: {len(gt)}건")
print(f"실시간 예측 메시지: {len(pred):,}건\n")

# ---------- 공격 구간별 탐지율 + 탐지 지연시간 ----------
print("=== 공격유형별 탐지율 및 탐지 지연시간 ===")
recalls = {}
latencies = {}
for _, row in gt.iterrows():
    label = row["label"]
    mask = (pred["timestamp"] >= row["start_ts"]) & (pred["timestamp"] <= row["end_ts"])
    window_preds = pred[mask]
    if len(window_preds) == 0:
        continue
    # 정확한 유형까지 맞춘 것과, 최소한 뭔가 이상하다고 잡은 것(any-alert) 둘 다 계산
    exact_match = (window_preds["final_label"] == label).mean()
    any_alert = (window_preds["final_label"] != "R").mean()

    first_alert = window_preds[window_preds["final_label"] != "R"]
    if len(first_alert) > 0:
        det_latency = first_alert["timestamp"].min() - row["start_ts"]
    else:
        det_latency = None

    recalls.setdefault(label, []).append((exact_match, any_alert))
    if det_latency is not None:
        latencies.setdefault(label, []).append(det_latency)

    print(f"[{label}] 정확 유형 일치율: {exact_match:.1%} | 이상탐지(any-alert): {any_alert:.1%} | "
          f"첫 탐지까지 걸린 시간: {det_latency*1000:.1f}ms" if det_latency is not None else
          f"[{label}] 정확 유형 일치율: {exact_match:.1%} | any-alert: {any_alert:.1%} | 탐지 못함")

print("\n=== 공격유형별 평균 ===")
for label, vals in recalls.items():
    exact = np.mean([v[0] for v in vals])
    anyv = np.mean([v[1] for v in vals])
    lat = np.mean(latencies.get(label, [np.nan])) * 1000
    print(f"[{label}] 평균 정확일치율={exact:.1%}, 평균 any-alert={anyv:.1%}, 평균 탐지지연={lat:.1f}ms")

# ---------- 정상 구간 오탐율(FPR) ----------
print("\n=== 정상 구간 오탐율(FPR) ===")
attack_mask = np.zeros(len(pred), dtype=bool)
for _, row in gt.iterrows():
    attack_mask |= (pred["timestamp"] >= row["start_ts"]) & (pred["timestamp"] <= row["end_ts"])
normal_preds = pred[~attack_mask]
fpr = (normal_preds["final_label"] != "R").mean()
print(f"정상 구간 메시지 {len(normal_preds):,}건 중 오탐율: {fpr:.3%}")

# ---------- 전체 요약 저장 ----------
out_path = os.path.join(HERE, "..", "results", "realtime_eval_summary.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("=== 실시간 스트리밍 검증 결과 (vcan0, can-utils/python-can 기반) ===\n\n")
    for label, vals in recalls.items():
        exact = np.mean([v[0] for v in vals])
        anyv = np.mean([v[1] for v in vals])
        lat = np.mean(latencies.get(label, [np.nan])) * 1000
        f.write(f"[{label}] 정확일치율={exact:.1%} any-alert={anyv:.1%} 평균탐지지연={lat:.1f}ms\n")
    f.write(f"\n정상 구간 오탐율(FPR): {fpr:.3%}\n")

print(f"\n[완료] {out_path} 저장됨")
