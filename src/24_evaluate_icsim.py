"""
22번의 ground truth와 23번의 예측을 대조. ICSim은 이진(normal/anomaly) 판정.
"""
import pandas as pd
import numpy as np
import os

HERE = os.path.dirname(__file__)
gt = pd.read_csv(os.path.join(HERE, "..", "results", "realtime_ground_truth.csv"))
pred = pd.read_csv(os.path.join(HERE, "..", "results", "realtime_predictions.csv"))

print(f"Ground truth 공격 버스트: {len(gt)}건")
print(f"실시간 예측 메시지: {len(pred):,}건\n")

print("=== 공격유형별 탐지율(any-alert) 및 탐지 지연시간 ===")
recalls = {}
latencies = {}
for _, row in gt.iterrows():
    label = row["label"]
    mask = (pred["timestamp"] >= row["start_ts"]) & (pred["timestamp"] <= row["end_ts"])
    window_preds = pred[mask]
    if len(window_preds) == 0:
        continue
    any_alert = (window_preds["final_label"] == "anomaly").mean()
    first_alert = window_preds[window_preds["final_label"] == "anomaly"]
    det_latency = (first_alert["timestamp"].min() - row["start_ts"]) if len(first_alert) > 0 else None

    recalls.setdefault(label, []).append(any_alert)
    if det_latency is not None:
        latencies.setdefault(label, []).append(det_latency)
    lat_str = f"{det_latency*1000:.1f}ms" if det_latency is not None else "탐지못함"
    print(f"[{label}] any-alert: {any_alert:.1%} | 첫 탐지까지: {lat_str}")

print("\n=== 유형별 평균 ===")
for label, vals in recalls.items():
    lat = np.mean(latencies.get(label, [np.nan])) * 1000
    print(f"[{label}] 평균 any-alert={np.mean(vals):.1%}, 평균 탐지지연={lat:.1f}ms")

print("\n=== 정상 구간(ICSim idle) 오탐율(FPR) ===")
attack_mask = np.zeros(len(pred), dtype=bool)
for _, row in gt.iterrows():
    attack_mask |= (pred["timestamp"] >= row["start_ts"]) & (pred["timestamp"] <= row["end_ts"])
normal_preds = pred[~attack_mask]
fpr = (normal_preds["final_label"] == "anomaly").mean()
print(f"정상 구간 메시지 {len(normal_preds):,}건 중 오탐율: {fpr:.3%}")

out_path = os.path.join(HERE, "..", "results", "icsim_eval_summary.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("=== ICSim 실시간 검증 결과 (ICSim 자체 캘리브레이션 기준) ===\n\n")
    for label, vals in recalls.items():
        lat = np.mean(latencies.get(label, [np.nan])) * 1000
        f.write(f"[{label}] any-alert={np.mean(vals):.1%} 평균탐지지연={lat:.1f}ms\n")
    f.write(f"\n정상 구간 오탐율(FPR): {fpr:.3%}\n")

print(f"\n[완료] {out_path} 저장됨")
