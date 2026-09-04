"""
23번 실행 결과(dist 값 포함된 predictions.csv)를 다시 캡처하지 않고,
여러 임계값(threshold)을 오프라인으로 스윕해서 FPR/탐지율 트레이드오프를 확인.
"""
import pandas as pd
import numpy as np
import os

HERE = os.path.dirname(__file__)
gt = pd.read_csv(os.path.join(HERE, "..", "results", "realtime_ground_truth.csv"))
pred = pd.read_csv(os.path.join(HERE, "..", "results", "realtime_predictions.csv"))

attack_mask = np.zeros(len(pred), dtype=bool)
for _, row in gt.iterrows():
    attack_mask |= (pred["timestamp"] >= row["start_ts"]) & (pred["timestamp"] <= row["end_ts"])

normal_dist = pred.loc[~attack_mask, "dist"].values
attack_pred = pred[attack_mask].copy()

attack_pred["label"] = None
for _, row in gt.iterrows():
    m = (attack_pred["timestamp"] >= row["start_ts"]) & (attack_pred["timestamp"] <= row["end_ts"])
    attack_pred.loc[m, "label"] = row["label"]

print(f"정상 메시지: {len(normal_dist):,}건, 공격 메시지: {len(attack_pred):,}건\n")
print(f"{'percentile':>10} | {'threshold':>10} | {'FPR':>8} | {'Fuzzy':>7} | {'DoS':>7} | {'spoof1':>7} | {'spoof2':>7}")
print("-" * 75)

for pct in [99.9, 99.5, 99.0, 98.0, 95.0, 90.0, 85.0, 80.0]:
    thr = np.percentile(normal_dist, pct)
    fpr = (normal_dist > thr).mean()
    recalls = {}
    for label in ["Fuzzy", "DoS", "spoof1", "spoof2"]:
        sub = attack_pred[attack_pred["label"] == label]
        if len(sub) == 0:
            recalls[label] = float("nan")
            continue
        recalls[label] = (sub["dist"] > thr).mean()
    print(f"{pct:>10} | {thr:>10.3f} | {fpr:>7.3%} | {recalls['Fuzzy']:>6.1%} | {recalls['DoS']:>6.1%} | {recalls['spoof1']:>6.1%} | {recalls['spoof2']:>6.1%}")

out_path = os.path.join(HERE, "..", "results", "icsim_threshold_sweep.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("percentile,threshold,FPR,Fuzzy,DoS,spoof1,spoof2\n")
    for pct in [99.9, 99.5, 99.0, 98.0, 95.0, 90.0, 85.0, 80.0]:
        thr = np.percentile(normal_dist, pct)
        fpr = (normal_dist > thr).mean()
        recalls = {}
        for label in ["Fuzzy", "DoS", "spoof1", "spoof2"]:
            sub = attack_pred[attack_pred["label"] == label]
            recalls[label] = (sub["dist"] > thr).mean() if len(sub) > 0 else float("nan")
        f.write(f"{pct},{thr:.3f},{fpr:.4%},{recalls['Fuzzy']:.1%},{recalls['DoS']:.1%},{recalls['spoof1']:.1%},{recalls['spoof2']:.1%}\n")

print(f"\n[완료] {out_path} 저장됨")
