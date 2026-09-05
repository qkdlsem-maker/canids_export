"""
24_evaluate_icsim.py를 대체하는 정정판.
기존 24번은 "공격 시간구간 내 전체 메시지"를 분모로 recall을 계산해서,
ICSim의 고밀도 배경 트래픽(초당 ~2000개)이 저속으로 주입된 공격 메시지를
20배 가까이 희석시키는 문제가 있었음(9/4 실험 중 발견).

이 스크립트는 "실제로 주입된 공격 메시지"만 정밀하게 필터링해서 재평가:
- DoS/spoof: 공격이 사용한 정확한 CAN ID로 필터링
- Fuzzy: is_unknown_id==1로 필터링 (ICSim 배경 트래픽은 전부 known_id라 0)
- spoof는 추가로 value_zscore>10으로, 같은 ID의 ICSim 자체 정상 메시지와 분리

사용법: python3 27_evaluate_icsim_fixed.py <dos_id_hex> <spoof_id1_hex> <spoof_id2_hex>
예:     python3 27_evaluate_icsim_fixed.py 0x39 0x294 0x143
"""
import sys, os
import pandas as pd
import numpy as np

HERE = os.path.dirname(__file__)
if len(sys.argv) < 4:
    print("사용법: python3 27_evaluate_icsim_fixed.py <dos_id_hex> <spoof_id1_hex> <spoof_id2_hex>")
    sys.exit(1)

DOS_ID = int(sys.argv[1], 16)
SPOOF_ID1 = int(sys.argv[2], 16)
SPOOF_ID2 = int(sys.argv[3], 16)
VALUE_ZSCORE_THRESHOLD = 10

gt = pd.read_csv(os.path.join(HERE, "..", "results", "realtime_ground_truth.csv"))
pred = pd.read_csv(os.path.join(HERE, "..", "results", "realtime_predictions.csv"))

if "value_zscore" not in pred.columns:
    print("[경고] predictions.csv에 value_zscore 컬럼이 없습니다.")
    sys.exit(1)

pred["label"] = "normal_bg"
for _, row in gt.iterrows():
    m = (pred["timestamp"] >= row["start_ts"]) & (pred["timestamp"] <= row["end_ts"])
    pred.loc[m, "label"] = row["label"]

print("=== 정정된 평가: 실제 주입된 공격 메시지만 필터링 ===\n")

results = {}

dos_real = pred[(pred["label"] == "DoS") & (pred["can_id"] == DOS_ID)]
if len(dos_real) > 0:
    recall = (dos_real["final_label"] == "anomaly").mean()
    results["DoS"] = (recall, len(dos_real))
    print(f"[DoS] 진짜 공격 메시지 n={len(dos_real)}, 탐지율={recall:.1%}")
else:
    print("[DoS] 이번 실행에 캡처된 버스트 없음")

fuzzy_real = pred[(pred["label"] == "Fuzzy") & (pred["is_unknown_id"] == 1)]
if len(fuzzy_real) > 0:
    recall = (fuzzy_real["final_label"] == "anomaly").mean()
    results["Fuzzy"] = (recall, len(fuzzy_real))
    print(f"[Fuzzy] 진짜 공격 메시지 n={len(fuzzy_real)}, 탐지율={recall:.1%}")
else:
    print("[Fuzzy] 이번 실행에 캡처된 버스트 없음")

for name, cid in [("spoof1", SPOOF_ID1), ("spoof2", SPOOF_ID2)]:
    same_id = pred[(pred["label"] == name) & (pred["can_id"] == cid)]
    if len(same_id) == 0:
        print(f"[{name}] 이번 실행에 캡처된 버스트 없음")
        continue
    real = same_id[same_id["value_zscore"] > VALUE_ZSCORE_THRESHOLD]
    legit_same_id = same_id[same_id["value_zscore"] <= VALUE_ZSCORE_THRESHOLD]
    if len(real) > 0:
        recall = (real["final_label"] == "anomaly").mean()
        results[name] = (recall, len(real))
        print(f"[{name}] 진짜 스푸핑 메시지 n={len(real)}, 탐지율={recall:.1%}  (같은 ID의 ICSim 정상 메시지 {len(legit_same_id)}개는 제외)")
    else:
        print(f"[{name}] value_zscore>{VALUE_ZSCORE_THRESHOLD} 메시지 없음")

attack_mask = np.zeros(len(pred), dtype=bool)
for _, row in gt.iterrows():
    attack_mask |= (pred["timestamp"] >= row["start_ts"]) & (pred["timestamp"] <= row["end_ts"])
normal_preds = pred[~attack_mask]
fpr = (normal_preds["final_label"] == "anomaly").mean()
print(f"\n[정상구간 FPR] {fpr:.3%} ({len(normal_preds):,}건)")

out_path = os.path.join(HERE, "..", "results", "icsim_eval_fixed_summary.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("=== ICSim 정정 평가 (실제 공격 메시지 기준) ===\n\n")
    for name, (recall, n) in results.items():
        f.write(f"{name}: 탐지율={recall:.1%} (n={n})\n")
    f.write(f"\n정상구간 FPR: {fpr:.3%}\n")

print(f"\n[완료] {out_path} 저장됨")
