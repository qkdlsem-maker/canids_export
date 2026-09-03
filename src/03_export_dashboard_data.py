"""
실제 학습된 LightGBM + Mahalanobis 모델로 test 구간 메시지를 추론하고,
그 결과(진짜 예측값·지연시간)를 JSON으로 저장 -> 대시보드가 이 파일을 읽어서 재생.
(즉, 대시보드의 '실시간성'은 재생 애니메이션이지만, 표시되는 예측/지연시간 값 자체는
 실제 모델이 낸 결과이며 무작위 생성이 아님)
"""
import pandas as pd, numpy as np, lightgbm as lgb, time, json, pickle

import os
fd = pd.read_parquet(os.path.join(os.path.dirname(__file__), "..", "data", "features_v3.parquet"))
FEATS = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy", "mean_byte", "delta_zscore", "value_zscore", "norm_id"]
test = fd[fd["is_test_region"] == 1].reset_index(drop=True)

m = lgb.Booster(model_file="../models/lgbm_final.txt")
npz = np.load("../models/maha_detector.npz")
mean_, inv_cov, thr = npz["mean"], npz["inv_cov"], float(npz["thr"])
classes = sorted(fd["Label"].unique())

def maha_dist(row):
    diff = row - mean_
    return np.sqrt(diff @ inv_cov @ diff)

# 시각화용으로 각 라벨에서 골고루 샘플링 (정상 다수 + 공격 소수, 실제 등장 순서 유지)
sample = pd.concat([
    test[test["Label"] == "R"].sample(40, random_state=1),
    test[test["Label"] == "DoS"].sample(6, random_state=1),
    test[test["Label"] == "Fuzzy"].sample(6, random_state=1),
    test[test["Label"] == "gear"].sample(6, random_state=1),
    test[test["Label"] == "RPM"].sample(6, random_state=1),
]).sample(frac=1, random_state=7).reset_index(drop=True)  # 셔플해서 재생 순서로

events = []
for _, row in sample.iterrows():
    x = row[FEATS].values.astype(float).reshape(1, -1)
    t0 = time.time()
    proba = m.predict(x)[0]
    pred_class = classes[int(np.argmax(proba))]
    is_known_attack = pred_class != "R"
    d = maha_dist(x[0])
    is_anomaly = d > thr
    lat_ms = (time.time() - t0) * 1000

    if is_known_attack:
        status, atk_name = "attack", pred_class
    elif is_anomaly:
        status, atk_name = "attack", "미지 이상패턴"
    else:
        status, atk_name = "normal", None

    events.append({
        "true_label": row["Label"],
        "status": status,
        "attack_name": atk_name,
        "latency_ms": round(lat_ms, 3),
        "maha_dist": round(float(d), 2),
    })

with open("../models/dashboard_data.json", "w", encoding="utf-8") as f:
    json.dump(events, f, ensure_ascii=False, indent=2)

n_attack = sum(1 for e in events if e["status"] == "attack")
print(f"exported {len(events)} real inference events ({n_attack} attacks) -> dashboard_data.json")
print(events[:3])
