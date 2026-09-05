"""
F4 최종 보강: zero-day(미학습 변형공격)에서 "탐지"와 "유형정확"을 분리.
"""
import pandas as pd, numpy as np, lightgbm as lgb, os

HERE = os.path.dirname(__file__)
FEATS = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy",
          "mean_byte", "delta_zscore", "value_zscore", "norm_id"]
ATTACKS = ["DoS", "Fuzzy", "gear", "RPM"]

print("[로드] features_full.parquet ...")
fd = pd.read_parquet(os.path.join(HERE, "..", "data_full", "features_full.parquet"))
train = fd[fd["is_test_region"] == False].reset_index(drop=True)
test = fd[fd["is_test_region"] == True].reset_index(drop=True)

npz = np.load(os.path.join(HERE, "..", "models", "maha_detector_full.npz"))
maha_mean, maha_inv_cov, maha_thr = npz["mean"], npz["inv_cov"], float(npz["thr"])

def maha_predict(X):
    diff = X[FEATS].values - maha_mean
    dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, maha_inv_cov, diff))
    return (dist > maha_thr).astype(int)

params = {"objective": "multiclass", "metric": "multi_logloss",
          "num_leaves": 15, "max_depth": 5, "learning_rate": 0.1, "verbose": -1, "seed": 42}

print(f"\n{'held-out':10s} | {'LGBM 탐지':>10s} | {'LGBM 유형정확':>12s} | {'Maha 탐지':>10s} | {'Hybrid 탐지':>12s} | {'Hybrid 유형정확':>14s}")
print("-" * 85)

rows = []
for held_out in ATTACKS:
    tr_sub = train[train["Label"] != held_out]
    classes2 = sorted(tr_sub["Label"].unique())
    c2i2 = {c: i for i, c in enumerate(classes2)}
    ds2 = lgb.Dataset(tr_sub[FEATS], label=tr_sub["Label"].map(c2i2))
    m2 = lgb.train({**params, "num_class": len(classes2)}, ds2, num_boost_round=80)

    te_atk = test[test["Label"] == held_out]
    pred2 = np.argmax(m2.predict(te_atk[FEATS]), axis=1)
    i2c2 = {v: k for k, v in c2i2.items()}
    pred2_label = np.array([i2c2[p] for p in pred2])

    lgbm_detect = (pred2_label != "R").mean()
    lgbm_type_acc = 0.0

    maha_sub = maha_predict(te_atk)
    maha_detect = maha_sub.mean()

    hybrid_detect = ((pred2_label != "R") | ((pred2_label == "R") & (maha_sub == 1))).mean()
    hybrid_type_acc = 0.0

    print(f"{held_out:10s} | {lgbm_detect:>9.1%} | {lgbm_type_acc:>11.1%} | {maha_detect:>9.1%} | {hybrid_detect:>11.1%} | {hybrid_type_acc:>13.1%}")
    rows.append({"held_out": held_out, "lgbm_detect": lgbm_detect, "maha_detect": maha_detect,
                 "hybrid_detect": hybrid_detect})

print("\n[핵심 해석] zero-day에서는 '유형정확'이 애초에 불가능(학습에 없던 공격이므로 이름을 맞출 수 없음).")
print("           대신 '탐지'만으로 비교하면, LightGBM 단독이 놓치는 gear/RPM을")
print("           Hybrid가 Mahalanobis 백업으로 확실히 커버한다는 게 명확히 드러남.")

out_path = os.path.join(HERE, "..", "results", "f4_zeroday_detection_split.csv")
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"\n[완료] {out_path} 저장됨")
