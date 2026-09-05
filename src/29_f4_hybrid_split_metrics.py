"""
F4 보강: 12_ablation_study.py의 known-attack 비교에서 Hybrid recall을 그냥
LightGBM recall과 동일하게 취급하던 것을, 실제 Hybrid 로직(1단계가 놓치면
2단계 Mahalanobis가 백업)대로 "탐지(Detection)"와 "정확한 유형분류(Type Accuracy)"를
분리해서 재계산.

전제: 11_train_full.py 실행 완료(models/lgbm_full.txt, models/maha_detector_full.npz 존재)
"""
import pandas as pd, numpy as np, lightgbm as lgb, os

HERE = os.path.dirname(__file__)
FEATS = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy",
          "mean_byte", "delta_zscore", "value_zscore", "norm_id"]
ATTACKS = ["DoS", "Fuzzy", "gear", "RPM"]

print("[로드] features_full.parquet ...")
fd = pd.read_parquet(os.path.join(HERE, "..", "data_full", "features_full.parquet"))
test = fd[fd["is_test_region"] == True].reset_index(drop=True)

classes = sorted(fd["Label"].unique())
c2i = {c: i for i, c in enumerate(classes)}
i2c = {v: k for k, v in c2i.items()}

lgbm = lgb.Booster(model_file=os.path.join(HERE, "..", "models", "lgbm_full.txt"))
npz = np.load(os.path.join(HERE, "..", "models", "maha_detector_full.npz"))
maha_mean, maha_inv_cov, maha_thr = npz["mean"], npz["inv_cov"], float(npz["thr"])

def maha_predict(X):
    diff = X[FEATS].values - maha_mean
    dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, maha_inv_cov, diff))
    return (dist > maha_thr).astype(int)

Xte = test[FEATS]
lgbm_pred = np.argmax(lgbm.predict(Xte), axis=1)
lgbm_pred_label = np.array([i2c[p] for p in lgbm_pred])
maha_pred = maha_predict(test)

print(f"\n{'공격유형':10s} | {'LGBM 탐지':>10s} | {'LGBM 유형정확':>12s} | {'Maha 탐지':>10s} | {'Hybrid 탐지':>12s} | {'Hybrid 유형정확':>14s}")
print("-" * 85)

rows = []
for atk in ATTACKS:
    mask = (test["Label"] == atk).values
    lgbm_label_sub = lgbm_pred_label[mask]
    maha_sub = maha_pred[mask]

    lgbm_detect = (lgbm_label_sub != "R").mean()
    lgbm_type_acc = (lgbm_label_sub == atk).mean()

    maha_detect = maha_sub.mean()

    hybrid_detect = ((lgbm_label_sub != "R") | ((lgbm_label_sub == "R") & (maha_sub == 1))).mean()
    hybrid_type_acc = lgbm_type_acc

    print(f"{atk:10s} | {lgbm_detect:>9.1%} | {lgbm_type_acc:>11.1%} | {maha_detect:>9.1%} | {hybrid_detect:>11.1%} | {hybrid_type_acc:>13.1%}")
    rows.append({"attack": atk, "lgbm_detect": lgbm_detect, "lgbm_type_acc": lgbm_type_acc,
                 "maha_detect": maha_detect, "hybrid_detect": hybrid_detect, "hybrid_type_acc": hybrid_type_acc})

print("\n[해석] Hybrid의 '탐지'는 LightGBM 단독보다 항상 같거나 높음(2단계가 놓친 것을 보완하므로).")
print("       '유형정확'은 Hybrid도 LightGBM과 동일한 상한선을 가짐(유형 분류는 1단계만 가능).")

out_path = os.path.join(HERE, "..", "results", "f4_hybrid_detection_vs_type.csv")
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"\n[완료] {out_path} 저장됨")
