"""
F4 보강: 12_ablation_study.py의 LightGBM 부분에 시드를 고정하고 3회(42,123,2024) 반복해서
LightGBM 단독 vs Mahalanobis 단독 vs Hybrid 비교 결과가 시드 무관하게 안정적인지 확인.
"""
import pandas as pd, numpy as np, lightgbm as lgb, time, os

HERE = os.path.dirname(__file__)
FEATS = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy",
          "mean_byte", "delta_zscore", "value_zscore", "norm_id"]
ATTACKS = ["DoS", "Fuzzy", "gear", "RPM"]
SEEDS = [42, 123, 2024]

print("[로드] features_full.parquet ...")
fd = pd.read_parquet(os.path.join(HERE, "..", "data_full", "features_full.parquet"))
train = fd[fd["is_test_region"] == False].reset_index(drop=True)
test = fd[fd["is_test_region"] == True].reset_index(drop=True)
classes = sorted(train["Label"].unique())
c2i = {c: i for i, c in enumerate(classes)}
i2c = {v: k for k, v in c2i.items()}

train_normal = train[train["Label"] == "R"]
mean_ = train_normal[FEATS].mean().values
cov = np.cov(train_normal[FEATS].values, rowvar=False) + np.eye(len(FEATS)) * 1e-6
inv_cov = np.linalg.inv(cov)
def maha_dist(X):
    diff = X[FEATS].values - mean_
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))
thr = np.percentile(maha_dist(train_normal), 99.9)
def maha_predict(X):
    return (maha_dist(X) > thr).astype(int)

maha_pred_test = maha_predict(test)
print(f"[Mahalanobis, 결정론적] threshold={thr:.3f}")

all_runs = []
for seed in SEEDS:
    print(f"\n{'='*50}\n[시드={seed}]\n{'='*50}")
    params = {"objective": "multiclass", "num_class": len(classes), "metric": "multi_logloss",
              "num_leaves": 15, "max_depth": 5, "learning_rate": 0.1, "verbose": -1,
              "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed, "deterministic": True}
    ds = lgb.Dataset(train[FEATS], label=train["Label"].map(c2i))
    m = lgb.train(params, ds, num_boost_round=80)

    Xte = test[FEATS]
    lgbm_pred = np.argmax(m.predict(Xte), axis=1)
    lgbm_pred_label = np.array([i2c[p] for p in lgbm_pred])

    row = {"seed": seed}
    for atk in ATTACKS:
        mask = (test["Label"] == atk).values
        row[f"known_{atk}"] = (lgbm_pred_label[mask] == atk).mean()

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
        row[f"zeroday_lgbm_{held_out}"] = (pred2_label != "R").mean()

    row_ = Xte.iloc[[0]]
    t0 = time.time()
    for _ in range(300):
        m.predict(row_)
    row["lgbm_latency_ms"] = (time.time() - t0) / 300 * 1000

    print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    all_runs.append(row)

df = pd.DataFrame(all_runs)
print(f"\n{'='*50}\n[3회 반복 요약: 평균 ± 표준편차]\n{'='*50}")
summary_lines = []
for col in df.columns:
    if col == "seed":
        continue
    mean, std = df[col].mean(), df[col].std()
    line = f"{col}: {mean:.4f} ± {std:.4f}"
    print(line)
    summary_lines.append(line)

out_path = os.path.join(HERE, "..", "results", "f4_ablation_repeated_summary.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("=== F4 Ablation: 시드 3회 반복 (LightGBM 부분) ===\n\n")
    f.write("\n".join(summary_lines))
    f.write(f"\n\nMahalanobis(결정론적, 시드무관): threshold={thr:.3f}\n")
    for atk in ATTACKS:
        sub = test[test["Label"] == atk]
        recall = maha_predict(sub).mean()
        f.write(f"  known_{atk}(Mahalanobis 단독)={recall:.4f}\n")

print(f"\n[완료] {out_path} 저장됨")
