"""
F2 보강: 시드를 고정하고 3개의 다른 시드로 3회 반복 실행하여
F1/FPR/지연시간/zero-day 탐지율이 얼마나 안정적인지(평균±표준편차) 확인.
5% 서브셋 결과(F1=1.0000)가 "우연"이 아니라 재현 가능한지 통계적으로 뒷받침.
"""
import pandas as pd, numpy as np, lightgbm as lgb, time, os
from sklearn.metrics import f1_score

HERE = os.path.dirname(__file__)
FEATS = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy",
          "mean_byte", "delta_zscore", "value_zscore", "norm_id"]
SEEDS = [42, 123, 2024]

print("[로드] features_full.parquet ...")
fd = pd.read_parquet(os.path.join(HERE, "..", "data_full", "features_full.parquet"))
train = fd[fd["is_test_region"] == False].reset_index(drop=True)
test = fd[fd["is_test_region"] == True].reset_index(drop=True)
print(f"train={len(train):,} test={len(test):,}")

classes = sorted(train["Label"].unique())
c2i = {c: i for i, c in enumerate(classes)}
Xtr, ytr = train[FEATS], train["Label"].map(c2i)
Xte, yte = test[FEATS], test["Label"].map(c2i)

class MahalanobisDetector:
    def __init__(self, cols): self.cols = cols
    def fit(self, X, percentile=99.9):
        self.mean_ = X[self.cols].mean().values
        cov = np.cov(X[self.cols].values, rowvar=False) + np.eye(len(self.cols)) * 1e-6
        self.inv_cov_ = np.linalg.inv(cov)
        self.threshold_ = np.percentile(self._dist(X), percentile)
        return self
    def _dist(self, X):
        diff = X[self.cols].values - self.mean_
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self.inv_cov_, diff))
    def predict(self, X):
        return (self._dist(X) > self.threshold_).astype(int)

results = []
for seed in SEEDS:
    print(f"\n{'='*50}\n[시드={seed}]\n{'='*50}")
    t0 = time.time()

    params = {"objective": "multiclass", "num_class": len(classes), "metric": "multi_logloss",
              "num_leaves": 15, "max_depth": 5, "learning_rate": 0.1, "verbose": -1,
              "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
              "deterministic": True}
    ds = lgb.Dataset(Xtr, label=ytr)
    m = lgb.train(params, ds, num_boost_round=80)
    pred = np.argmax(m.predict(Xte), axis=1)
    f1 = f1_score(yte, pred, average="macro")

    row = Xte.iloc[[0]]
    n_reps = 300
    t1 = time.time()
    for _ in range(n_reps):
        m.predict(row)
    lgbm_latency = (time.time() - t1) / n_reps * 1000

    train_normal = train[train["Label"] == "R"]
    maha = MahalanobisDetector(FEATS).fit(train_normal)
    test_normal = test[test["Label"] == "R"]
    fpr = maha.predict(test_normal).mean()

    zeroday_recalls = {}
    for held_out in ["DoS", "Fuzzy", "gear", "RPM"]:
        te_atk = test[test["Label"] == held_out]
        zeroday_recalls[held_out] = maha.predict(te_atk).mean()

    elapsed = time.time() - t0
    print(f"F1={f1:.4f}  FPR={fpr:.4%}  LightGBM지연={lgbm_latency:.4f}ms  소요={elapsed:.1f}s")
    for atk, r in zeroday_recalls.items():
        print(f"  zero-day[{atk}]={r:.2%}")

    results.append({"seed": seed, "f1": f1, "fpr": fpr, "latency_ms": lgbm_latency, **{f"zd_{k}": v for k, v in zeroday_recalls.items()}})

# ---------- 3회 요약 (평균 ± 표준편차) ----------
print(f"\n{'='*50}\n[3회 반복 요약: 평균 ± 표준편차]\n{'='*50}")
df = pd.DataFrame(results)
summary_lines = []
for col in ["f1", "fpr", "latency_ms", "zd_DoS", "zd_Fuzzy", "zd_gear", "zd_RPM"]:
    mean, std = df[col].mean(), df[col].std()
    line = f"{col}: {mean:.4f} ± {std:.4f}  (개별값: {list(df[col].round(4))})"
    print(line)
    summary_lines.append(line)

out_path = os.path.join(HERE, "..", "results", "f2_repeated_summary.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("=== F2: 시드 고정 3회 반복 실행 결과 ===\n\n")
    f.write("\n".join(summary_lines))
    f.write("\n\n표준편차가 작을수록 결과가 시드에 무관하게 안정적임을 의미\n")

print(f"\n[완료] {out_path} 저장됨")
