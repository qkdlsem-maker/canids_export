"""
HCRL 원본 전체 데이터셋(features_full.parquet)으로 하이브리드 모델 재학습/재검증.
5% 서브셋 결과(F1=1.0000, zero-day 99.9~100%, 메모리 0.694MB, 지연 1.21ms)와
동일하게 재현되는지 확인하는 것이 목적.
"""
import pandas as pd, numpy as np, lightgbm as lgb, time, os
from sklearn.metrics import f1_score, classification_report

HERE = os.path.dirname(__file__)
FEATS = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy",
          "mean_byte", "delta_zscore", "value_zscore", "norm_id"]

print("[로드] features_full.parquet ...")
t0 = time.time()
fd = pd.read_parquet(os.path.join(HERE, "..", "data_full", "features_full.parquet"))
print(f"  -> {fd.shape}, {time.time()-t0:.1f}s")

train = fd[fd["is_test_region"] == False].reset_index(drop=True)
test = fd[fd["is_test_region"] == True].reset_index(drop=True)
print(f"train={len(train):,} test={len(test):,} (실제 타임스탬프 기준 진짜 시간순 분리)")

# ---------- 경량 Mahalanobis 이상탐지기 (정상 데이터만 학습) ----------
class MahalanobisDetector:
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, percentile=99.9):
        self.mean_ = X[self.cols].mean().values
        cov = np.cov(X[self.cols].values, rowvar=False) + np.eye(len(self.cols)) * 1e-6
        self.inv_cov_ = np.linalg.inv(cov)
        d = self._dist(X)
        self.threshold_ = np.percentile(d, percentile)
        return self
    def _dist(self, X):
        diff = X[self.cols].values - self.mean_
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self.inv_cov_, diff))
    def predict(self, X):
        return (self._dist(X) > self.threshold_).astype(int)
    def save(self, path):
        np.savez(path, mean=self.mean_, inv_cov=self.inv_cov_, thr=self.threshold_)

train_normal = train[train["Label"] == "R"]
print(f"\n[Mahalanobis 학습] 정상 데이터 {len(train_normal):,}건")
t0 = time.time()
maha = MahalanobisDetector(FEATS).fit(train_normal)
print(f"  -> {time.time()-t0:.1f}s")
maha.save(os.path.join(HERE, "..", "models", "maha_detector_full.npz"))
maha_size_kb = os.path.getsize(os.path.join(HERE, "..", "models", "maha_detector_full.npz")) / 1024
print(f"  크기: {maha_size_kb:.2f} KB")

test_normal = test[test["Label"] == "R"]
fpr = maha.predict(test_normal).mean()
print(f"  정상 오탐율(FPR): {fpr:.2%}")
for atk in ["DoS", "Fuzzy", "gear", "RPM"]:
    sub = test[test["Label"] == atk]
    if len(sub) == 0:
        continue
    recall = maha.predict(sub).mean()
    print(f"  [{atk}] Mahalanobis 단독 탐지율: {recall:.1%} ({len(sub):,}건)")

# ---------- LightGBM (1단계, 알려진 공격) ----------
print("\n[LightGBM 학습]")
classes = sorted(train["Label"].unique())
c2i = {c: i for i, c in enumerate(classes)}
Xtr, ytr = train[FEATS], train["Label"].map(c2i)
Xte, yte = test[FEATS], test["Label"].map(c2i)
ds = lgb.Dataset(Xtr, label=ytr)
params = {"objective": "multiclass", "num_class": len(classes), "metric": "multi_logloss",
          "num_leaves": 15, "max_depth": 5, "learning_rate": 0.1, "verbose": -1}
t0 = time.time()
m = lgb.train(params, ds, num_boost_round=80)
print(f"  학습 시간: {time.time()-t0:.1f}s")
pred = np.argmax(m.predict(Xte), axis=1)
f1 = f1_score(yte, pred, average="macro")
m.save_model(os.path.join(HERE, "..", "models", "lgbm_full.txt"))
lgbm_size_kb = os.path.getsize(os.path.join(HERE, "..", "models", "lgbm_full.txt")) / 1024
print(f"  Macro F1: {f1:.4f}, 크기: {lgbm_size_kb:.1f}KB")
print(classification_report(yte, pred, target_names=classes))

# ---------- 하이브리드 e2e: 합산 메모리/지연시간 ----------
total_size_mb = (lgbm_size_kb + maha_size_kb) / 1024
print(f"\n[하이브리드 합산 메모리] {total_size_mb:.3f} MB (목표 <=1MB) -> {'통과' if total_size_mb<=1 else '미달'}")

row = Xte.iloc[[0]]
n_reps = 500
t0 = time.time()
for _ in range(n_reps):
    m.predict(row)
    maha.predict(test.iloc[[0]])
e2e_latency_ms = (time.time() - t0) / n_reps * 1000
print(f"[하이브리드 e2e 지연시간] {e2e_latency_ms:.3f} ms/message (목표 <=10ms)")

# ---------- 미학습 변형공격(Zero-day) 검증: leave-one-attack-out ----------
print("\n[미학습 변형공격(Zero-day) 검증, 시간순 test 기준]")
for held_out in ["DoS", "Fuzzy", "gear", "RPM"]:
    tr_sub = train[train["Label"] != held_out]
    classes2 = sorted(tr_sub["Label"].unique())
    c2i2 = {c: i for i, c in enumerate(classes2)}
    ds2 = lgb.Dataset(tr_sub[FEATS], label=tr_sub["Label"].map(c2i2))
    m2 = lgb.train({**params, "num_class": len(classes2)}, ds2, num_boost_round=80)

    te_atk = test[test["Label"] == held_out]
    if len(te_atk) == 0:
        continue
    stage2_catch = maha.predict(te_atk).mean()
    print(f"[held-out={held_out}] Mahalanobis(2단계)로 탐지: {stage2_catch:.1%} (정상 오탐 {fpr:.2%}, {len(te_atk):,}건)")

with open(os.path.join(HERE, "..", "results", "full_dataset_summary.txt"), "w") as f:
    f.write(f"lgbm_f1={f1:.4f}\nlgbm_size_kb={lgbm_size_kb:.1f}\nmaha_size_kb={maha_size_kb:.2f}\n")
    f.write(f"total_mb={total_size_mb:.3f}\ne2e_latency_ms={e2e_latency_ms:.3f}\nfpr={fpr:.4f}\n")

print("\n[완료] results/full_dataset_summary.txt 저장됨")
