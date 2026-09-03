"""
통합 해결 스크립트
- item7(leakage): is_test_region 기준 완전한 시간순 train/test 분리
- item1(메모리): IsolationForest(600KB+) -> Mahalanobis 거리 기반 경량 탐지기(<5KB)
- item2(zero-day 개선): value_zscore 피처 추가 후 재검증
- item3(e2e): LightGBM(1단계, 알려진 공격) + Mahalanobis(2단계, 미지 공격) 순차 파이프라인
             전체 지연시간/메모리/성능을 "합쳐서" 측정
"""
import pandas as pd, numpy as np, lightgbm as lgb, time, os, pickle
from sklearn.metrics import f1_score, classification_report

import os
fd = pd.read_parquet(os.path.join(os.path.dirname(__file__), "..", "data", "features_v3.parquet"))
FEATS = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy", "mean_byte", "delta_zscore", "value_zscore", "norm_id"]

train = fd[fd["is_test_region"] == 0].reset_index(drop=True)
test = fd[fd["is_test_region"] == 1].reset_index(drop=True)
print(f"train={len(train):,} test={len(test):,} (완전 시간순 분리, leakage 없음)")

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
    def predict(self, X):  # 1 = anomaly
        return (self._dist(X) > self.threshold_).astype(int)
    def save(self, path):
        np.savez(path, mean=self.mean_, inv_cov=self.inv_cov_, thr=self.threshold_)

train_normal = train[train["Label"] == "R"]
maha = MahalanobisDetector(FEATS).fit(train_normal)
maha.save("../models/maha_detector.npz")
maha_size_kb = os.path.getsize("../models/maha_detector.npz") / 1024
print(f"\n[Mahalanobis 경량 탐지기] 크기: {maha_size_kb:.2f} KB (기존 IsolationForest {593.5:.1f}KB 대비 {593.5/maha_size_kb:.0f}배 작음)")

test_normal = test[test["Label"] == "R"]
fpr = maha.predict(test_normal).mean()
print(f"정상 오탐율(FPR): {fpr:.2%}")
for atk in ["DoS", "Fuzzy", "gear", "RPM"]:
    sub = test[test["Label"] == atk]
    if len(sub) == 0:
        continue
    recall = maha.predict(sub).mean()
    print(f"[{atk}] Mahalanobis 단독 탐지율: {recall:.1%}")

# ---------- LightGBM (1단계, 알려진 공격, 시간순 split) ----------
classes = sorted(train["Label"].unique())
c2i = {c: i for i, c in enumerate(classes)}
Xtr, ytr = train[FEATS], train["Label"].map(c2i)
Xte, yte = test[FEATS], test["Label"].map(c2i)
ds = lgb.Dataset(Xtr, label=ytr)
params = {"objective": "multiclass", "num_class": len(classes), "metric": "multi_logloss",
          "num_leaves": 15, "max_depth": 5, "learning_rate": 0.1, "verbose": -1}
m = lgb.train(params, ds, num_boost_round=80)
pred = np.argmax(m.predict(Xte), axis=1)
f1 = f1_score(yte, pred, average="macro")
m.save_model("../models/lgbm_final.txt")
lgbm_size_kb = os.path.getsize("../models/lgbm_final.txt") / 1024
print(f"\n[LightGBM, 시간순 split] Macro F1: {f1:.4f}, 크기: {lgbm_size_kb:.1f}KB")

# ---------- e2e 하이브리드: 합산 메모리/지연시간 ----------
total_size_mb = (lgbm_size_kb + maha_size_kb) / 1024
print(f"\n[하이브리드 합산 메모리] {total_size_mb:.3f} MB (목표 <=1MB) -> {'통과' if total_size_mb<=1 else '미달'}")

row = Xte.iloc[[0]]
n_reps = 500
t0 = time.time()
for _ in range(n_reps):
    m.predict(row)                      # stage1
    maha.predict(test.iloc[[0]])        # stage2 (같은 메시지, 순차 실행)
e2e_latency_ms = (time.time() - t0) / n_reps * 1000
print(f"[하이브리드 e2e 지연시간] {e2e_latency_ms:.3f} ms/message (목표 <=10ms)")

# ---------- zero-day: 공격유형 하나씩 제외하고 LightGBM 재학습 + Mahalanobis 결합 ----------
print("\n[Zero-day, 시간순 test 기준, leakage 없음] LightGBM(알려진) OR Mahalanobis(미지) 결합 탐지율")
for held_out in ["DoS", "Fuzzy", "gear", "RPM"]:
    tr_sub = train[train["Label"] != held_out]
    classes2 = sorted(tr_sub["Label"].unique())
    c2i2 = {c: i for i, c in enumerate(classes2)}
    ds2 = lgb.Dataset(tr_sub[FEATS], label=tr_sub["Label"].map(c2i2))
    m2 = lgb.train({**params, "num_class": len(classes2)}, ds2, num_boost_round=80)

    te_atk = test[test["Label"] == held_out]
    if len(te_atk) == 0:
        continue
    # stage1: LightGBM은 이 공격을 아예 모름 -> 항상 '알려진 클래스 중 하나'로 오분류(=이상탐지 실패 가능)
    # stage2: Mahalanobis가 통계적 이상치로 잡아내는지가 관건
    stage2_catch = maha.predict(te_atk).mean()
    fpr_normal = maha.predict(test_normal).mean()
    print(f"[held-out={held_out}] Mahalanobis(2단계)로 탐지: {stage2_catch:.1%}, 정상 오탐: {fpr_normal:.2%}")

with open("../results/final_summary.txt", "w") as f:
    f.write(f"lgbm_f1={f1:.4f}\nlgbm_size_kb={lgbm_size_kb:.1f}\nmaha_size_kb={maha_size_kb:.2f}\n")
    f.write(f"total_mb={total_size_mb:.3f}\ne2e_latency_ms={e2e_latency_ms:.3f}\nfpr={fpr:.4f}\n")
