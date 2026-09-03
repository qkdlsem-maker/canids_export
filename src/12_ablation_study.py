"""
Ablation Study: LightGBM 단독 vs Mahalanobis 단독 vs 제안한 Hybrid 구조를
(1) 알려진 공격 탐지율, (2) 미학습 변형공격(Zero-day) 탐지율, (3) 공격유형 라벨링 가능여부,
(4) 추론 지연시간 4가지 축에서 비교. 전체 데이터셋(features_full.parquet) 기준.

전제: 11_train_full.py가 먼저 실행되어 models/lgbm_full.txt, models/maha_detector_full.npz 가 있어야 함.
"""
import pandas as pd, numpy as np, lightgbm as lgb, time, os
from sklearn.metrics import recall_score

HERE = os.path.dirname(__file__)
FEATS = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy",
          "mean_byte", "delta_zscore", "value_zscore", "norm_id"]
ATTACKS = ["DoS", "Fuzzy", "gear", "RPM"]

print("[로드] features_full.parquet ...")
fd = pd.read_parquet(os.path.join(HERE, "..", "data_full", "features_full.parquet"))
train = fd[fd["is_test_region"] == False].reset_index(drop=True)
test = fd[fd["is_test_region"] == True].reset_index(drop=True)

classes = sorted(train["Label"].unique())
c2i = {c: i for i, c in enumerate(classes)}

# ---------- 기존 모델 로드 (11_train_full.py 결과물) ----------
lgbm = lgb.Booster(model_file=os.path.join(HERE, "..", "models", "lgbm_full.txt"))
npz = np.load(os.path.join(HERE, "..", "models", "maha_detector_full.npz"))
maha_mean, maha_inv_cov, maha_thr = npz["mean"], npz["inv_cov"], float(npz["thr"])

def maha_predict(X):
    diff = X[FEATS].values - maha_mean
    dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, maha_inv_cov, diff))
    return (dist > maha_thr).astype(int)

# ================= (1) 알려진 공격 탐지율 =================
print("\n=== (1) 알려진 공격 탐지율(Recall) ===")
Xte = test[FEATS]
lgbm_pred = np.argmax(lgbm.predict(Xte), axis=1)
i2c = {v: k for k, v in c2i.items()}
lgbm_pred_label = np.array([i2c[p] for p in lgbm_pred])
maha_pred = maha_predict(test)

results_known = {}
for atk in ATTACKS:
    mask = test["Label"] == atk
    n = mask.sum()
    lgbm_recall = (lgbm_pred_label[mask.values] == atk).mean()  # LightGBM: 정확한 유형까지 맞춰야 인정
    maha_recall = maha_pred[mask.values].mean()                  # Mahalanobis: 이상치로만 잡으면 인정(유형 라벨 불가)
    hybrid_recall = lgbm_recall  # 하이브리드 1단계 = LightGBM과 동일 (알려진 공격은 1단계에서 즉시 분류)
    results_known[atk] = (lgbm_recall, maha_recall, hybrid_recall, n)
    print(f"[{atk}] LightGBM 단독: {lgbm_recall:.1%} | Mahalanobis 단독: {maha_recall:.1%} | Hybrid: {hybrid_recall:.1%} ({n:,}건)")

# ================= (2) 미학습 변형공격(Zero-day) 탐지율 =================
print("\n=== (2) 미학습 변형공격(Zero-day) 탐지율 (Leave-One-Attack-Out) ===")
params = {"objective": "multiclass", "metric": "multi_logloss",
          "num_leaves": 15, "max_depth": 5, "learning_rate": 0.1, "verbose": -1}
results_zeroday = {}
for held_out in ATTACKS:
    tr_sub = train[train["Label"] != held_out]
    classes2 = sorted(tr_sub["Label"].unique())
    c2i2 = {c: i for i, c in enumerate(classes2)}
    ds2 = lgb.Dataset(tr_sub[FEATS], label=tr_sub["Label"].map(c2i2))
    m2 = lgb.train({**params, "num_class": len(classes2)}, ds2, num_boost_round=80)

    te_atk = test[test["Label"] == held_out]
    n = len(te_atk)
    # LightGBM 단독: held-out 유형은 학습에 없으므로, "R이 아닌 다른 클래스로라도 튀었는지"를 탐지 성공으로 관대하게 카운트
    pred2 = np.argmax(m2.predict(te_atk[FEATS]), axis=1)
    i2c2 = {v: k for k, v in c2i2.items()}
    pred2_label = np.array([i2c2[p] for p in pred2])
    lgbm_zd_recall = (pred2_label != "R").mean()

    maha_zd_recall = maha_pred[(test["Label"] == held_out).values].mean()
    hybrid_zd_recall = maha_zd_recall  # 하이브리드 2단계 = Mahalanobis와 동일 (미지 공격은 2단계가 담당)

    results_zeroday[held_out] = (lgbm_zd_recall, maha_zd_recall, hybrid_zd_recall, n)
    print(f"[held-out={held_out}] LightGBM 단독: {lgbm_zd_recall:.1%} | Mahalanobis 단독: {maha_zd_recall:.1%} | Hybrid: {hybrid_zd_recall:.1%} ({n:,}건)")

# ================= (3) 공격유형 라벨링 가능 여부 =================
print("\n=== (3) 공격유형 세부 라벨링 가능 여부 ===")
print("LightGBM 단독: 가능 (DoS/Fuzzy/gear/RPM 구분)")
print("Mahalanobis 단독: 불가능 (이상치 여부만 판단, 유형 구분 못함)")
print("Hybrid: 알려진 공격은 유형까지, 미지 공격은 '이상 있음'까지 라벨링 가능")

# ================= (4) 추론 지연시간 =================
print("\n=== (4) 추론 지연시간 (단건, 500회 평균) ===")
row = Xte.iloc[[0]]
row_full = test.iloc[[0]]
n_reps = 500

t0 = time.time()
for _ in range(n_reps):
    lgbm.predict(row)
lgbm_only_ms = (time.time() - t0) / n_reps * 1000

t0 = time.time()
for _ in range(n_reps):
    maha_predict(row_full)
maha_only_ms = (time.time() - t0) / n_reps * 1000

t0 = time.time()
for _ in range(n_reps):
    lgbm.predict(row)
    maha_predict(row_full)
hybrid_ms = (time.time() - t0) / n_reps * 1000

print(f"LightGBM 단독: {lgbm_only_ms:.3f}ms")
print(f"Mahalanobis 단독: {maha_only_ms:.3f}ms")
print(f"Hybrid(순차 실행): {hybrid_ms:.3f}ms")

# ================= 요약 저장 =================
out_path = os.path.join(HERE, "..", "results", "ablation_study.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("=== Ablation Study: LightGBM 단독 vs Mahalanobis 단독 vs Hybrid ===\n\n")
    f.write("(1) 알려진 공격 탐지율(Recall)\n")
    for atk, (l, mh, hy, n) in results_known.items():
        f.write(f"  [{atk}] LightGBM={l:.1%} Mahalanobis={mh:.1%} Hybrid={hy:.1%} (n={n:,})\n")
    f.write("\n(2) 미학습 변형공격(Zero-day) 탐지율\n")
    for atk, (l, mh, hy, n) in results_zeroday.items():
        f.write(f"  [held-out={atk}] LightGBM={l:.1%} Mahalanobis={mh:.1%} Hybrid={hy:.1%} (n={n:,})\n")
    f.write("\n(3) 공격유형 세부 라벨링: LightGBM=가능, Mahalanobis=불가능, Hybrid=부분 가능\n")
    f.write(f"\n(4) 추론 지연시간: LightGBM={lgbm_only_ms:.3f}ms, Mahalanobis={maha_only_ms:.3f}ms, Hybrid={hybrid_ms:.3f}ms\n")

print(f"\n[완료] {out_path} 저장됨")
