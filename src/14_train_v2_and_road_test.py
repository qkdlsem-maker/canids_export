"""
1부: V2(ID-agnostic) 피처로 Car-Hacking 자체 성능이 V1과 비슷하게 유지되는지 확인
2부: 학습된 통계(idagnostic_stats.pkl, ID 키 없음)를 ROAD 데이터셋(전혀 다른 차량)에
     그대로 적용해서, 진짜로 일반화가 개선됐는지 검증 (이전엔 V1으로 FPR 100% 참패했음)

전제: 13_build_features_v2.py 실행 완료, road_data/attacks, road_data/ambient 존재
"""
import pandas as pd, numpy as np, lightgbm as lgb, time, os, re, glob, pickle
from sklearn.metrics import f1_score

HERE = os.path.dirname(__file__)
FEATS = ["freq_in_window", "unique_ids_in_window", "entropy", "mean_byte",
          "global_delta_zscore", "global_value_zscore"]

# ================= 1부: Car-Hacking 자체 재검증 (V2) =================
print("=" * 60)
print("1부: V2(ID-agnostic) 피처 - Car-Hacking 자체 성능 검증")
print("=" * 60)
fd = pd.read_parquet(os.path.join(HERE, "..", "data_full", "features_v2_idagnostic.parquet"))
train = fd[fd["is_test_region"] == False].reset_index(drop=True)
test = fd[fd["is_test_region"] == True].reset_index(drop=True)

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
m.save_model(os.path.join(HERE, "..", "models", "lgbm_v2.txt"))
print(f"[LightGBM V2] Macro F1: {f1:.4f} (V1은 1.0000이었음)")

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
    def predict(self, X): return (self._dist(X) > self.threshold_).astype(int)
    def save(self, path): np.savez(path, mean=self.mean_, inv_cov=self.inv_cov_, thr=self.threshold_)

train_normal = train[train["Label"] == "R"]
maha = MahalanobisDetector(FEATS).fit(train_normal)
maha.save(os.path.join(HERE, "..", "models", "maha_v2.npz"))

test_normal = test[test["Label"] == "R"]
fpr_v2 = maha.predict(test_normal).mean()
print(f"[Mahalanobis V2] 정상 오탐율(FPR): {fpr_v2:.2%} (V1은 0.02%였음)")
for atk in ["DoS", "Fuzzy", "gear", "RPM"]:
    sub = test[test["Label"] == atk]
    recall = maha.predict(sub).mean()
    print(f"  [{atk}] 탐지율: {recall:.1%}")

# ================= 2부: ROAD 교차검증 (진짜 목적) =================
print("\n" + "=" * 60)
print("2부: ROAD 데이터셋 교차검증 (V2 ID-agnostic 통계 재사용)")
print("=" * 60)

ROAD_ATTACKS = os.path.join(HERE, "..", "road_data", "attacks")
ROAD_AMBIENT = os.path.join(HERE, "..", "road_data", "ambient")

with open(os.path.join(HERE, "..", "models", "idagnostic_stats.pkl"), "rb") as f:
    STATS = pickle.load(f)
WINDOW = STATS["window"]
g_delta_mean, g_delta_std = STATS["global_delta_mean"], STATS["global_delta_std"]
byte_pos_mean, byte_pos_std = STATS["byte_pos_mean"], STATS["byte_pos_std"]

npz = np.load(os.path.join(HERE, "..", "models", "maha_v2.npz"))
MEAN_, INV_COV, THR = npz["mean"], npz["inv_cov"], float(npz["thr"])

LINE_RE = re.compile(r"\(([\d.]+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)")

def parse_log(path, max_lines=200000):
    rows = []
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            m = LINE_RE.search(line)
            if not m:
                continue
            cid = int(m.group(2), 16)
            hexdata = m.group(3)
            byts = [int(hexdata[j:j+2], 16) for j in range(0, min(len(hexdata), 16), 2)]
            byts = (byts + [0] * 8)[:8]
            rows.append((cid, byts))
    return rows

def payload_entropy(b):
    v, c = np.unique(b, return_counts=True)
    p = c / c.sum()
    return -np.sum(p * np.log2(p + 1e-12))

def extract_features_v2(rows):
    feats = []
    id_window, last_payload_per_id = [], {}
    for cid, pl in rows:
        pl = np.array(pl, dtype=float)
        id_window.append(cid)
        if len(id_window) > WINDOW:
            id_window.pop(0)
        freq_in_window = id_window.count(cid) / len(id_window)
        unique_ids_in_window = len(set(id_window))
        entropy = payload_entropy(pl)
        mean_byte = pl.mean()
        if cid in last_payload_per_id:
            delta = np.abs(pl - last_payload_per_id[cid]).sum()
        else:
            delta = 0.0
        last_payload_per_id[cid] = pl
        global_delta_zscore = (delta - g_delta_mean) / g_delta_std
        global_value_zscore = np.max(np.abs((pl - byte_pos_mean) / byte_pos_std))
        feats.append([freq_in_window, unique_ids_in_window, entropy, mean_byte,
                       global_delta_zscore, global_value_zscore])
    return np.array(feats)

def maha_flag(X):
    diff = X - MEAN_
    dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, INV_COV, diff))
    return dist > THR

def evaluate_file(path):
    rows = parse_log(path)
    if len(rows) < WINDOW + 1:
        return None
    X = extract_features_v2(rows)
    flags = maha_flag(X)
    return flags.mean(), len(rows)

print("\n[ROAD 공격 - 완전히 새로운 차량+공격유형]")
attack_results = []
for path in sorted(glob.glob(os.path.join(ROAD_ATTACKS, "*.log"))):
    name = os.path.basename(path)
    if "_masquerade" in name:
        continue
    result = evaluate_file(path)
    if result is None:
        continue
    recall, n = result
    attack_results.append((name, recall))
    print(f"[{name}] 탐지율: {recall:.1%} ({n:,}건)")

print("\n[ROAD 정상(ambient) - 오탐율(FPR)]")
fpr_results = []
for path in sorted(glob.glob(os.path.join(ROAD_AMBIENT, "*.log")))[:5]:
    name = os.path.basename(path)
    result = evaluate_file(path)
    if result is None:
        continue
    fpr, n = result
    fpr_results.append((name, fpr))
    print(f"[{name}] 오탐율: {fpr:.2%} ({n:,}건)")

print("\n" + "=" * 60)
print("요약: V1(ID-dependent)은 FPR 100%로 완전 실패했었음.")
if fpr_results:
    avg_fpr = np.mean([f for _, f in fpr_results])
    avg_recall = np.mean([r for _, r in attack_results]) if attack_results else 0
    print(f"V2(ID-agnostic) 결과: 평균 FPR={avg_fpr:.2%}, 평균 탐지율={avg_recall:.1%}")
