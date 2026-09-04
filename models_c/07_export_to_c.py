import lightgbm as lgb, m2cgen as m2c, numpy as np, os, pandas as pd

here = os.path.dirname(__file__)

fd = pd.read_parquet(os.path.join(here, "..", "data_full", "features_full.parquet"))
FEATS = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy", "mean_byte", "delta_zscore", "value_zscore", "norm_id"]
train = fd[fd["is_test_region"] == False]
classes_sorted = sorted(fd["Label"].unique())
c2i = {c: i for i, c in enumerate(classes_sorted)}
print("classes_sorted:", classes_sorted)
print("train size:", len(train))

clf = lgb.LGBMClassifier(num_leaves=15, max_depth=5, learning_rate=0.1, n_estimators=80, verbose=-1)
clf.fit(train[FEATS], train["Label"].map(c2i))

c_code_lgbm = m2c.export_to_c(clf)
m = clf.booster_

train_normal = train[train["Label"] == "R"]
mean_ = train_normal[FEATS].mean().values
cov = np.cov(train_normal[FEATS].values, rowvar=False) + np.eye(len(FEATS)) * 1e-6
inv_cov = np.linalg.inv(cov)
diff = train_normal[FEATS].values - mean_
dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))
thr = np.percentile(dist, 99.9)
n = len(FEATS)

def c_array(name, arr):
    flat = arr.flatten()
    body = ", ".join(f"{v:.10f}" for v in flat)
    return f"static const double {name}[{len(flat)}] = {{{body}}};"

maha_c = f"""
/* Mahalanobis 이상탐지기 (전체 데이터셋 정상 데이터로 학습됨, {n}개 피처) */
{c_array("MAHA_MEAN", mean_)}
{c_array("MAHA_INV_COV", inv_cov)}
static const double MAHA_THRESHOLD = {thr:.10f};

int maha_predict(double *x) {{
    double diff[{n}];
    for (int i = 0; i < {n}; i++) diff[i] = x[i] - MAHA_MEAN[i];
    double dist2 = 0.0;
    for (int i = 0; i < {n}; i++) {{
        double s = 0.0;
        for (int j = 0; j < {n}; j++) s += MAHA_INV_COV[i * {n} + j] * diff[j];
        dist2 += diff[i] * s;
    }}
    double dist = sqrt(dist2);
    return dist > MAHA_THRESHOLD ? 1 : 0;
}}
"""

header = """#include <math.h>
#include <string.h>

/* ===== 1단계: LightGBM (알려진 공격 분류, m2cgen 자동생성, 전체 데이터셋 기준) ===== */"""

NORMAL_IDX = classes_sorted.index("R")
OUTPUT_ORDER = ["R", "DoS", "Fuzzy", "RPM", "gear"]
code_map = [OUTPUT_ORDER.index(c) for c in classes_sorted]
code_map_c = ", ".join(str(x) for x in code_map)

full_c = header + "\n" + c_code_lgbm + "\n" + maha_c + f"""
#define NORMAL_IDX {NORMAL_IDX}
static const int CODE_MAP[5] = {{{code_map_c}}};

int can_ids_predict(double *x) {{
    double scores[5];
    score(x, scores);
    int best = 0;
    for (int i = 1; i < 5; i++) if (scores[i] > scores[best]) best = i;
    if (best != NORMAL_IDX) return CODE_MAP[best];
    if (maha_predict(x)) return 5;
    return 0;
}}
"""

out_path = os.path.join(here, "can_ids_embedded_full.c")
with open(out_path, "w") as f:
    f.write(full_c)
print("생성됨:", out_path, f"({os.path.getsize(out_path)/1024:.1f} KB 소스코드)")
