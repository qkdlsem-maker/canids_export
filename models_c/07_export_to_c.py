"""
item: 실제 임베디드(C/ECU) 포팅
- LightGBM -> m2cgen으로 순수 C 코드 변환 (if-else 트리, 외부 라이브러리 불필요)
- Mahalanobis -> 평균벡터/역공분산행렬을 C 배열로 export, 순수 산술 연산만으로 구현
- 최종적으로 실제 gcc로 컴파일까지 검증 (동작 가능한 C 코드임을 증명)
"""
import lightgbm as lgb, m2cgen as m2c, numpy as np, os, pandas as pd

here = os.path.dirname(__file__)

# m2cgen은 raw Booster를 지원하지 않으므로, 동일한 설정으로 sklearn 래퍼(LGBMClassifier)를 재학습
fd = pd.read_parquet(os.path.join(here, "..", "data", "features_v3.parquet"))
FEATS = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy", "mean_byte", "delta_zscore", "value_zscore", "norm_id"]
train = fd[fd["is_test_region"] == 0]
classes_sorted = sorted(fd["Label"].unique())
c2i = {c: i for i, c in enumerate(classes_sorted)}

clf = lgb.LGBMClassifier(num_leaves=15, max_depth=5, learning_rate=0.1, n_estimators=80, verbose=-1)
clf.fit(train[FEATS], train["Label"].map(c2i))

c_code_lgbm = m2c.export_to_c(clf)
m = clf.booster_

npz = np.load(os.path.join(here, "..", "models", "maha_detector.npz"))
mean_, inv_cov, thr = npz["mean"], npz["inv_cov"], float(npz["thr"])
FEATS = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy", "mean_byte", "delta_zscore", "value_zscore", "norm_id"]
n = len(FEATS)

def c_array(name, arr):
    flat = arr.flatten()
    body = ", ".join(f"{v:.10f}" for v in flat)
    return f"static const double {name}[{len(flat)}] = {{{body}}};"

maha_c = f"""
/* Mahalanobis 이상탐지기 (정상 데이터만으로 학습됨, {n}개 피처) */
{c_array("MAHA_MEAN", mean_)}
{c_array("MAHA_INV_COV", inv_cov)}
static const double MAHA_THRESHOLD = {thr:.10f};

/* 0=정상, 1=이상(미지 공격 가능성) 반환 */
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

/* ===== 1단계: LightGBM (알려진 공격 분류, m2cgen 자동생성) ===== */"""

full_c = header + "\n" + c_code_lgbm + "\n" + maha_c + """
/* ===== 통합 판정 함수 =====
   x: 피처 배열 [freq_in_window, unique_ids_in_window, is_unknown_id,
                 entropy, mean_byte, delta_zscore, value_zscore, norm_id]
   반환: 0=R(정상), 1=DoS, 2=Fuzzy, 3=RPM, 4=gear, 5=미지의 이상패턴(Zero-day)
*/
int can_ids_predict(double *x) {
    double scores[5];
    score(x, scores);  /* m2cgen이 생성한 LightGBM 함수 */
    int best = 0;
    for (int i = 1; i < 5; i++) if (scores[i] > scores[best]) best = i;
    if (best != 0) return best;          /* 알려진 공격으로 분류됨 */
    if (maha_predict(x)) return 5;       /* 알려진 패턴은 아니지만 통계적으로 이상함 (Zero-day) */
    return 0;                            /* 정상 */
}
"""

out_path = os.path.join(here, "..", "models", "can_ids_embedded.c")
with open(out_path, "w") as f:
    f.write(full_c)
print("생성됨:", out_path, f"({os.path.getsize(out_path)/1024:.1f} KB 소스코드)")
