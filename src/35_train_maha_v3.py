"""
V3 Mahalanobis training

- ROAD 사용 금지
- HCRL train region의 NORMAL(R)만 fit
- threshold = train-normal Mahalanobis distance 99.9 percentile
- 모델 저장 후 HCRL test에서 sanity check
"""

import os
import glob
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))

PART_DIR = os.path.join(
    ROOT, "data_full", "features_v3_parts"
)
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

with open(
    os.path.join(MODEL_DIR, "v3_feature_config.pkl"), "rb"
) as f:
    CFG = pickle.load(f)

FEATS = CFG["feature_names"]

parts = sorted(glob.glob(
    os.path.join(PART_DIR, "*_features_v3.parquet")
))

if not parts:
    raise RuntimeError(
        "V3 parquet 없음. 먼저 python3 src/34_build_features_v3_temporal.py"
    )


# --------------------------------------------------
# 1. train-normal covariance를 streaming 계산
# --------------------------------------------------
count = 0
sum_x = np.zeros(len(FEATS), dtype=np.float64)
sum_xx = np.zeros((len(FEATS), len(FEATS)), dtype=np.float64)

print("=== V3 train-normal 통계 계산 ===")

for path in parts:
    df = pd.read_parquet(
        path,
        columns=FEATS + ["Label", "is_test_region"]
    )

    sub = df[
        (~df["is_test_region"]) &
        (df["Label"] == "R")
    ]

    X = sub[FEATS].to_numpy(dtype=np.float64)

    count += len(X)
    sum_x += X.sum(axis=0)
    sum_xx += X.T @ X

    print(
        os.path.basename(path),
        f"normal train={len(X):,}"
    )

if count < 100:
    raise RuntimeError("train normal 표본이 너무 적음")

mean = sum_x / count

cov = (
    sum_xx - count * np.outer(mean, mean)
) / (count - 1)

# 수치 안정성
ridge = 1e-6 * max(float(np.trace(cov) / len(FEATS)), 1.0)
cov = cov + np.eye(len(FEATS)) * ridge

# pinv 사용: temporal feature 상관 때문에 singular해도 안정적
inv_cov = np.linalg.pinv(cov)


def dist(X):
    d = X - mean
    q = np.einsum("ij,jk,ik->i", d, inv_cov, d)
    return np.sqrt(np.maximum(q, 0.0))


# --------------------------------------------------
# 2. threshold = HCRL train NORMAL의 99.9 percentile
# --------------------------------------------------
train_dists = []

for path in parts:
    df = pd.read_parquet(
        path,
        columns=FEATS + ["Label", "is_test_region"]
    )

    sub = df[
        (~df["is_test_region"]) &
        (df["Label"] == "R")
    ]

    X = sub[FEATS].to_numpy(dtype=np.float64)
    train_dists.append(dist(X))

train_dists = np.concatenate(train_dists)

threshold = float(np.percentile(train_dists, 99.9))

model_path = os.path.join(MODEL_DIR, "maha_v3_temporal.npz")

np.savez(
    model_path,
    mean=mean,
    inv_cov=inv_cov,
    thr=threshold,
    feature_names=np.asarray(FEATS),
)

print(f"\nthreshold(99.9%) = {threshold:.6f}")
print(f"저장: {model_path}")


# --------------------------------------------------
# 3. HCRL test sanity check
# --------------------------------------------------
lines = []
lines.append("=== HCRL V3 Mahalanobis sanity check ===")
lines.append(f"threshold={threshold:.6f}")
lines.append("")

normal_flags = []

for path in parts:
    df = pd.read_parquet(
        path,
        columns=FEATS + ["Label", "is_test_region"]
    )

    test = df[df["is_test_region"]]

    Xn = test[test["Label"] == "R"][FEATS].to_numpy(
        dtype=np.float64
    )

    if len(Xn):
        flags = dist(Xn) > threshold
        normal_flags.append(flags)

if normal_flags:
    fpr = np.concatenate(normal_flags).mean()
else:
    fpr = np.nan

lines.append(f"HCRL test normal FPR: {fpr:.4%}")
lines.append("")

for atk in ["DoS", "Fuzzy", "gear", "RPM"]:
    atk_flags = []

    for path in parts:
        df = pd.read_parquet(
            path,
            columns=FEATS + ["Label", "is_test_region"]
        )

        sub = df[
            (df["is_test_region"]) &
            (df["Label"] == atk)
        ]

        if len(sub):
            X = sub[FEATS].to_numpy(dtype=np.float64)
            atk_flags.append(dist(X) > threshold)

    if atk_flags:
        tpr = np.concatenate(atk_flags).mean()
        lines.append(f"{atk:8s} TPR: {tpr:.2%}")

result = "\n".join(lines)

print("\n" + result)

out = os.path.join(
    RESULT_DIR,
    "v3_hcrl_sanity_check.txt"
)

with open(out, "w", encoding="utf-8") as f:
    f.write(result + "\n")

print(f"\n결과 저장: {out}")
