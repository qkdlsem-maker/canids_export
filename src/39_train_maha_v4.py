"""
V4 Mahalanobis

- HCRL train normal만 fit
- threshold = train-normal 99.9 percentile
- ROAD는 사용하지 않음
"""

import os
import glob
import pickle

import numpy as np
import pandas as pd


HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(
    os.path.join(HERE, "..")
)

PART_DIR = os.path.join(
    ROOT,
    "data_full",
    "features_v4_parts"
)

MODEL_DIR = os.path.join(
    ROOT,
    "models"
)

RESULT_DIR = os.path.join(
    ROOT,
    "results"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


with open(
    os.path.join(
        MODEL_DIR,
        "v4_feature_config.pkl"
    ),
    "rb"
) as f:
    CFG = pickle.load(f)

FEATS = CFG[
    "feature_names"
]


parts = sorted(
    glob.glob(
        os.path.join(
            PART_DIR,
            "*_features_v4.parquet"
        )
    )
)

if not parts:
    raise RuntimeError(
        "V4 parquet 없음. "
        "38번 먼저 실행"
    )


# ==========================================
# Streaming covariance
# ==========================================

d = len(FEATS)

count = 0

sum_x = np.zeros(
    d,
    dtype=np.float64
)

sum_xx = np.zeros(
    (d, d),
    dtype=np.float64
)


print(
    "=== V4 train-normal fit ==="
)


for path in parts:

    df = pd.read_parquet(
        path,
        columns=
        FEATS
        + [
            "Label",
            "is_test_region"
        ]
    )

    sub = df[
        (~df["is_test_region"])
        &
        (df["Label"] == "R")
    ]

    X = sub[
        FEATS
    ].to_numpy(
        dtype=np.float64
    )

    count += len(X)

    sum_x += X.sum(
        axis=0
    )

    sum_xx += X.T @ X

    print(
        os.path.basename(path),
        f"N={len(X):,}"
    )


if count < 100:
    raise RuntimeError(
        "normal train samples 부족"
    )


mean = (
    sum_x / count
)


cov = (
    sum_xx
    - count
    * np.outer(
        mean,
        mean
    )
) / (
    count - 1
)


ridge = (
    1e-6
    * max(
        float(
            np.trace(cov)
            / d
        ),
        1.0
    )
)


cov += (
    np.eye(d)
    * ridge
)


inv_cov = np.linalg.pinv(
    cov
)


def distance(X):

    diff = X - mean

    q = np.einsum(
        "ij,jk,ik->i",
        diff,
        inv_cov,
        diff
    )

    return np.sqrt(
        np.maximum(
            q,
            0.0
        )
    )


# ==========================================
# Frozen threshold
# ==========================================

train_dist = []


for path in parts:

    df = pd.read_parquet(
        path,
        columns=
        FEATS
        + [
            "Label",
            "is_test_region"
        ]
    )

    sub = df[
        (~df["is_test_region"])
        &
        (df["Label"] == "R")
    ]

    X = sub[
        FEATS
    ].to_numpy(
        dtype=np.float64
    )

    train_dist.append(
        distance(X)
    )


train_dist = np.concatenate(
    train_dist
)


threshold = float(
    np.percentile(
        train_dist,
        99.9
    )
)


model_path = os.path.join(
    MODEL_DIR,
    "maha_v4_byte_temporal.npz"
)


np.savez(
    model_path,
    mean=mean,
    inv_cov=inv_cov,
    thr=threshold,
    feature_names=np.asarray(
        FEATS
    )
)


print(
    f"\nthreshold="
    f"{threshold:.6f}"
)

print(
    f"saved: "
    f"{model_path}"
)


# ==========================================
# HCRL sanity check
# ==========================================

lines = []

lines.append(
    "=== HCRL V4 Mahalanobis sanity check ==="
)

lines.append(
    f"threshold={threshold:.6f}"
)

lines.append("")


normal_flags = []


for path in parts:

    df = pd.read_parquet(
        path,
        columns=
        FEATS
        + [
            "Label",
            "is_test_region"
        ]
    )

    sub = df[
        (df["is_test_region"])
        &
        (df["Label"] == "R")
    ]

    if len(sub) == 0:
        continue

    X = sub[
        FEATS
    ].to_numpy(
        dtype=np.float64
    )

    normal_flags.append(
        distance(X)
        > threshold
    )


if normal_flags:

    fpr = float(
        np.concatenate(
            normal_flags
        ).mean()
    )

else:

    fpr = np.nan


lines.append(
    f"HCRL test normal FPR: "
    f"{fpr:.4%}"
)


for attack in [
    "DoS",
    "Fuzzy",
    "gear",
    "RPM",
]:

    flags_all = []

    for path in parts:

        df = pd.read_parquet(
            path,
            columns=
            FEATS
            + [
                "Label",
                "is_test_region"
            ]
        )

        sub = df[
            (df["is_test_region"])
            &
            (df["Label"] == attack)
        ]

        if len(sub) == 0:
            continue

        X = sub[
            FEATS
        ].to_numpy(
            dtype=np.float64
        )

        flags_all.append(
            distance(X)
            > threshold
        )

    if flags_all:

        tpr = float(
            np.concatenate(
                flags_all
            ).mean()
        )

        lines.append(
            f"{attack:8s} "
            f"TPR: {tpr:.2%}"
        )


result = "\n".join(lines)

print(
    "\n" + result
)


out = os.path.join(
    RESULT_DIR,
    "v4_hcrl_sanity_check.txt"
)


with open(
    out,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        result + "\n"
    )


print(
    f"\nresult: {out}"
)
