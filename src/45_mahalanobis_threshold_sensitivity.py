"""
45_mahalanobis_threshold_sensitivity.py

Final V2 Mahalanobis threshold sensitivity analysis.

PURPOSE
-------
Test whether the frozen V2 result depends excessively on the single
99.9th-percentile Mahalanobis threshold.

CRITICAL PROTOCOL
-----------------
- Mean/covariance: frozen V2 model (HCRL train-normal only)
- Threshold candidates: HCRL train-normal distances ONLY
- ROAD is NEVER used to calculate or select a threshold
- 99.9 percentile remains the final frozen operating point
- Other percentiles are sensitivity-analysis points only

Outputs:
  results/f7_mahalanobis_threshold_sensitivity.csv
  results/f7_mahalanobis_threshold_sensitivity.txt
"""

import os
import re
import glob
import json
import pickle

import numpy as np
import pandas as pd


HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))

FEATURE_PATH = os.path.join(
    ROOT,
    "data_full",
    "features_v2_idagnostic.parquet",
)

MODEL_PATH = os.path.join(
    ROOT,
    "models",
    "maha_v2.npz",
)

STATS_PATH = os.path.join(
    ROOT,
    "models",
    "idagnostic_stats.pkl",
)

ROAD_ATTACKS = os.path.join(
    ROOT,
    "road_data",
    "attacks",
)

ROAD_AMBIENT = os.path.join(
    ROOT,
    "road_data",
    "ambient",
)

METADATA_PATH = os.path.join(
    ROAD_ATTACKS,
    "capture_metadata.json",
)

OUT_CSV = os.path.join(
    ROOT,
    "results",
    "f7_mahalanobis_threshold_sensitivity.csv",
)

OUT_TXT = os.path.join(
    ROOT,
    "results",
    "f7_mahalanobis_threshold_sensitivity.txt",
)


FEATS = [
    "freq_in_window",
    "unique_ids_in_window",
    "entropy",
    "mean_byte",
    "global_delta_zscore",
    "global_value_zscore",
]

PERCENTILES = [
    99.00,
    99.50,
    99.70,
    99.80,
    99.90,
    99.95,
    99.99,
]

FINAL_PERCENTILE = 99.90


# =========================================================
# Frozen V2 model/statistics
# =========================================================

npz = np.load(MODEL_PATH)

MEAN_ = npz["mean"]
INV_COV = npz["inv_cov"]
SAVED_THR = float(npz["thr"])


with open(STATS_PATH, "rb") as f:
    STATS = pickle.load(f)

WINDOW = STATS["window"]

g_delta_mean = STATS["global_delta_mean"]
g_delta_std = STATS["global_delta_std"]

byte_pos_mean = STATS["byte_pos_mean"]
byte_pos_std = STATS["byte_pos_std"]


with open(METADATA_PATH, "r") as f:
    METADATA = json.load(f)


def maha_dist(X):
    X = np.asarray(X, dtype=float)
    diff = X - MEAN_

    return np.sqrt(
        np.einsum(
            "ij,jk,ik->i",
            diff,
            INV_COV,
            diff,
        )
    )


# =========================================================
# HCRL distances
# =========================================================

print("=" * 90)
print("Loading HCRL V2 feature dataset...")
print("=" * 90)

fd = pd.read_parquet(FEATURE_PATH)

train = fd[
    fd["is_test_region"] == False
].reset_index(drop=True)

test = fd[
    fd["is_test_region"] == True
].reset_index(drop=True)

train_normal = train[
    train["Label"] == "R"
].reset_index(drop=True)

test_normal = test[
    test["Label"] == "R"
].reset_index(drop=True)


train_normal_dist = maha_dist(
    train_normal[FEATS].values
)

hcrl_test_normal_dist = maha_dist(
    test_normal[FEATS].values
)


# HCRL attack distances, useful as an internal sanity check.
HCRL_ATTACKS = [
    "DoS",
    "Fuzzy",
    "gear",
    "RPM",
]

hcrl_attack_dists = {}

for attack in HCRL_ATTACKS:
    sub = test[
        test["Label"] == attack
    ]

    if len(sub) > 0:
        hcrl_attack_dists[attack] = maha_dist(
            sub[FEATS].values
        )


# =========================================================
# Verify exact frozen 99.9 threshold
# =========================================================

recomputed_999 = float(
    np.percentile(
        train_normal_dist,
        FINAL_PERCENTILE,
    )
)

threshold_diff = abs(
    recomputed_999 - SAVED_THR
)

print()
print("=== Frozen threshold audit ===")
print(
    f"Saved maha_v2.npz threshold : "
    f"{SAVED_THR:.10f}"
)
print(
    f"Recomputed HCRL 99.9%       : "
    f"{recomputed_999:.10f}"
)
print(
    f"Absolute difference         : "
    f"{threshold_diff:.12g}"
)

if not np.isclose(
    recomputed_999,
    SAVED_THR,
    rtol=1e-6,
    atol=1e-6,
):
    raise RuntimeError(
        "ERROR: Recomputed 99.9 percentile does not "
        "match frozen maha_v2.npz threshold. "
        "Stop sensitivity analysis."
    )

print("[OK] Frozen 99.9% threshold reproduced exactly.")


# =========================================================
# ROAD parsing / V2 feature extraction
# Same definitions as scripts 14 / 33.
# =========================================================

LINE_RE = re.compile(
    r"\(([\d.]+)\)\s+"
    r"\S+\s+"
    r"([0-9A-Fa-f]+)#"
    r"([0-9A-Fa-f]*)"
)


def parse_log(path):
    rows = []

    with open(
        path,
        "r",
        errors="ignore",
    ) as f:

        for line in f:
            m = LINE_RE.search(line)

            if not m:
                continue

            ts = float(m.group(1))
            cid = int(m.group(2), 16)

            hexdata = m.group(3)

            byts = [
                int(
                    hexdata[j:j + 2],
                    16,
                )
                for j in range(
                    0,
                    min(len(hexdata), 16),
                    2,
                )
            ]

            byts = (
                byts + [0] * 8
            )[:8]

            rows.append(
                (
                    ts,
                    cid,
                    byts,
                )
            )

    return rows


def payload_entropy(b):
    v, c = np.unique(
        b,
        return_counts=True,
    )

    p = c / c.sum()

    return -np.sum(
        p * np.log2(p + 1e-12)
    )


def feats_v2(rows):
    out = []
    win = []
    last = {}

    for _, cid, pl in rows:

        pl = np.array(
            pl,
            dtype=float,
        )

        win.append(cid)

        if len(win) > WINDOW:
            win.pop(0)

        freq = (
            win.count(cid)
            / len(win)
        )

        unique = len(
            set(win)
        )

        entropy = payload_entropy(pl)
        mean_byte = pl.mean()

        if cid in last:
            delta = np.abs(
                pl - last[cid]
            ).sum()
        else:
            delta = 0.0

        last[cid] = pl

        delta_z = (
            delta - g_delta_mean
        ) / g_delta_std

        value_z = np.max(
            np.abs(
                (
                    pl - byte_pos_mean
                )
                / byte_pos_std
            )
        )

        out.append(
            [
                freq,
                unique,
                entropy,
                mean_byte,
                delta_z,
                value_z,
            ]
        )

    return np.asarray(
        out,
        dtype=float,
    )


# =========================================================
# Precompute ROAD distances ONCE.
#
# Threshold does not affect feature extraction or distance.
# This guarantees the only changing variable in the sweep
# is the threshold.
# =========================================================

print()
print("=" * 90)
print("Precomputing frozen ROAD V2 distances...")
print("=" * 90)


road_attack_sets = []

for path in sorted(
    glob.glob(
        os.path.join(
            ROAD_ATTACKS,
            "*.log",
        )
    )
):

    name = os.path.basename(path)

    if "_masquerade" in name:
        continue

    meta = METADATA.get(
        name.replace(".log", ""),
        {},
    )

    interval = meta.get(
        "injection_interval"
    )

    injection_id = meta.get(
        "injection_id"
    )

    if interval is None:
        continue

    rows = parse_log(path)

    if len(rows) < WINDOW + 1:
        continue

    X = feats_v2(rows)
    dists = maha_dist(X)

    t0 = rows[0][0]

    rel = np.asarray(
        [
            r[0] - t0
            for r in rows
        ]
    )

    ids = np.asarray(
        [
            r[1]
            for r in rows
        ]
    )

    mask = (
        (rel >= interval[0])
        & (rel <= interval[1])
    )

    if (
        injection_id
        and injection_id != "XXX"
    ):
        mask = mask & (
            ids
            == int(
                injection_id,
                16,
            )
        )

    if mask.sum() == 0:
        continue

    road_attack_sets.append(
        {
            "name": name,
            "dist": dists[mask],
            "n": int(mask.sum()),
        }
    )

    print(
        f"[attack] {name:40s} "
        f"n={int(mask.sum()):,}"
    )


road_ambient_sets = []

ambient_paths = sorted(
    glob.glob(
        os.path.join(
            ROAD_AMBIENT,
            "*.log",
        )
    )
)[:5]

for path in ambient_paths:

    name = os.path.basename(path)

    rows = parse_log(path)

    if len(rows) < WINDOW + 1:
        continue

    X = feats_v2(rows)
    dists = maha_dist(X)

    road_ambient_sets.append(
        {
            "name": name,
            "dist": dists,
            "n": len(dists),
        }
    )

    print(
        f"[ambient] {name:39s} "
        f"n={len(dists):,}"
    )


if not road_attack_sets:
    raise RuntimeError(
        "No ROAD attack evaluation sets found."
    )

if not road_ambient_sets:
    raise RuntimeError(
        "No ROAD ambient evaluation sets found."
    )


# =========================================================
# Sweep
# =========================================================

print()
print("=" * 110)
print("Mahalanobis V2 Threshold Sensitivity")
print("=" * 110)

header = (
    f"{'Pct':>7s} "
    f"{'Threshold':>12s} "
    f"{'HCRL_FPR':>11s} "
    f"{'HCRL_ATK':>11s} "
    f"{'ROAD_FPR':>11s} "
    f"{'ROAD_TPR':>11s} "
    f"{'ROAD_BA':>11s}"
)

print(header)
print("-" * len(header))


summary_rows = []
per_attack_rows = []


for pct in PERCENTILES:

    thr = float(
        np.percentile(
            train_normal_dist,
            pct,
        )
    )

    # ---------------- HCRL ----------------

    hcrl_fpr = np.mean(
        hcrl_test_normal_dist
        > thr
    )

    hcrl_recalls = {}

    for name, dist in (
        hcrl_attack_dists.items()
    ):
        hcrl_recalls[name] = np.mean(
            dist > thr
        )

    hcrl_attack_mean = np.mean(
        list(
            hcrl_recalls.values()
        )
    )


    # ---------------- ROAD attacks ----------------

    road_recalls = []

    for item in road_attack_sets:

        recall = np.mean(
            item["dist"] > thr
        )

        road_recalls.append(
            recall
        )

        per_attack_rows.append(
            {
                "percentile": pct,
                "threshold": thr,
                "dataset": "ROAD_attack",
                "name": item["name"],
                "n": item["n"],
                "rate": recall,
            }
        )

    # Preserve script 33 convention:
    # mean of per-file attack TPRs.
    road_tpr = np.mean(
        road_recalls
    )


    # ---------------- ROAD ambient ----------------

    road_fprs = []

    for item in road_ambient_sets:

        fpr = np.mean(
            item["dist"] > thr
        )

        road_fprs.append(
            fpr
        )

        per_attack_rows.append(
            {
                "percentile": pct,
                "threshold": thr,
                "dataset": "ROAD_ambient",
                "name": item["name"],
                "n": item["n"],
                "rate": fpr,
            }
        )

    # Preserve script 33 convention:
    # mean of per-file FPRs.
    road_fpr = np.mean(
        road_fprs
    )

    road_tnr = 1.0 - road_fpr

    road_ba = (
        road_tpr + road_tnr
    ) / 2.0


    is_final = bool(
        np.isclose(
            pct,
            FINAL_PERCENTILE,
        )
    )


    summary_rows.append(
        {
            "percentile": pct,
            "threshold": thr,
            "is_frozen_final": is_final,
            "hcrl_test_normal_fpr": hcrl_fpr,
            "hcrl_mean_attack_tpr": hcrl_attack_mean,
            "road_mean_fpr": road_fpr,
            "road_mean_tpr": road_tpr,
            "road_tnr": road_tnr,
            "road_balanced_accuracy": road_ba,
        }
    )


    mark = "*" if is_final else " "

    print(
        f"{pct:6.2f}{mark} "
        f"{thr:12.6f} "
        f"{hcrl_fpr:10.4%} "
        f"{hcrl_attack_mean:10.4%} "
        f"{road_fpr:10.4%} "
        f"{road_tpr:10.4%} "
        f"{road_ba:10.4%}"
    )


# =========================================================
# Save CSV
# =========================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUT_CSV,
    index=False,
)


detail_path = os.path.join(
    ROOT,
    "results",
    "f7_mahalanobis_threshold_sensitivity_detail.csv",
)

pd.DataFrame(
    per_attack_rows
).to_csv(
    detail_path,
    index=False,
)


# =========================================================
# Local-neighborhood robustness around frozen 99.9
# =========================================================

final_idx = next(
    i
    for i, r in enumerate(summary_rows)
    if r["is_frozen_final"]
)

final_row = summary_rows[
    final_idx
]


neighbor_pcts = [
    99.70,
    99.80,
    99.90,
    99.95,
]

neighbor_rows = [
    r
    for r in summary_rows
    if r["percentile"]
    in neighbor_pcts
]


road_fpr_range = (
    max(
        r["road_mean_fpr"]
        for r in neighbor_rows
    )
    - min(
        r["road_mean_fpr"]
        for r in neighbor_rows
    )
)

road_tpr_range = (
    max(
        r["road_mean_tpr"]
        for r in neighbor_rows
    )
    - min(
        r["road_mean_tpr"]
        for r in neighbor_rows
    )
)

road_ba_range = (
    max(
        r["road_balanced_accuracy"]
        for r in neighbor_rows
    )
    - min(
        r["road_balanced_accuracy"]
        for r in neighbor_rows
    )
)


# =========================================================
# Text report
# =========================================================

lines = []

lines.append(
    "=== Mahalanobis V2 Threshold Sensitivity ==="
)

lines.append("")

lines.append(
    "[Protocol]"
)

lines.append(
    "Mean/covariance: frozen HCRL V2 train-normal model"
)

lines.append(
    "Threshold candidates: HCRL train-normal distance percentiles only"
)

lines.append(
    "ROAD used for threshold calculation: NO"
)

lines.append(
    "ROAD used for threshold selection/tuning: NO"
)

lines.append(
    "Frozen final operating point: 99.90 percentile"
)

lines.append("")

lines.append(
    "[Frozen threshold audit]"
)

lines.append(
    f"Saved threshold       : "
    f"{SAVED_THR:.10f}"
)

lines.append(
    f"Recomputed 99.90%     : "
    f"{recomputed_999:.10f}"
)

lines.append(
    f"Absolute difference   : "
    f"{threshold_diff:.12g}"
)

lines.append("")

lines.append(
    "Pct, Threshold, HCRL_FPR, HCRL_attack_TPR, "
    "ROAD_FPR, ROAD_TPR, ROAD_BA"
)

for r in summary_rows:

    mark = " [FINAL]" \
        if r["is_frozen_final"] \
        else ""

    lines.append(
        f"{r['percentile']:.2f}, "
        f"{r['threshold']:.6f}, "
        f"{r['hcrl_test_normal_fpr']:.6%}, "
        f"{r['hcrl_mean_attack_tpr']:.6%}, "
        f"{r['road_mean_fpr']:.6%}, "
        f"{r['road_mean_tpr']:.6%}, "
        f"{r['road_balanced_accuracy']:.6%}"
        f"{mark}"
    )


lines.append("")
lines.append(
    "[Frozen 99.90% operating point]"
)

lines.append(
    f"HCRL normal FPR : "
    f"{final_row['hcrl_test_normal_fpr']:.6%}"
)

lines.append(
    f"HCRL attack TPR : "
    f"{final_row['hcrl_mean_attack_tpr']:.6%}"
)

lines.append(
    f"ROAD normal FPR : "
    f"{final_row['road_mean_fpr']:.6%}"
)

lines.append(
    f"ROAD attack TPR : "
    f"{final_row['road_mean_tpr']:.6%}"
)

lines.append(
    f"ROAD TNR        : "
    f"{final_row['road_tnr']:.6%}"
)

lines.append(
    f"ROAD BA         : "
    f"{final_row['road_balanced_accuracy']:.6%}"
)


lines.append("")
lines.append(
    "[Local sensitivity: percentiles "
    "99.70 / 99.80 / 99.90 / 99.95]"
)

lines.append(
    f"ROAD FPR range (max-min): "
    f"{road_fpr_range:.6%}"
)

lines.append(
    f"ROAD TPR range (max-min): "
    f"{road_tpr_range:.6%}"
)

lines.append(
    f"ROAD BA range  (max-min): "
    f"{road_ba_range:.6%}"
)

lines.append("")
lines.append(
    "[Interpretation rule]"
)

lines.append(
    "This sweep is a post-hoc sensitivity analysis only. "
    "The frozen 99.90% threshold is NOT re-selected using ROAD."
)


report = "\n".join(
    lines
)

with open(
    OUT_TXT,
    "w",
    encoding="utf-8",
) as f:
    f.write(
        report + "\n"
    )


print()
print(report)

print()
print(
    f"[saved] {OUT_CSV}"
)

print(
    f"[saved] {detail_path}"
)

print(
    f"[saved] {OUT_TXT}"
)
