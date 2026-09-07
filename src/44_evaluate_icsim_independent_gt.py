"""
44_evaluate_icsim_independent_gt.py

ICSim evaluation using independent injector-side per-frame ground truth.

Ground-truth chain:
    injector GT
        -> 43: exact ID + payload + timestamp match to candump
        -> 44: candump frame -> detector prediction via ID + timestamp

IMPORTANT:
No detector feature is used to define attack ground truth.
No value_zscore.
No is_unknown_id.
No Mahalanobis distance.
No model prediction.
No attack time-window heuristic.

Usage:
    python3 src/44_evaluate_icsim_independent_gt.py \
        results/icsim_candump.log \
        results/icsim_attack_ground_truth_matched.csv \
        results/realtime_predictions.csv \
        5.0

The final optional argument is candump <-> IDS prediction
timestamp tolerance in milliseconds.
"""

import csv
import os
import re
import sys
import bisect
from collections import defaultdict

import numpy as np
import pandas as pd


HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))

CANDUMP_PATH = (
    sys.argv[1]
    if len(sys.argv) >= 2
    else os.path.join(
        ROOT,
        "results",
        "icsim_candump.log"
    )
)

GT_MATCHED_PATH = (
    sys.argv[2]
    if len(sys.argv) >= 3
    else os.path.join(
        ROOT,
        "results",
        "icsim_attack_ground_truth_matched.csv"
    )
)

PRED_PATH = (
    sys.argv[3]
    if len(sys.argv) >= 4
    else os.path.join(
        ROOT,
        "results",
        "realtime_predictions.csv"
    )
)

TOL_MS = (
    float(sys.argv[4])
    if len(sys.argv) >= 5
    else 5.0
)

TOL_SEC = TOL_MS / 1000.0

OUT_SUMMARY = os.path.join(
    ROOT,
    "results",
    "icsim_independent_gt_summary.txt"
)

OUT_FRAME = os.path.join(
    ROOT,
    "results",
    "icsim_independent_gt_frame_eval.csv"
)


LINE_RE = re.compile(
    r"\(([\d.]+)\)\s+"
    r"\S+\s+"
    r"([0-9A-Fa-f]+)#"
    r"([0-9A-Fa-f]*)"
)


def load_candump(path):
    rows = []

    with open(
        path,
        "r",
        errors="ignore"
    ) as f:

        for line_no, line in enumerate(f, 1):
            m = LINE_RE.search(line)

            if not m:
                continue

            rows.append({
                "candump_index": len(rows),
                "candump_line_no": line_no,
                "candump_ts": float(m.group(1)),
                "can_id": int(m.group(2), 16),
                "payload_hex": m.group(3).upper(),
                "pred_index": None,
                "pred_timestamp": None,
                "final_label": None,
                "pred_delta_ms": None,
            })

    return rows


def load_attack_gt(path):
    df = pd.read_csv(path)

    required = {
        "seq",
        "can_id",
        "payload_hex",
        "attack_type",
        "matched",
        "candump_line_no",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Matched GT missing columns: {sorted(missing)}"
        )

    df = df[df["matched"] == 1].copy()

    df["candump_line_no"] = (
        pd.to_numeric(
            df["candump_line_no"],
            errors="coerce"
        )
    )

    df = df.dropna(
        subset=["candump_line_no"]
    )

    df["candump_line_no"] = (
        df["candump_line_no"].astype(int)
    )

    return df


def load_predictions(path):
    df = pd.read_csv(path)

    required = {
        "timestamp",
        "can_id",
        "final_label",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Predictions missing columns: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    df["can_id"] = pd.to_numeric(
        df["can_id"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp", "can_id"]
    ).copy()

    df["can_id"] = df["can_id"].astype(int)

    df = df.reset_index(drop=True)

    return df


candump = load_candump(CANDUMP_PATH)
attack_gt = load_attack_gt(GT_MATCHED_PATH)
pred = load_predictions(PRED_PATH)


print(
    "=== ICSim Independent-GT Evaluation ==="
)
print()
print(
    f"Candump frames        : {len(candump):,}"
)
print(
    f"Matched attack GT     : {len(attack_gt):,}"
)
print(
    f"IDS predictions       : {len(pred):,}"
)
print(
    f"Pred match tolerance  : ±{TOL_MS:.3f} ms"
)


# ---------------------------------------------------------
# Attach independent attack GT to exact candump line.
# ---------------------------------------------------------

attack_by_line = {}

for _, row in attack_gt.iterrows():

    line_no = int(
        row["candump_line_no"]
    )

    if line_no in attack_by_line:
        raise RuntimeError(
            "More than one GT attack frame mapped to "
            f"candump line {line_no}"
        )

    attack_by_line[line_no] = {
        "gt_seq": int(row["seq"]),
        "attack_type": str(
            row["attack_type"]
        ),
    }


for c in candump:

    gt_info = attack_by_line.get(
        c["candump_line_no"]
    )

    if gt_info is None:
        c["is_attack"] = False
        c["gt_seq"] = None
        c["attack_type"] = "normal"
    else:
        c["is_attack"] = True
        c["gt_seq"] = gt_info["gt_seq"]
        c["attack_type"] = (
            gt_info["attack_type"]
        )


# Verify every matched GT line exists in candump.
candump_lines = {
    c["candump_line_no"]
    for c in candump
}

missing_gt_lines = (
    set(attack_by_line.keys())
    - candump_lines
)

if missing_gt_lines:
    raise RuntimeError(
        f"{len(missing_gt_lines)} attack GT candump "
        "line(s) not found in candump file."
    )


# ---------------------------------------------------------
# candump <-> detector prediction 1:1 matching.
#
# Match ONLY with:
#   same CAN ID
#   nearest timestamp
#
# No IDS feature participates in matching.
# ---------------------------------------------------------

candump_by_id = defaultdict(list)
candump_ts_by_id = defaultdict(list)

for idx, c in enumerate(candump):
    cid = c["can_id"]

    candump_by_id[cid].append(idx)
    candump_ts_by_id[cid].append(
        c["candump_ts"]
    )


used_candump = set()
prediction_matches = []

for pred_idx, row in pred.iterrows():

    cid = int(row["can_id"])
    pts = float(row["timestamp"])

    candidate_indices = (
        candump_by_id.get(cid, [])
    )

    candidate_times = (
        candump_ts_by_id.get(cid, [])
    )

    if not candidate_indices:
        prediction_matches.append(None)
        continue

    pos = bisect.bisect_left(
        candidate_times,
        pts
    )

    # Search a small neighborhood around nearest
    # timestamp. More than enough because CAN frames
    # are chronologically sorted.
    left = max(0, pos - 8)
    right = min(
        len(candidate_indices),
        pos + 9
    )

    best_idx = None
    best_abs_dt = None
    best_signed_dt = None

    for j in range(left, right):

        cidx = candidate_indices[j]

        if cidx in used_candump:
            continue

        cts = candump[cidx][
            "candump_ts"
        ]

        signed_dt = pts - cts
        abs_dt = abs(signed_dt)

        if abs_dt > TOL_SEC:
            continue

        if (
            best_abs_dt is None
            or abs_dt < best_abs_dt
        ):
            best_idx = cidx
            best_abs_dt = abs_dt
            best_signed_dt = signed_dt

    if best_idx is None:
        prediction_matches.append(None)
        continue

    used_candump.add(best_idx)

    candump[best_idx]["pred_index"] = (
        int(pred_idx)
    )

    candump[best_idx][
        "pred_timestamp"
    ] = pts

    candump[best_idx][
        "final_label"
    ] = str(
        row["final_label"]
    )

    candump[best_idx][
        "pred_delta_ms"
    ] = (
        best_signed_dt * 1000.0
    )

    prediction_matches.append(best_idx)


# ---------------------------------------------------------
# Coverage statistics
# ---------------------------------------------------------

n_candump = len(candump)

n_candump_with_pred = sum(
    c["pred_index"] is not None
    for c in candump
)

candump_pred_coverage = (
    n_candump_with_pred / n_candump
    if n_candump
    else float("nan")
)

n_pred = len(pred)

n_pred_matched = sum(
    x is not None
    for x in prediction_matches
)

pred_match_rate = (
    n_pred_matched / n_pred
    if n_pred
    else float("nan")
)


pred_deltas = [
    c["pred_delta_ms"]
    for c in candump
    if c["pred_delta_ms"] is not None
]


# ---------------------------------------------------------
# Attack evaluation
#
# Denominator = independent injector GT frames that
# were verified in candump by script 43.
#
# If detector failed to produce a prediction for an
# attack frame, it counts as not detected (FN).
# ---------------------------------------------------------

attack_frames = [
    c for c in candump
    if c["is_attack"]
]

attack_total = len(attack_frames)

attack_processed = sum(
    c["pred_index"] is not None
    for c in attack_frames
)

tp = sum(
    c["pred_index"] is not None
    and c["final_label"] == "anomaly"
    for c in attack_frames
)

fn = attack_total - tp

attack_recall = (
    tp / attack_total
    if attack_total
    else float("nan")
)

attack_processing_coverage = (
    attack_processed / attack_total
    if attack_total
    else float("nan")
)


# ---------------------------------------------------------
# Normal/FPR evaluation
#
# Normal = candump frames that are NOT present in the
# independent injector GT.
#
# FPR is calculated over normal frames actually processed
# by the IDS.
# ---------------------------------------------------------

normal_frames = [
    c for c in candump
    if not c["is_attack"]
]

normal_total = len(normal_frames)

normal_processed_frames = [
    c for c in normal_frames
    if c["pred_index"] is not None
]

normal_processed = len(
    normal_processed_frames
)

fp = sum(
    c["final_label"] == "anomaly"
    for c in normal_processed_frames
)

tn = sum(
    c["final_label"] == "normal"
    for c in normal_processed_frames
)

normal_processing_coverage = (
    normal_processed / normal_total
    if normal_total
    else float("nan")
)

fpr = (
    fp / normal_processed
    if normal_processed
    else float("nan")
)


# ---------------------------------------------------------
# Per-attack statistics
# ---------------------------------------------------------

attack_types = sorted(
    set(
        c["attack_type"]
        for c in attack_frames
    )
)

per_attack = []

for attack_type in attack_types:

    subset = [
        c for c in attack_frames
        if c["attack_type"]
        == attack_type
    ]

    n = len(subset)

    processed = sum(
        c["pred_index"] is not None
        for c in subset
    )

    detected = sum(
        c["pred_index"] is not None
        and c["final_label"]
        == "anomaly"
        for c in subset
    )

    recall = (
        detected / n
        if n
        else float("nan")
    )

    coverage = (
        processed / n
        if n
        else float("nan")
    )

    per_attack.append(
        (
            attack_type,
            n,
            processed,
            detected,
            recall,
            coverage,
        )
    )


# ---------------------------------------------------------
# Save frame-level audit file
# ---------------------------------------------------------

with open(
    OUT_FRAME,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fields = [
        "candump_line_no",
        "candump_ts",
        "can_id",
        "payload_hex",
        "is_attack",
        "gt_seq",
        "attack_type",
        "pred_index",
        "pred_timestamp",
        "pred_delta_ms",
        "final_label",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields
    )

    w.writeheader()

    for c in candump:

        row = {
            k: c.get(k, "")
            for k in fields
        }

        w.writerow(row)


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

def pct(x):
    if np.isnan(x):
        return "N/A"

    return f"{x * 100:.4f}%"


lines = []

lines.append(
    "=== ICSim Independent-GT Evaluation ==="
)

lines.append("")

lines.append(
    "[Ground Truth Integrity]"
)

lines.append(
    f"Independent attack GT frames : "
    f"{attack_total:,}"
)

lines.append(
    "GT definition source         : "
    "injector-side per-frame log"
)

lines.append(
    "GT->candump identity         : "
    "CAN ID + payload + timestamp"
)

lines.append(
    "Detector feature used for GT : NO"
)

lines.append(
    "value_zscore used for GT     : NO"
)

lines.append(
    "is_unknown_id used for GT    : NO"
)

lines.append(
    "Mahalanobis score used for GT: NO"
)

lines.append(
    "Model prediction used for GT : NO"
)

lines.append(
    "Attack-window heuristic      : NO"
)


lines.append("")
lines.append(
    "[Candump <-> IDS Prediction Matching]"
)

lines.append(
    f"Candump frames              : "
    f"{n_candump:,}"
)

lines.append(
    f"IDS prediction rows         : "
    f"{n_pred:,}"
)

lines.append(
    f"Matched predictions         : "
    f"{n_pred_matched:,}/{n_pred:,} "
    f"({pct(pred_match_rate)})"
)

lines.append(
    f"Candump processing coverage : "
    f"{n_candump_with_pred:,}/{n_candump:,} "
    f"({pct(candump_pred_coverage)})"
)

lines.append(
    f"Timestamp tolerance         : "
    f"±{TOL_MS:.3f} ms"
)

if pred_deltas:

    arr = np.abs(
        np.asarray(
            pred_deltas,
            dtype=float
        )
    )

    signed = np.asarray(
        pred_deltas,
        dtype=float
    )

    lines.append(
        f"Mean delta "
        f"(IDS-candump)       : "
        f"{signed.mean():.6f} ms"
    )

    lines.append(
        f"Median delta               : "
        f"{np.median(signed):.6f} ms"
    )

    lines.append(
        f"P95 absolute delta         : "
        f"{np.percentile(arr, 95):.6f} ms"
    )

    lines.append(
        f"P99 absolute delta         : "
        f"{np.percentile(arr, 99):.6f} ms"
    )

    lines.append(
        f"Max absolute delta         : "
        f"{arr.max():.6f} ms"
    )


lines.append("")
lines.append("[Attack Detection]")

lines.append(
    f"Attack frames              : "
    f"{attack_total:,}"
)

lines.append(
    f"Attack frames processed    : "
    f"{attack_processed:,} "
    f"({pct(attack_processing_coverage)})"
)

lines.append(
    f"True Positives             : {tp:,}"
)

lines.append(
    f"False Negatives            : {fn:,}"
)

lines.append(
    f"TPR / Recall               : "
    f"{pct(attack_recall)}"
)


lines.append("")
lines.append("[Per Attack]")

if per_attack:

    for (
        attack_type,
        n,
        processed,
        detected,
        recall,
        coverage,
    ) in per_attack:

        lines.append(
            f"{attack_type}: "
            f"TP={detected:,}/{n:,}, "
            f"TPR={pct(recall)}, "
            f"processed={processed:,}/{n:,} "
            f"({pct(coverage)})"
        )
else:
    lines.append(
        "No independently matched "
        "attack frames."
    )


lines.append("")
lines.append("[Normal / False Positive]")

lines.append(
    f"Normal candump frames      : "
    f"{normal_total:,}"
)

lines.append(
    f"Normal frames processed    : "
    f"{normal_processed:,} "
    f"({pct(normal_processing_coverage)})"
)

lines.append(
    f"False Positives            : {fp:,}"
)

lines.append(
    f"True Negatives             : {tn:,}"
)

lines.append(
    f"FPR                        : "
    f"{pct(fpr)}"
)

if normal_total == 0:
    lines.append(
        "WARNING: no normal background "
        "frames were present; FPR cannot "
        "be evaluated in this run."
    )


report = "\n".join(lines)

print()
print(report)

with open(
    OUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:
    f.write(report + "\n")


print()
print(
    f"[saved] {OUT_SUMMARY}"
)

print(
    f"[saved] {OUT_FRAME}"
)
