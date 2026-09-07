"""
43_match_icsim_ground_truth.py

Independent injector GT ↔ candump 1:1 matcher.

Matching uses ONLY:
- CAN ID
- payload
- timestamp proximity

No IDS feature / prediction / score / threshold is used.

Usage:
    python3 src/43_match_icsim_ground_truth.py \
        <candump_log> \
        [gt_csv] \
        [tolerance_ms]

Example:
    python3 src/43_match_icsim_ground_truth.py \
        results/icsim_candump.log \
        results/icsim_attack_ground_truth.csv \
        2.0
"""

import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np


HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))

if len(sys.argv) < 2:
    print(
        "Usage: python3 src/43_match_icsim_ground_truth.py "
        "<candump_log> [gt_csv] [tolerance_ms]"
    )
    sys.exit(1)

CANDUMP_PATH = sys.argv[1]

GT_PATH = (
    sys.argv[2]
    if len(sys.argv) >= 3
    else os.path.join(
        ROOT,
        "results",
        "icsim_attack_ground_truth.csv"
    )
)

TOL_MS = (
    float(sys.argv[3])
    if len(sys.argv) >= 4
    else 2.0
)

TOL_SEC = TOL_MS / 1000.0

OUT_PATH = os.path.join(
    ROOT,
    "results",
    "icsim_attack_ground_truth_matched.csv"
)

REPORT_PATH = os.path.join(
    ROOT,
    "results",
    "icsim_gt_matching_report.txt"
)


# candump -L style:
# (1788790731.441812) vcan0 13A#00000000000000FF
LINE_RE = re.compile(
    r"\(([\d.]+)\)\s+"
    r"\S+\s+"
    r"([0-9A-Fa-f]+)#"
    r"([0-9A-Fa-f]*)"
)


def normalize_payload(hexstr):
    s = hexstr.strip().upper()
    return s


def load_gt(path):
    rows = []

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "seq",
            "timestamp",
            "can_id",
            "payload_hex",
            "attack_type",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(
                f"GT missing columns: {sorted(missing)}"
            )

        for row in reader:
            rows.append({
                "seq": int(row["seq"]),
                "gt_ts": float(row["timestamp"]),
                "can_id": int(row["can_id"]),
                "payload_hex": normalize_payload(
                    row["payload_hex"]
                ),
                "attack_type": row["attack_type"],
            })

    return rows


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
                "line_no": line_no,
                "candump_ts": float(m.group(1)),
                "can_id": int(m.group(2), 16),
                "payload_hex": normalize_payload(
                    m.group(3)
                ),
                "used": False,
            })

    return rows


gt_rows = load_gt(GT_PATH)
can_rows = load_candump(CANDUMP_PATH)

print(f"GT frames      : {len(gt_rows):,}")
print(f"Candump frames : {len(can_rows):,}")
print(f"Tolerance      : ±{TOL_MS:.3f} ms")


# Index candump frames by exact (CAN ID, payload)
index = defaultdict(list)

for c in can_rows:
    key = (
        c["can_id"],
        c["payload_hex"]
    )

    index[key].append(
        c["candump_index"]
    )


matched_rows = []
matched_dt_ms = []


for g in gt_rows:

    key = (
        g["can_id"],
        g["payload_hex"]
    )

    best_idx = None
    best_abs_dt = None
    best_signed_dt = None

    for idx in index.get(key, []):

        c = can_rows[idx]

        if c["used"]:
            continue

        signed_dt = (
            c["candump_ts"]
            - g["gt_ts"]
        )

        abs_dt = abs(signed_dt)

        if abs_dt > TOL_SEC:
            continue

        if (
            best_abs_dt is None
            or abs_dt < best_abs_dt
        ):
            best_idx = idx
            best_abs_dt = abs_dt
            best_signed_dt = signed_dt

    if best_idx is None:

        matched_rows.append({
            **g,
            "matched": 0,
            "candump_ts": "",
            "delta_ms": "",
            "candump_line_no": "",
        })

        continue

    c = can_rows[best_idx]
    c["used"] = True

    dt_ms = best_signed_dt * 1000.0
    matched_dt_ms.append(dt_ms)

    matched_rows.append({
        **g,
        "matched": 1,
        "candump_ts":
            f"{c['candump_ts']:.9f}",
        "delta_ms":
            f"{dt_ms:.6f}",
        "candump_line_no":
            c["line_no"],
    })


with open(
    OUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "seq",
        "gt_ts",
        "candump_ts",
        "delta_ms",
        "can_id",
        "payload_hex",
        "attack_type",
        "matched",
        "candump_line_no",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    w.writeheader()

    for row in matched_rows:
        w.writerow(row)


n_gt = len(gt_rows)
n_match = sum(
    r["matched"]
    for r in matched_rows
)

match_rate = (
    n_match / n_gt
    if n_gt
    else 0.0
)


lines = []

lines.append(
    "=== ICSim Independent Ground Truth Matching ==="
)

lines.append(
    f"GT frames: {n_gt:,}"
)

lines.append(
    f"Candump frames: {len(can_rows):,}"
)

lines.append(
    f"Tolerance: ±{TOL_MS:.3f} ms"
)

lines.append(
    f"Matched: {n_match:,}/{n_gt:,}"
)

lines.append(
    f"Matching rate: {match_rate:.4%}"
)


if matched_dt_ms:

    a = np.asarray(
        matched_dt_ms,
        dtype=float
    )

    abs_a = np.abs(a)

    lines.extend([
        "",
        "Timestamp delta "
        "(candump - injector GT):",
        f"mean = {a.mean():.6f} ms",
        f"median = {np.median(a):.6f} ms",
        f"p95 abs = {np.percentile(abs_a, 95):.6f} ms",
        f"p99 abs = {np.percentile(abs_a, 99):.6f} ms",
        f"max abs = {abs_a.max():.6f} ms",
    ])


# Attack-type matching statistics
lines.append("")
lines.append("Per attack type:")

attack_types = sorted(
    set(
        r["attack_type"]
        for r in matched_rows
    )
)

for attack in attack_types:

    subset = [
        r
        for r in matched_rows
        if r["attack_type"] == attack
    ]

    n = len(subset)
    m = sum(
        r["matched"]
        for r in subset
    )

    rate = (
        m / n
        if n
        else 0.0
    )

    lines.append(
        f"{attack}: "
        f"{m:,}/{n:,} "
        f"({rate:.4%})"
    )


report = "\n".join(lines)

print()
print(report)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:
    f.write(report + "\n")


print()
print(f"Matched GT : {OUT_PATH}")
print(f"Report     : {REPORT_PATH}")
