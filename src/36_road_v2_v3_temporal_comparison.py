"""
ROAD frozen comparison: V2 vs V3 temporal

원칙:
- V2, V3 모두 HCRL에서 학습/threshold 고정
- ROAD로 재학습/정규화/threshold 조정 절대 없음
- capture_metadata.json의 injection interval + ID로
  실제 공격 message만 TPR 계산
- ambient는 FPR 계산
"""

import os
import re
import glob
import json
import pickle
from collections import defaultdict, deque

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))

MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")

ROAD_ATTACKS = os.path.join(ROOT, "road_data", "attacks")
ROAD_AMBIENT = os.path.join(ROOT, "road_data", "ambient")

os.makedirs(RESULT_DIR, exist_ok=True)


# ==================================================
# Frozen V2 stats
# ==================================================
with open(
    os.path.join(MODEL_DIR, "idagnostic_stats.pkl"), "rb"
) as f:
    STATS = pickle.load(f)

WINDOW = int(STATS["window"])
G_DELTA_MEAN = float(STATS["global_delta_mean"])
G_DELTA_STD = float(STATS["global_delta_std"])
BYTE_MEAN = np.asarray(STATS["byte_pos_mean"], dtype=float)
BYTE_STD = np.asarray(STATS["byte_pos_std"], dtype=float)

v2_npz = np.load(os.path.join(MODEL_DIR, "maha_v2.npz"))

V2_MEAN = v2_npz["mean"]
V2_INV = v2_npz["inv_cov"]
V2_THR = float(v2_npz["thr"])


# ==================================================
# Frozen V3 model
# ==================================================
with open(
    os.path.join(MODEL_DIR, "v3_feature_config.pkl"), "rb"
) as f:
    V3_CFG = pickle.load(f)

TEMP_WINDOW = int(V3_CFG["temp_window"])

v3_npz = np.load(
    os.path.join(MODEL_DIR, "maha_v3_temporal.npz"),
    allow_pickle=True
)

V3_MEAN = v3_npz["mean"]
V3_INV = v3_npz["inv_cov"]
V3_THR = float(v3_npz["thr"])


# ==================================================
# ROAD metadata
# ==================================================
with open(
    os.path.join(ROAD_ATTACKS, "capture_metadata.json")
) as f:
    METADATA = json.load(f)

LINE_RE = re.compile(
    r"\(([\d.]+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)"
)


def parse_log(path):
    rows = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)

            if not m:
                continue

            ts = float(m.group(1))
            cid = int(m.group(2), 16)
            hexdata = m.group(3)

            byts = [
                int(hexdata[j:j+2], 16)
                for j in range(0, min(len(hexdata), 16), 2)
            ]

            byts = (byts + [0] * 8)[:8]

            rows.append((ts, cid, byts))

    return rows


def entropy(b):
    _, c = np.unique(b, return_counts=True)
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def extract_both(rows):
    X2 = []
    X3 = []

    id_window = deque(maxlen=WINDOW)

    last_payload = {}
    last_ts = {}

    delta_hist = defaultdict(
        lambda: deque(maxlen=TEMP_WINDOW)
    )
    iat_hist = defaultdict(
        lambda: deque(maxlen=TEMP_WINDOW)
    )

    run_length = defaultdict(int)

    for ts, cid, raw_pl in rows:
        pl = np.asarray(raw_pl, dtype=float)

        id_window.append(cid)

        freq = sum(x == cid for x in id_window) / len(id_window)
        unique_ids = len(set(id_window))

        ent = entropy(pl)
        mean_byte = float(pl.mean())

        if cid in last_payload:
            delta = float(
                np.abs(pl - last_payload[cid]).sum()
            )
            same = bool(
                np.array_equal(pl, last_payload[cid])
            )
        else:
            delta = 0.0
            same = False

        delta_z = (
            delta - G_DELTA_MEAN
        ) / G_DELTA_STD

        value_z = float(
            np.max(
                np.abs(
                    (pl - BYTE_MEAN) / BYTE_STD
                )
            )
        )

        base = [
            freq,
            unique_ids,
            ent,
            mean_byte,
            delta_z,
            value_z,
        ]

        X2.append(base)

        # -------- temporal --------
        dh = delta_hist[cid]
        dh.append(delta)

        roll_mean = float(np.mean(dh))
        roll_std = float(np.std(dh))
        change_rate = float(
            np.mean(np.asarray(dh) > 0.0)
        )

        if cid not in last_payload:
            run_length[cid] = 1
        elif same:
            run_length[cid] += 1
        else:
            run_length[cid] = 1

        run_norm = (
            min(run_length[cid], TEMP_WINDOW)
            / TEMP_WINDOW
        )

        if cid in last_ts:
            dt = max(float(ts) - last_ts[cid], 0.0)
            iat_hist[cid].append(dt)

        ih = iat_hist[cid]

        if len(ih) >= 2:
            m = float(np.mean(ih))
            s = float(np.std(ih))
            iat_cv = s / (m + 1e-9)
        else:
            iat_cv = 0.0

        iat_cv = min(iat_cv, 20.0)

        X3.append(
            base + [
                roll_mean,
                roll_std,
                change_rate,
                run_norm,
                iat_cv,
            ]
        )

        last_payload[cid] = pl.copy()
        last_ts[cid] = float(ts)

    return (
        np.asarray(X2, dtype=np.float64),
        np.asarray(X3, dtype=np.float64),
    )


def maha_flags(X, mean, inv_cov, thr):
    d = X - mean
    q = np.einsum(
        "ij,jk,ik->i",
        d,
        inv_cov,
        d
    )
    dist = np.sqrt(np.maximum(q, 0.0))
    return dist > thr


def family(name):
    n = name.lower()

    if "correlated_signal" in n:
        return "correlated_signal"
    if "fuzz" in n:
        return "fuzzing"
    if "speedometer" in n:
        return "max_speedometer"
    if "reverse_light" in n:
        return "reverse_light"
    if "coolant" in n:
        return "max_engine_coolant_temp"

    return "other"


# ==================================================
# Attack evaluation
# ==================================================
print("=" * 110)
print("ROAD Frozen V2 vs V3 temporal comparison")
print("=" * 110)

attack_rows = []
family_v2 = defaultdict(list)
family_v3 = defaultdict(list)

header = (
    f"{'Attack file':48s} "
    f"{'V2 TPR':>10s} "
    f"{'V3 TPR':>10s} "
    f"{'Δ':>10s} "
    f"{'N':>10s}"
)

print(header)
print("-" * 110)

for path in sorted(
    glob.glob(os.path.join(ROAD_ATTACKS, "*.log"))
):
    name = os.path.basename(path)

    if "_masquerade" in name:
        continue

    key = name.replace(".log", "")
    meta = METADATA.get(key, {})

    interval = meta.get("injection_interval")
    injection_id = meta.get("injection_id")

    # 28번과 동일: metadata 없는 행동기반 공격은
    # corrected TPR 표에서는 제외
    if interval is None:
        continue

    rows = parse_log(path)

    if len(rows) < WINDOW + 1:
        continue

    X2, X3 = extract_both(rows)

    f2 = maha_flags(
        X2, V2_MEAN, V2_INV, V2_THR
    )
    f3 = maha_flags(
        X3, V3_MEAN, V3_INV, V3_THR
    )

    t0 = rows[0][0]

    rel_ts = np.asarray(
        [r[0] - t0 for r in rows]
    )

    ids = np.asarray(
        [r[1] for r in rows]
    )

    mask = (
        (rel_ts >= interval[0]) &
        (rel_ts <= interval[1])
    )

    if injection_id and injection_id != "XXX":
        target_id = int(injection_id, 16)
        mask &= (ids == target_id)

    n = int(mask.sum())

    if n == 0:
        continue

    tpr2 = float(f2[mask].mean())
    tpr3 = float(f3[mask].mean())
    delta = tpr3 - tpr2

    fam = family(name)

    family_v2[fam].append(tpr2)
    family_v3[fam].append(tpr3)

    attack_rows.append(
        (name, fam, tpr2, tpr3, delta, n)
    )

    print(
        f"{name:48s} "
        f"{tpr2:10.2%} "
        f"{tpr3:10.2%} "
        f"{delta:+10.2%} "
        f"{n:10,d}"
    )


# ==================================================
# Ambient FPR
# ==================================================
print("\n" + "=" * 110)
print("ROAD ambient FPR")
print("=" * 110)

ambient_rows = []

for path in sorted(
    glob.glob(os.path.join(ROAD_AMBIENT, "*.log"))
):
    name = os.path.basename(path)

    rows = parse_log(path)

    if len(rows) < WINDOW + 1:
        continue

    X2, X3 = extract_both(rows)

    f2 = maha_flags(
        X2, V2_MEAN, V2_INV, V2_THR
    )
    f3 = maha_flags(
        X3, V3_MEAN, V3_INV, V3_THR
    )

    fpr2 = float(f2.mean())
    fpr3 = float(f3.mean())

    ambient_rows.append(
        (name, fpr2, fpr3, len(rows))
    )

    print(
        f"{name:48s} "
        f"V2={fpr2:9.3%} "
        f"V3={fpr3:9.3%} "
        f"N={len(rows):,}"
    )


# ==================================================
# Summary
# ==================================================
out_lines = []

out_lines.append(
    "=== ROAD Frozen V2 vs V3 Temporal Comparison ==="
)
out_lines.append(
    "ROAD 재학습/threshold 조정 없음."
)
out_lines.append(
    "공격 TPR은 capture_metadata interval + injection ID 기준."
)
out_lines.append("")

out_lines.append(
    "| Attack | Family | V2 TPR | V3 TPR | Delta | N |"
)
out_lines.append(
    "|---|---|---:|---:|---:|---:|"
)

for name, fam, a, b, d, n in attack_rows:
    out_lines.append(
        f"| {name} | {fam} | "
        f"{a:.2%} | {b:.2%} | {d:+.2%} | {n:,} |"
    )

out_lines.append("")
out_lines.append("=== Family 평균 ===")

for fam in sorted(family_v2):
    a = float(np.mean(family_v2[fam]))
    b = float(np.mean(family_v3[fam]))

    out_lines.append(
        f"{fam}: V2={a:.2%}, "
        f"V3={b:.2%}, Δ={b-a:+.2%}"
    )

if ambient_rows:
    mean_fpr2 = float(
        np.mean([x[1] for x in ambient_rows])
    )
    mean_fpr3 = float(
        np.mean([x[2] for x in ambient_rows])
    )

    out_lines.append("")
    out_lines.append("=== ROAD Ambient ===")
    out_lines.append(
        f"V2 mean FPR = {mean_fpr2:.4%}"
    )
    out_lines.append(
        f"V3 mean FPR = {mean_fpr3:.4%}"
    )
    out_lines.append(
        f"FPR delta = {mean_fpr3-mean_fpr2:+.4%}"
    )

out_path = os.path.join(
    RESULT_DIR,
    "f3_v2_vs_v3_temporal_road.txt"
)

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")

print("\n" + "\n".join(out_lines[-20:]))
print(f"\n결과 저장: {out_path}")
