"""
41_road_v2_v3_original_masquerade.py

목적
----
기존 V3 temporal detector의 ROAD 성능을
ORIGINAL injection / MASQUERADE 조건으로 분리 평가한다.

핵심 질문
---------
V3 correlated_signal 99.84% 향상이
payload-content 일반화인가,
아니면 injection-induced timing/traffic anomaly에 대한 민감도인가?

원칙
----
- V2/V3 모델, feature stats, threshold 모두 HCRL에서 frozen
- ROAD 재학습 / 재정규화 / threshold 조정 없음
- capture_metadata.json의 injection interval + injection ID 사용
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
ATTACK_DIR = os.path.join(ROOT, "road_data", "attacks")
AMBIENT_DIR = os.path.join(ROOT, "road_data", "ambient")

os.makedirs(RESULT_DIR, exist_ok=True)


# =========================================================
# Frozen V2 feature statistics
# =========================================================
with open(
    os.path.join(MODEL_DIR, "idagnostic_stats.pkl"),
    "rb"
) as f:
    STATS = pickle.load(f)

WINDOW = int(STATS["window"])
G_DELTA_MEAN = float(STATS["global_delta_mean"])
G_DELTA_STD = float(STATS["global_delta_std"])

BYTE_MEAN = np.asarray(
    STATS["byte_pos_mean"],
    dtype=float
)

BYTE_STD = np.asarray(
    STATS["byte_pos_std"],
    dtype=float
)

BYTE_STD = np.where(
    BYTE_STD < 1e-9,
    1.0,
    BYTE_STD
)


# =========================================================
# Frozen V2 Mahalanobis
# =========================================================
v2 = np.load(
    os.path.join(MODEL_DIR, "maha_v2.npz")
)

V2_MEAN = v2["mean"]
V2_INV = v2["inv_cov"]
V2_THR = float(v2["thr"])


# =========================================================
# Frozen V3
# =========================================================
with open(
    os.path.join(MODEL_DIR, "v3_feature_config.pkl"),
    "rb"
) as f:
    V3_CFG = pickle.load(f)

TEMP_WINDOW = int(
    V3_CFG["temp_window"]
)

v3 = np.load(
    os.path.join(
        MODEL_DIR,
        "maha_v3_temporal.npz"
    ),
    allow_pickle=True
)

V3_MEAN = v3["mean"]
V3_INV = v3["inv_cov"]
V3_THR = float(v3["thr"])


# =========================================================
# ROAD metadata
# =========================================================
with open(
    os.path.join(
        ATTACK_DIR,
        "capture_metadata.json"
    ),
    "r"
) as f:
    METADATA = json.load(f)


LINE_RE = re.compile(
    r"\(([\d.]+)\)\s+\S+\s+"
    r"([0-9A-Fa-f]+)#([0-9A-Fa-f]*)"
)


def parse_log(path):
    rows = []

    with open(
        path,
        "r",
        errors="ignore"
    ) as f:

        for line in f:
            m = LINE_RE.search(line)

            if not m:
                continue

            ts = float(m.group(1))
            cid = int(m.group(2), 16)
            hexdata = m.group(3)

            data = [
                int(hexdata[i:i+2], 16)
                for i in range(
                    0,
                    min(len(hexdata), 16),
                    2
                )
            ]

            data = (data + [0] * 8)[:8]

            rows.append(
                (
                    ts,
                    cid,
                    np.asarray(
                        data,
                        dtype=np.float64
                    )
                )
            )

    return rows


def entropy(pl):
    _, counts = np.unique(
        pl,
        return_counts=True
    )

    p = counts / counts.sum()

    return float(
        -np.sum(
            p * np.log2(p + 1e-12)
        )
    )


def extract_v2_v3(rows):
    X2 = []
    X3 = []

    id_window = deque(
        maxlen=WINDOW
    )

    last_payload = {}
    last_ts = {}

    delta_hist = defaultdict(
        lambda: deque(
            maxlen=TEMP_WINDOW
        )
    )

    iat_hist = defaultdict(
        lambda: deque(
            maxlen=TEMP_WINDOW
        )
    )

    run_length = defaultdict(int)

    for ts, cid, pl in rows:

        id_window.append(cid)

        # -------------------------
        # V2
        # -------------------------
        freq = (
            sum(
                x == cid
                for x in id_window
            )
            / len(id_window)
        )

        unique_ids = len(
            set(id_window)
        )

        ent = entropy(pl)

        mean_byte = float(
            pl.mean()
        )

        if cid in last_payload:
            delta = float(
                np.abs(
                    pl - last_payload[cid]
                ).sum()
            )

            same_payload = bool(
                np.array_equal(
                    pl,
                    last_payload[cid]
                )
            )

        else:
            delta = 0.0
            same_payload = False

        delta_z = (
            delta - G_DELTA_MEAN
        ) / (
            G_DELTA_STD + 1e-12
        )

        value_z = float(
            np.max(
                np.abs(
                    (
                        pl - BYTE_MEAN
                    )
                    / BYTE_STD
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

        # -------------------------
        # V3 temporal
        # -------------------------
        dh = delta_hist[cid]
        dh.append(delta)

        rolling_delta_mean = float(
            np.mean(dh)
        )

        rolling_delta_std = float(
            np.std(dh)
        )

        payload_change_rate = float(
            np.mean(
                np.asarray(dh) > 0.0
            )
        )

        if cid not in last_payload:
            run_length[cid] = 1

        elif same_payload:
            run_length[cid] += 1

        else:
            run_length[cid] = 1

        payload_run_length = (
            min(
                run_length[cid],
                TEMP_WINDOW
            )
            / TEMP_WINDOW
        )

        if cid in last_ts:
            dt = max(
                float(ts) - last_ts[cid],
                0.0
            )

            iat_hist[cid].append(dt)

        ih = iat_hist[cid]

        if len(ih) >= 2:
            m = float(np.mean(ih))
            s = float(np.std(ih))

            interarrival_cv = (
                s / (m + 1e-9)
            )

        else:
            interarrival_cv = 0.0

        interarrival_cv = min(
            interarrival_cv,
            20.0
        )

        X3.append(
            base + [
                rolling_delta_mean,
                rolling_delta_std,
                payload_change_rate,
                payload_run_length,
                interarrival_cv,
            ]
        )

        last_payload[cid] = pl.copy()
        last_ts[cid] = float(ts)

    return (
        np.asarray(
            X2,
            dtype=np.float64
        ),
        np.asarray(
            X3,
            dtype=np.float64
        ),
    )


def anomaly_flags(
    X,
    mean,
    inv_cov,
    threshold
):
    d = X - mean

    q = np.einsum(
        "ij,jk,ik->i",
        d,
        inv_cov,
        d
    )

    dist = np.sqrt(
        np.maximum(q, 0.0)
    )

    return dist > threshold


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


def metadata_for(name):
    key = name.replace(".log", "")

    meta = METADATA.get(key)

    if meta is None:
        key2 = key.replace(
            "_masquerade",
            ""
        )

        meta = METADATA.get(key2)

    return meta


def evaluate_attack(path):
    name = os.path.basename(path)

    meta = metadata_for(name)

    if not meta:
        return None

    interval = meta.get(
        "injection_interval"
    )

    injection_id = meta.get(
        "injection_id"
    )

    if interval is None:
        return None

    if (
        injection_id is None
        or injection_id == "XXX"
    ):
        return None

    target_id = int(
        injection_id,
        16
    )

    rows = parse_log(path)

    if len(rows) < WINDOW + 1:
        return None

    X2, X3 = extract_v2_v3(rows)

    f2 = anomaly_flags(
        X2,
        V2_MEAN,
        V2_INV,
        V2_THR
    )

    f3 = anomaly_flags(
        X3,
        V3_MEAN,
        V3_INV,
        V3_THR
    )

    t0 = rows[0][0]

    rel_ts = np.asarray(
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
        (rel_ts >= interval[0])
        &
        (rel_ts <= interval[1])
        &
        (ids == target_id)
    )

    n = int(mask.sum())

    if n == 0:
        return None

    return {
        "name": name,
        "family": family(name),
        "masquerade":
            "_masquerade" in name,
        "v2":
            float(f2[mask].mean()),
        "v3":
            float(f3[mask].mean()),
        "n": n,
    }


# =========================================================
# Attack evaluation
# =========================================================
results = []

for path in sorted(
    glob.glob(
        os.path.join(
            ATTACK_DIR,
            "*.log"
        )
    )
):
    r = evaluate_attack(path)

    if r is not None:
        results.append(r)


def family_summary(
    masquerade
):
    g2 = defaultdict(list)
    g3 = defaultdict(list)

    for r in results:

        if r["masquerade"] != masquerade:
            continue

        g2[
            r["family"]
        ].append(
            r["v2"]
        )

        g3[
            r["family"]
        ].append(
            r["v3"]
        )

    out = []

    for fam in sorted(g2):
        a = float(
            np.mean(g2[fam])
        )

        b = float(
            np.mean(g3[fam])
        )

        out.append(
            (
                fam,
                a,
                b,
                b - a
            )
        )

    return out


# =========================================================
# Ambient FPR
# =========================================================
ambient = []

for path in sorted(
    glob.glob(
        os.path.join(
            AMBIENT_DIR,
            "*.log"
        )
    )
):

    rows = parse_log(path)

    if len(rows) < WINDOW + 1:
        continue

    X2, X3 = extract_v2_v3(rows)

    f2 = anomaly_flags(
        X2,
        V2_MEAN,
        V2_INV,
        V2_THR
    )

    f3 = anomaly_flags(
        X3,
        V3_MEAN,
        V3_INV,
        V3_THR
    )

    ambient.append(
        (
            os.path.basename(path),
            float(f2.mean()),
            float(f3.mean()),
            len(rows)
        )
    )


# =========================================================
# Save report
# =========================================================
lines = []

lines.append(
    "=== ROAD Frozen V2 vs V3 Original/Masquerade ==="
)

lines.append(
    "ROAD 재학습 / 재정규화 / threshold 조정 없음."
)

lines.append(
    "Original injection과 masquerade를 분리 평가."
)

lines.append("")


for mode in [
    False,
    True
]:

    label = (
        "MASQUERADE"
        if mode
        else "ORIGINAL"
    )

    lines.append(
        f"=== {label} ==="
    )

    lines.append(
        "| Attack | Family | "
        "V2 TPR | V3 TPR | Delta | N |"
    )

    lines.append(
        "|---|---|---:|---:|---:|---:|"
    )

    for r in results:

        if r["masquerade"] != mode:
            continue

        lines.append(
            f"| {r['name']} "
            f"| {r['family']} "
            f"| {r['v2']:.2%} "
            f"| {r['v3']:.2%} "
            f"| {r['v3']-r['v2']:+.2%} "
            f"| {r['n']:,} |"
        )

    lines.append("")
    lines.append(
        f"=== {label} Family Mean ==="
    )

    for fam, a, b, d in family_summary(
        mode
    ):
        lines.append(
            f"{fam}: "
            f"V2={a:.2%}, "
            f"V3={b:.2%}, "
            f"Delta={d:+.2%}"
        )

    lines.append("")


if ambient:
    mean2 = float(
        np.mean(
            [x[1] for x in ambient]
        )
    )

    mean3 = float(
        np.mean(
            [x[2] for x in ambient]
        )
    )

    lines.append(
        "=== ROAD Ambient FPR ==="
    )

    lines.append(
        f"V2 mean FPR = "
        f"{mean2:.4%}"
    )

    lines.append(
        f"V3 mean FPR = "
        f"{mean3:.4%}"
    )

    lines.append(
        f"Delta = "
        f"{mean3-mean2:+.4%}"
    )

    lines.append("")

    for name, a, b, n in ambient:
        lines.append(
            f"{name}: "
            f"V2={a:.4%}, "
            f"V3={b:.4%}, "
            f"N={n:,}"
        )


out_path = os.path.join(
    RESULT_DIR,
    "f3_v2_vs_v3_original_masquerade.txt"
)


with open(
    out_path,
    "w",
    encoding="utf-8"
) as f:
    f.write(
        "\n".join(lines)
        + "\n"
    )


# =========================================================
# Console summary
# =========================================================
print("=" * 100)
print(
    "ROAD Frozen V2 vs V3 "
    "Original / Masquerade"
)
print("=" * 100)


for mode in [
    False,
    True
]:

    print(
        "\n"
        + (
            "MASQUERADE"
            if mode
            else "ORIGINAL"
        )
    )

    print("-" * 100)

    for fam, a, b, d in family_summary(
        mode
    ):

        print(
            f"{fam:30s} "
            f"V2={a:8.2%} "
            f"V3={b:8.2%} "
            f"Delta={d:+8.2%}"
        )


if ambient:
    print(
        "\nROAD Ambient"
    )

    print(
        f"V2 FPR = "
        f"{mean2:.4%}"
    )

    print(
        f"V3 FPR = "
        f"{mean3:.4%}"
    )


print(
    f"\n결과 저장: "
    f"{out_path}"
)
