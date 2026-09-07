"""
ROAD Frozen V2 vs V4

핵심:
1. original injection log
2. masquerade log

를 반드시 분리해서 평가한다.

original:
  timing + payload 변화 모두 존재 가능

masquerade:
  injection frequency shortcut 제거
  payload-content 탐지 능력 확인

ROAD 재학습/재캘리브레이션 없음.
"""

import os
import re
import glob
import json
import pickle
from collections import defaultdict, deque

import numpy as np


HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(
    os.path.join(HERE, "..")
)

MODEL_DIR = os.path.join(
    ROOT,
    "models"
)

RESULT_DIR = os.path.join(
    ROOT,
    "results"
)

ATTACK_DIR = os.path.join(
    ROOT,
    "road_data",
    "attacks"
)

AMBIENT_DIR = os.path.join(
    ROOT,
    "road_data",
    "ambient"
)


# ==============================================
# Frozen HCRL stats
# ==============================================

with open(
    os.path.join(
        MODEL_DIR,
        "idagnostic_stats.pkl"
    ),
    "rb"
) as f:
    STATS = pickle.load(f)


WINDOW = int(
    STATS["window"]
)

G_DELTA_MEAN = float(
    STATS["global_delta_mean"]
)

G_DELTA_STD = float(
    STATS["global_delta_std"]
)

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


# ==============================================
# V2
# ==============================================

v2 = np.load(
    os.path.join(
        MODEL_DIR,
        "maha_v2.npz"
    )
)

V2_MEAN = v2["mean"]
V2_INV = v2["inv_cov"]
V2_THR = float(
    v2["thr"]
)


# ==============================================
# V4
# ==============================================

with open(
    os.path.join(
        MODEL_DIR,
        "v4_feature_config.pkl"
    ),
    "rb"
) as f:
    CFG = pickle.load(f)


BYTE_WINDOW = int(
    CFG["byte_window"]
)


v4 = np.load(
    os.path.join(
        MODEL_DIR,
        "maha_v4_byte_temporal.npz"
    ),
    allow_pickle=True
)


V4_MEAN = v4["mean"]
V4_INV = v4["inv_cov"]
V4_THR = float(
    v4["thr"]
)


# ==============================================
# metadata
# ==============================================

with open(
    os.path.join(
        ATTACK_DIR,
        "capture_metadata.json"
    ),
    "r"
) as f:
    METADATA = json.load(f)


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
        errors="ignore"
    ) as f:

        for line in f:

            m = LINE_RE.search(line)

            if not m:
                continue

            ts = float(
                m.group(1)
            )

            cid = int(
                m.group(2),
                16
            )

            hexdata = m.group(3)

            data = [
                int(
                    hexdata[i:i+2],
                    16
                )
                for i in range(
                    0,
                    min(
                        len(hexdata),
                        16
                    ),
                    2
                )
            ]

            data = (
                data + [0] * 8
            )[:8]

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

    _, c = np.unique(
        pl,
        return_counts=True
    )

    p = c / c.sum()

    return float(
        -np.sum(
            p * np.log2(
                p + 1e-12
            )
        )
    )


def extract_v2_v4(rows):

    X2 = []
    X4 = []

    id_window = deque(
        maxlen=WINDOW
    )

    last_payload = {}

    payload_hist = defaultdict(
        lambda: deque(
            maxlen=BYTE_WINDOW
        )
    )

    for ts, cid, pl in rows:

        id_window.append(cid)

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

            byte_delta = np.abs(
                pl
                - last_payload[cid]
            )

            total_delta = float(
                byte_delta.sum()
            )

            max_byte_abs_delta = float(
                byte_delta.max()
            )

        else:

            total_delta = 0.0
            max_byte_abs_delta = 0.0

        delta_z = (
            total_delta
            - G_DELTA_MEAN
        ) / (
            G_DELTA_STD
            + 1e-12
        )

        value_z = float(
            np.max(
                np.abs(
                    (
                        pl
                        - BYTE_MEAN
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

        # ------------------------------
        # V4
        # ------------------------------

        hist = payload_hist[cid]

        hist.append(
            pl.copy()
        )

        if len(hist) >= 2:

            H = np.stack(
                list(hist),
                axis=0
            )

            byte_stds = np.std(
                H,
                axis=0
            )

            std_max = float(
                byte_stds.max()
            )

            std_mean = float(
                byte_stds.mean()
            )

            std_range = float(
                byte_stds.max()
                - byte_stds.min()
            )

        else:

            std_max = 0.0
            std_mean = 0.0
            std_range = 0.0

        X4.append(
            base
            + [
                max_byte_abs_delta,
                std_max,
                std_mean,
                std_range,
            ]
        )

        last_payload[cid] = (
            pl.copy()
        )

    return (
        np.asarray(
            X2,
            dtype=np.float64
        ),
        np.asarray(
            X4,
            dtype=np.float64
        ),
    )


def flags(
    X,
    mean,
    inv,
    threshold
):

    d = X - mean

    q = np.einsum(
        "ij,jk,ik->i",
        d,
        inv,
        d
    )

    dist = np.sqrt(
        np.maximum(
            q,
            0.0
        )
    )

    return (
        dist > threshold
    )


def attack_family(name):

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

    key = name.replace(
        ".log",
        ""
    )

    meta = METADATA.get(key)

    # fallback:
    # masquerade metadata가 별도 없을 경우
    if meta is None:

        key2 = key.replace(
            "_masquerade",
            ""
        )

        meta = METADATA.get(
            key2
        )

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

    X2, X4 = extract_v2_v4(
        rows
    )

    f2 = flags(
        X2,
        V2_MEAN,
        V2_INV,
        V2_THR
    )

    f4 = flags(
        X4,
        V4_MEAN,
        V4_INV,
        V4_THR
    )

    t0 = rows[0][0]

    rel_ts = np.asarray(
        [
            row[0] - t0
            for row in rows
        ]
    )

    ids = np.asarray(
        [
            row[1]
            for row in rows
        ]
    )

    mask = (
        (rel_ts >= interval[0])
        &
        (rel_ts <= interval[1])
        &
        (ids == target_id)
    )

    n = int(
        mask.sum()
    )

    if n == 0:
        return None

    return {
        "name": name,
        "family":
            attack_family(name),
        "masquerade":
            "_masquerade"
            in name,
        "v2":
            float(
                f2[mask].mean()
            ),
        "v4":
            float(
                f4[mask].mean()
            ),
        "n": n,
    }


print("=" * 110)
print(
    "ROAD Frozen V2 vs V4 "
    "Byte-Level Temporal"
)
print("=" * 110)


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


# ==============================================
# Original / masquerade tables
# ==============================================

for mode in [
    False,
    True
]:

    title = (
        "MASQUERADE"
        if mode
        else "ORIGINAL INJECTION"
    )

    print(
        "\n"
        + "=" * 110
    )

    print(title)

    print(
        "=" * 110
    )

    print(
        f"{'Attack':50s}"
        f"{'V2':>10s}"
        f"{'V4':>10s}"
        f"{'Delta':>10s}"
        f"{'N':>10s}"
    )

    print("-" * 110)

    for r in results:

        if (
            r["masquerade"]
            != mode
        ):
            continue

        print(
            f"{r['name']:50s}"
            f"{r['v2']:10.2%}"
            f"{r['v4']:10.2%}"
            f"{r['v4']-r['v2']:+10.2%}"
            f"{r['n']:10,d}"
        )


# ==============================================
# family mean 분리
# ==============================================

def family_summary(
    masquerade
):

    grouped2 = defaultdict(list)
    grouped4 = defaultdict(list)

    for r in results:

        if (
            r["masquerade"]
            != masquerade
        ):
            continue

        grouped2[
            r["family"]
        ].append(
            r["v2"]
        )

        grouped4[
            r["family"]
        ].append(
            r["v4"]
        )

    out = []

    for fam in sorted(
        grouped2
    ):

        a = float(
            np.mean(
                grouped2[fam]
            )
        )

        b = float(
            np.mean(
                grouped4[fam]
            )
        )

        out.append(
            (
                fam,
                a,
                b,
                b-a
            )
        )

    return out


# ==============================================
# Ambient FPR
# ==============================================

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

    X2, X4 = extract_v2_v4(
        rows
    )

    f2 = flags(
        X2,
        V2_MEAN,
        V2_INV,
        V2_THR
    )

    f4 = flags(
        X4,
        V4_MEAN,
        V4_INV,
        V4_THR
    )

    ambient.append(
        (
            os.path.basename(path),
            float(f2.mean()),
            float(f4.mean()),
            len(rows)
        )
    )


# ==============================================
# Save final
# ==============================================

lines = []

lines.append(
    "=== ROAD Frozen V2 vs V4 "
    "Byte-Level Temporal ==="
)

lines.append(
    "ROAD 재학습/threshold 조정 없음."
)

lines.append(
    "Original injection과 "
    "masquerade를 분리 평가."
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
        "V2 TPR | V4 TPR | "
        "Delta | N |"
    )

    lines.append(
        "|---|---|---:|---:|---:|---:|"
    )

    for r in results:

        if (
            r["masquerade"]
            != mode
        ):
            continue

        lines.append(
            f"| {r['name']} "
            f"| {r['family']} "
            f"| {r['v2']:.2%} "
            f"| {r['v4']:.2%} "
            f"| {r['v4']-r['v2']:+.2%} "
            f"| {r['n']:,} |"
        )

    lines.append("")
    lines.append(
        f"=== {label} Family Mean ==="
    )

    for (
        fam,
        a,
        b,
        d
    ) in family_summary(mode):

        lines.append(
            f"{fam}: "
            f"V2={a:.2%}, "
            f"V4={b:.2%}, "
            f"Delta={d:+.2%}"
        )

    lines.append("")


if ambient:

    mean2 = float(
        np.mean(
            [
                x[1]
                for x in ambient
            ]
        )
    )

    mean4 = float(
        np.mean(
            [
                x[2]
                for x in ambient
            ]
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
        f"V4 mean FPR = "
        f"{mean4:.4%}"
    )

    lines.append(
        f"Delta = "
        f"{mean4-mean2:+.4%}"
    )

    lines.append("")

    for (
        name,
        a,
        b,
        n
    ) in ambient:

        lines.append(
            f"{name}: "
            f"V2={a:.4%}, "
            f"V4={b:.4%}, "
            f"N={n:,}"
        )


out_path = os.path.join(
    RESULT_DIR,
    "f3_v2_vs_v4_byte_temporal_road.txt"
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


print(
    "\n"
    + "=" * 110
)

print(
    "Family summary"
)

print(
    "=" * 110
)


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

    for (
        fam,
        a,
        b,
        d
    ) in family_summary(mode):

        print(
            f"{fam:30s} "
            f"V2={a:8.2%} "
            f"V4={b:8.2%} "
            f"Delta={d:+8.2%}"
        )


if ambient:

    print(
        "\nROAD Ambient:"
    )

    print(
        f"V2 FPR="
        f"{mean2:.4%}"
    )

    print(
        f"V4 FPR="
        f"{mean4:.4%}"
    )


print(
    f"\n결과 저장: "
    f"{out_path}"
)
