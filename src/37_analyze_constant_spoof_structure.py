"""
37_analyze_constant_spoof_structure.py

목적
----
ROAD에서 V2/V3 모두 0%였던 constant-value spoofing의 구조를 분석한다.

대상
----
- max_engine_coolant_temp
- max_speedometer
- reverse_light_off
- reverse_light_on

비교
----
각 attack log에서 capture_metadata.json의
injection_interval + injection_id를 이용하여:

1) 공격 전 target-ID 정상 구간
2) 공격 주입 target-ID 구간
3) 공격 후 target-ID 구간

의 payload / temporal 통계를 비교한다.

주의
----
- detector feature를 ground truth 생성에 사용하지 않음.
- ROAD 재학습/threshold 조정 없음.
- 순수 구조 분석용 스크립트.
"""

import os
import re
import glob
import json
from collections import Counter

import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))

ROAD_ATTACKS = os.path.join(ROOT, "road_data", "attacks")
RESULT_DIR = os.path.join(ROOT, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

METADATA_PATH = os.path.join(
    ROAD_ATTACKS,
    "capture_metadata.json"
)

with open(METADATA_PATH, "r") as f:
    METADATA = json.load(f)

LINE_RE = re.compile(
    r"\(([\d.]+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)"
)

TARGET_PATTERNS = (
    "max_engine_coolant_temp",
    "max_speedometer",
    "reverse_light_off",
    "reverse_light_on",
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

            data = [
                int(hexdata[i:i+2], 16)
                for i in range(0, min(len(hexdata), 16), 2)
            ]
            data = (data + [0] * 8)[:8]

            rows.append(
                (
                    ts,
                    cid,
                    np.asarray(data, dtype=np.float64)
                )
            )

    return rows


def longest_true_run(values):
    """
    boolean array에서 가장 긴 True 연속 길이
    """
    best = 0
    cur = 0

    for x in values:
        if x:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0

    return best


def longest_same_payload_run(payloads):
    if len(payloads) == 0:
        return 0

    best = 1
    cur = 1

    for i in range(1, len(payloads)):
        if np.array_equal(payloads[i], payloads[i - 1]):
            cur += 1
            best = max(best, cur)
        else:
            cur = 1

    return best


def payload_tuple(x):
    return tuple(int(v) for v in x)


def summarize_segment(rows):
    """
    rows: [(ts, cid, payload), ...]
    동일 target ID만 들어온다고 가정.
    """

    result = {
        "n": len(rows),
    }

    if len(rows) == 0:
        return result

    ts = np.asarray([r[0] for r in rows], dtype=np.float64)
    payloads = np.stack([r[2] for r in rows])

    # ----------------------------------------
    # Payload-level
    # ----------------------------------------
    tuples = [payload_tuple(x) for x in payloads]
    counts = Counter(tuples)

    n_unique = len(counts)
    most_common_count = counts.most_common(1)[0][1]

    result["unique_payloads"] = n_unique
    result["unique_ratio"] = n_unique / len(payloads)
    result["dominant_payload_ratio"] = (
        most_common_count / len(payloads)
    )

    result["longest_same_payload_run"] = (
        longest_same_payload_run(payloads)
    )

    result["longest_same_payload_run_ratio"] = (
        result["longest_same_payload_run"] / len(payloads)
    )

    # ----------------------------------------
    # Consecutive payload delta
    # ----------------------------------------
    if len(payloads) >= 2:
        abs_delta = np.abs(
            np.diff(payloads, axis=0)
        ).sum(axis=1)

        zero_delta = (abs_delta == 0)

        result["zero_delta_ratio"] = float(
            zero_delta.mean()
        )

        result["mean_abs_delta"] = float(
            abs_delta.mean()
        )

        result["std_abs_delta"] = float(
            abs_delta.std()
        )

        result["median_abs_delta"] = float(
            np.median(abs_delta)
        )

        result["max_abs_delta"] = float(
            abs_delta.max()
        )

        result["longest_zero_delta_run"] = int(
            longest_true_run(zero_delta)
        )
    else:
        result["zero_delta_ratio"] = np.nan
        result["mean_abs_delta"] = np.nan
        result["std_abs_delta"] = np.nan
        result["median_abs_delta"] = np.nan
        result["max_abs_delta"] = np.nan
        result["longest_zero_delta_run"] = 0

    # ----------------------------------------
    # Inter-arrival time
    # ----------------------------------------
    if len(ts) >= 2:
        dt = np.diff(ts)
        dt = dt[dt >= 0]

        if len(dt):
            mean_dt = float(dt.mean())
            std_dt = float(dt.std())

            result["iat_mean_ms"] = mean_dt * 1000.0
            result["iat_std_ms"] = std_dt * 1000.0
            result["iat_cv"] = (
                std_dt / (mean_dt + 1e-12)
            )
            result["iat_min_ms"] = (
                float(dt.min()) * 1000.0
            )
            result["iat_max_ms"] = (
                float(dt.max()) * 1000.0
            )
        else:
            result["iat_mean_ms"] = np.nan
            result["iat_std_ms"] = np.nan
            result["iat_cv"] = np.nan
            result["iat_min_ms"] = np.nan
            result["iat_max_ms"] = np.nan
    else:
        result["iat_mean_ms"] = np.nan
        result["iat_std_ms"] = np.nan
        result["iat_cv"] = np.nan
        result["iat_min_ms"] = np.nan
        result["iat_max_ms"] = np.nan

    # ----------------------------------------
    # Byte-level statistics
    # ----------------------------------------
    for j in range(8):
        col = payloads[:, j]

        result[f"b{j}_mean"] = float(col.mean())
        result[f"b{j}_std"] = float(col.std())
        result[f"b{j}_min"] = float(col.min())
        result[f"b{j}_max"] = float(col.max())
        result[f"b{j}_range"] = float(
            col.max() - col.min()
        )
        result[f"b{j}_unique"] = int(
            len(np.unique(col))
        )

    return result


def fmt(x, pct=False, digits=4):
    if x is None:
        return "NA"

    try:
        if np.isnan(x):
            return "NA"
    except TypeError:
        pass

    if pct:
        return f"{100*x:.2f}%"

    if isinstance(x, (int, np.integer)):
        return f"{x:,}"

    return f"{float(x):.{digits}f}"


def ratio_change(a, b):
    """
    attack / pre ratio
    """
    if a is None or b is None:
        return np.nan

    try:
        if np.isnan(a) or np.isnan(b):
            return np.nan
    except TypeError:
        return np.nan

    if abs(a) < 1e-12:
        return np.nan

    return b / a


def segment_target_rows(
    rows,
    target_id,
    interval
):
    if not rows:
        return [], [], []

    capture_start = rows[0][0]

    attack_start = capture_start + float(interval[0])
    attack_end = capture_start + float(interval[1])

    target = [
        r for r in rows
        if r[1] == target_id
    ]

    pre = [
        r for r in target
        if r[0] < attack_start
    ]

    attack = [
        r for r in target
        if attack_start <= r[0] <= attack_end
    ]

    post = [
        r for r in target
        if r[0] > attack_end
    ]

    return pre, attack, post


def main():
    files = []

    for path in sorted(
        glob.glob(
            os.path.join(
                ROAD_ATTACKS,
                "*.log"
            )
        )
    ):
        name = os.path.basename(path)

        if any(
            p in name
            for p in TARGET_PATTERNS
        ):
            files.append(path)

    if not files:
        raise RuntimeError(
            "constant spoofing ROAD log를 찾지 못함"
        )

    output = []

    output.append(
        "=== Constant-value Spoof Structure Analysis ==="
    )
    output.append(
        "Ground truth: capture_metadata injection_interval + injection_id"
    )
    output.append(
        "Detector feature / threshold는 GT 생성에 사용하지 않음."
    )
    output.append("")

    compact_rows = []

    for path in files:
        name = os.path.basename(path)
        key = name.replace(".log", "")

        meta = METADATA.get(key)

        if not meta:
            output.append(
                f"[SKIP] metadata 없음: {name}"
            )
            continue

        interval = meta.get(
            "injection_interval"
        )
        injection_id = meta.get(
            "injection_id"
        )

        if interval is None:
            output.append(
                f"[SKIP] injection_interval 없음: {name}"
            )
            continue

        if (
            injection_id is None
            or injection_id == "XXX"
        ):
            output.append(
                f"[SKIP] injection_id 없음: {name}"
            )
            continue

        target_id = int(
            injection_id,
            16
        )

        rows = parse_log(path)

        pre, attack, post = (
            segment_target_rows(
                rows,
                target_id,
                interval
            )
        )

        s_pre = summarize_segment(pre)
        s_atk = summarize_segment(attack)
        s_post = summarize_segment(post)

        output.append("=" * 100)
        output.append(name)
        output.append(
            f"target CAN ID = 0x{target_id:X}"
        )
        output.append(
            f"injection interval = {interval}"
        )
        output.append(
            f"N: pre={len(pre):,}, "
            f"attack={len(attack):,}, "
            f"post={len(post):,}"
        )
        output.append("")

        output.append(
            "Metric                              "
            "PRE             ATTACK          POST"
        )
        output.append("-" * 100)

        key_metrics = [
            (
                "unique_payload_ratio",
                "unique_ratio",
                True,
            ),
            (
                "dominant_payload_ratio",
                "dominant_payload_ratio",
                True,
            ),
            (
                "zero_delta_ratio",
                "zero_delta_ratio",
                True,
            ),
            (
                "longest_same_run_ratio",
                "longest_same_payload_run_ratio",
                True,
            ),
            (
                "mean_abs_delta",
                "mean_abs_delta",
                False,
            ),
            (
                "std_abs_delta",
                "std_abs_delta",
                False,
            ),
            (
                "iat_mean_ms",
                "iat_mean_ms",
                False,
            ),
            (
                "iat_std_ms",
                "iat_std_ms",
                False,
            ),
            (
                "iat_cv",
                "iat_cv",
                False,
            ),
        ]

        for label, keyname, pct in key_metrics:
            output.append(
                f"{label:34s}"
                f"{fmt(s_pre.get(keyname), pct):>16s}"
                f"{fmt(s_atk.get(keyname), pct):>16s}"
                f"{fmt(s_post.get(keyname), pct):>16s}"
            )

        output.append("")
        output.append("Byte-level variance/range")
        output.append(
            "Byte        PRE std/range        "
            "ATTACK std/range      POST std/range"
        )
        output.append("-" * 100)

        for j in range(8):
            pre_val = (
                f"{fmt(s_pre.get(f'b{j}_std'))}/"
                f"{fmt(s_pre.get(f'b{j}_range'))}"
            )
            atk_val = (
                f"{fmt(s_atk.get(f'b{j}_std'))}/"
                f"{fmt(s_atk.get(f'b{j}_range'))}"
            )
            post_val = (
                f"{fmt(s_post.get(f'b{j}_std'))}/"
                f"{fmt(s_post.get(f'b{j}_range'))}"
            )

            output.append(
                f"B{j:<10d}"
                f"{pre_val:>20s}"
                f"{atk_val:>22s}"
                f"{post_val:>22s}"
            )

        # ----------------------------------------
        # 자동 진단 지표
        # ----------------------------------------
        pre_zero = s_pre.get(
            "zero_delta_ratio",
            np.nan
        )
        atk_zero = s_atk.get(
            "zero_delta_ratio",
            np.nan
        )

        pre_dom = s_pre.get(
            "dominant_payload_ratio",
            np.nan
        )
        atk_dom = s_atk.get(
            "dominant_payload_ratio",
            np.nan
        )

        pre_iat = s_pre.get(
            "iat_cv",
            np.nan
        )
        atk_iat = s_atk.get(
            "iat_cv",
            np.nan
        )

        zero_gap = (
            atk_zero - pre_zero
            if not (
                np.isnan(pre_zero)
                or np.isnan(atk_zero)
            )
            else np.nan
        )

        dom_gap = (
            atk_dom - pre_dom
            if not (
                np.isnan(pre_dom)
                or np.isnan(atk_dom)
            )
            else np.nan
        )

        iat_gap = (
            atk_iat - pre_iat
            if not (
                np.isnan(pre_iat)
                or np.isnan(atk_iat)
            )
            else np.nan
        )

        # byte variance가 가장 크게 줄어든 byte
        byte_std_delta = []

        for j in range(8):
            p = s_pre.get(
                f"b{j}_std",
                np.nan
            )
            a = s_atk.get(
                f"b{j}_std",
                np.nan
            )

            if (
                not np.isnan(p)
                and not np.isnan(a)
            ):
                byte_std_delta.append(
                    (a - p, j)
                )

        if byte_std_delta:
            strongest = min(
                byte_std_delta
            )
            strongest_byte = strongest[1]
            strongest_std_delta = strongest[0]
        else:
            strongest_byte = -1
            strongest_std_delta = np.nan

        output.append("")
        output.append("자동 비교")
        output.append(
            f"attack-pre zero_delta 차이: "
            f"{fmt(zero_gap, True)}"
        )
        output.append(
            f"attack-pre dominant_payload 차이: "
            f"{fmt(dom_gap, True)}"
        )
        output.append(
            f"attack-pre IAT CV 차이: "
            f"{fmt(iat_gap)}"
        )

        if strongest_byte >= 0:
            output.append(
                f"가장 variance가 감소한 byte: "
                f"B{strongest_byte}, "
                f"std delta={strongest_std_delta:.4f}"
            )

        output.append("")

        compact_rows.append(
            {
                "file": name,
                "target_id": f"0x{target_id:X}",
                "n_pre": len(pre),
                "n_attack": len(attack),
                "pre_zero": pre_zero,
                "attack_zero": atk_zero,
                "pre_dom": pre_dom,
                "attack_dom": atk_dom,
                "pre_unique": s_pre.get(
                    "unique_ratio",
                    np.nan
                ),
                "attack_unique": s_atk.get(
                    "unique_ratio",
                    np.nan
                ),
                "pre_iat_cv": pre_iat,
                "attack_iat_cv": atk_iat,
                "strongest_byte": strongest_byte,
                "strongest_std_delta": strongest_std_delta,
            }
        )

    # ==================================================
    # Compact final summary
    # ==================================================
    output.append("")
    output.append("=" * 120)
    output.append(
        "=== Compact Summary ==="
    )
    output.append("=" * 120)

    output.append(
        "| Attack | ID | zero-delta pre→attack | "
        "dominant pre→attack | unique pre→attack | "
        "IAT-CV pre→attack | strongest byte |"
    )
    output.append(
        "|---|---|---:|---:|---:|---:|---:|"
    )

    for r in compact_rows:
        output.append(
            f"| {r['file']} "
            f"| {r['target_id']} "
            f"| {fmt(r['pre_zero'], True)} → "
            f"{fmt(r['attack_zero'], True)} "
            f"| {fmt(r['pre_dom'], True)} → "
            f"{fmt(r['attack_dom'], True)} "
            f"| {fmt(r['pre_unique'], True)} → "
            f"{fmt(r['attack_unique'], True)} "
            f"| {fmt(r['pre_iat_cv'])} → "
            f"{fmt(r['attack_iat_cv'])} "
            f"| B{r['strongest_byte']} "
            f"|"
        )

    out_path = os.path.join(
        RESULT_DIR,
        "f3_constant_spoof_structure_analysis.txt"
    )

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "\n".join(output) + "\n"
        )

    print(
        "\n".join(output)
    )

    print(
        f"\n결과 저장: {out_path}"
    )


if __name__ == "__main__":
    main()
