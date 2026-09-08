import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "data" / "demo_bundle.json"

LINE_RE = re.compile(
    r"\(([\d.]+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)"
)


def parse_candump(path):
    rows = []

    with Path(path).open(errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            m = LINE_RE.search(line)
            if not m:
                continue

            ts = float(m.group(1))
            cid = int(m.group(2), 16)
            payload = m.group(3).upper()

            # Match final V2 semantics: max 8 bytes, then zero-pad.
            payload = payload[:16]
            payload = payload + ("0" * (16 - len(payload)))

            rows.append({
                "line_no": line_no,
                "timestamp": ts,
                "can_id": cid,
                "can_id_hex": f"{cid:03X}",
                "payload_hex": payload,
            })

    return rows


def sample_evenly(rows, n):
    if len(rows) <= n:
        return rows

    idxs = [
        round(i * (len(rows) - 1) / (n - 1))
        for i in range(n)
    ]
    return [rows[i] for i in idxs]


# ------------------------------------------------------------
# 1. ICSim normal
# ------------------------------------------------------------
normal_rows = parse_candump(
    ROOT / "results" / "icsim_normal_candump.log"
)

# Keep a contiguous sequence so streaming/window semantics remain real.
normal_demo = normal_rows[:300]


# ------------------------------------------------------------
# 2. ICSim independent-GT attacks
# ------------------------------------------------------------
gt = pd.read_csv(
    ROOT / "results" / "icsim_attack_ground_truth_matched.csv"
)

attack_scenarios = {}

for attack_name in ["DoS", "Fuzzy", "spoof"]:
    sub = gt[
        (gt["attack_type"] == attack_name) &
        (gt["matched"] == 1)
    ].copy()

    rows = []

    for _, r in sub.iterrows():
        payload = str(r["payload_hex"]).strip().upper()

        # pandas may turn all-zero payload into "0".
        payload = payload[:16].zfill(16)

        rows.append({
            "line_no": int(r["candump_line_no"]),
            "timestamp": float(r["candump_ts"]),
            "can_id": int(r["can_id"]),
            "can_id_hex": f"{int(r['can_id']):03X}",
            "payload_hex": payload,
            "ground_truth": attack_name,
        })

    # Keep chronological contiguous data rather than cherry-picking
    # frames based on model output.
    attack_scenarios[attack_name] = rows[:300]


# ------------------------------------------------------------
# 3. ROAD external zero-day / correlated signal
# ------------------------------------------------------------
road_path = (
    ROOT / "road_data" / "attacks" /
    "correlated_signal_attack_1.log"
)

road_rows = parse_candump(road_path)

capture_start = road_rows[0]["timestamp"]

inj_start_rel = 9.191851
inj_end_rel = 30.050109

inj_start = capture_start + inj_start_rel
inj_end = capture_start + inj_end_rel

# Include 100 frames immediately before the official injection interval
# as warm-up context, then 300 chronological frames from the actual
# injection interval. No selection based on IDS result.
pre = [r for r in road_rows if r["timestamp"] < inj_start]
during = [
    r for r in road_rows
    if inj_start <= r["timestamp"] <= inj_end
]

road_warmup = pre[-100:]
road_attack = during[:300]

for r in road_warmup:
    r["ground_truth"] = "pre-injection"

for r in road_attack:
    r["ground_truth"] = "correlated_signal"

road_demo = road_warmup + road_attack


bundle = {
    "schema_version": 1,
    "description": (
        "Offline judging demo bundle. Frames are real captured CAN "
        "traffic; IDS decisions are intentionally not stored and must "
        "be recomputed by the frozen V2 pipeline at runtime."
    ),
    "scenarios": {
        "normal": {
            "display_name": "Normal",
            "source": "ICSim normal candump",
            "frames": normal_demo,
        },
        "dos": {
            "display_name": "DoS",
            "source": "ICSim independent ground truth",
            "frames": attack_scenarios["DoS"],
        },
        "fuzzy": {
            "display_name": "Fuzzy",
            "source": "ICSim independent ground truth",
            "frames": attack_scenarios["Fuzzy"],
        },
        "spoof": {
            "display_name": "Spoof",
            "source": "ICSim independent ground truth",
            "note": (
                "Spoof is the injected scenario label, not a native "
                "LightGBM training class."
            ),
            "frames": attack_scenarios["spoof"],
        },
        "zero_day": {
            "display_name": "Zero-day / External Vehicle",
            "source": "ROAD correlated_signal_attack_1",
            "note": (
                "Uses official ROAD injection interval "
                "[9.191851, 30.050109] seconds. "
                "First 100 frames are pre-injection warm-up; "
                "next 300 are chronological injection-period frames. "
                "Frames were not selected by model outcome."
            ),
            "injection_id": "0x6E0",
            "injection_payload": "595945450000FFFF",
            "injection_interval_sec": [
                9.191851,
                30.050109
            ],
            "warmup_frames": len(road_warmup),
            "frames": road_demo,
        },
    },
}

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w") as f:
    json.dump(bundle, f, indent=2)

print(f"saved: {OUT}")

for key, scenario in bundle["scenarios"].items():
    frames = scenario["frames"]
    print(
        f"{key:10s}",
        f"frames={len(frames):4d}",
        f"first={frames[0]['can_id_hex']}#{frames[0]['payload_hex']}",
        f"last={frames[-1]['can_id_hex']}#{frames[-1]['payload_hex']}",
    )
