"""
Host/vcan CAN-load stress detector.

Purpose:
- Measure processing capacity of the current ICSim Mahalanobis IDS pipeline.
- Same feature extraction and Mahalanobis calculation as 23_icsim_realtime_detect.py.
- NO per-frame CSV I/O.
- NO per-alert console printing.
- Measure:
    received/processed frames
    processing FPS
    latency mean/p50/p95/p99/max
    CPU
    peak RSS

This is HOST-SIDE / vcan stress evidence.
It does NOT prove physical MCU CAN-controller FIFO overflow behavior.
"""

import can
import time
import pickle
import os
import sys
import threading

import numpy as np
import psutil


HERE = os.path.dirname(__file__)

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
CHANNEL = "vcan0"

STATS_PATH = os.path.join(
    HERE, "..", "models", "icsim_stats.pkl"
)

MODEL_PATH = os.path.join(
    HERE, "..", "models", "maha_detector_icsim.npz"
)

OUT_PATH = os.path.join(
    HERE, "..", "results", "can_load_stress_detector.txt"
)


with open(STATS_PATH, "rb") as f:
    STATS = pickle.load(f)

known_ids = STATS["known_ids"]
id_delta_stats = STATS["id_delta_stats"]
id_byte_stats = STATS["id_byte_stats"]
g_delta_mean = STATS["global_delta_mean"]
g_delta_std = STATS["global_delta_std"]
WINDOW = STATS["window"]


npz = np.load(MODEL_PATH)

MEAN_ = npz["mean"]
INV_COV = npz["inv_cov"]
THR = float(npz["thr"])


def payload_entropy(pl):
    _, c = np.unique(
        pl,
        return_counts=True,
    )

    p = c / c.sum()

    return -np.sum(
        p * np.log2(p + 1e-12)
    )


# ---------------------------------------------------------
# Resource monitoring
# ---------------------------------------------------------

proc = psutil.Process()

peak_rss = 0
cpu_samples = []

stop_monitor = threading.Event()


def monitor():
    global peak_rss

    proc.cpu_percent(interval=None)

    while not stop_monitor.is_set():

        rss = proc.memory_info().rss

        peak_rss = max(
            peak_rss,
            rss,
        )

        cpu_samples.append(
            proc.cpu_percent(interval=None)
        )

        time.sleep(0.25)


threading.Thread(
    target=monitor,
    daemon=True,
).start()


# ---------------------------------------------------------
# Detector
# ---------------------------------------------------------

id_window = []
last_payload_per_id = {}

latencies_ms = []

n_msg = 0
n_alert = 0


# Larger SocketCAN receive buffer where supported.
bus = can.interface.Bus(
    channel=CHANNEL,
    interface="socketcan",
)


print(
    f"[stress detector] {CHANNEL}, "
    f"duration={DURATION:.1f}s"
)

wall_start = time.time()
mono_start = time.perf_counter()

t_end = mono_start + DURATION


try:

    while time.perf_counter() < t_end:

        msg = bus.recv(timeout=0.2)

        if msg is None:
            continue

        t0 = time.perf_counter()

        cid = msg.arbitration_id

        pl = np.array(
            list(msg.data) + [0] * 8
        )[:8].astype(float)


        id_window.append(cid)

        if len(id_window) > WINDOW:
            id_window.pop(0)


        freq_in_window = (
            id_window.count(cid)
            / len(id_window)
        )

        unique_ids_in_window = len(
            set(id_window)
        )

        is_unknown_id = (
            0.0
            if cid in known_ids
            else 1.0
        )

        entropy = payload_entropy(pl)

        mean_byte = pl.mean()


        if cid in last_payload_per_id:

            delta = np.abs(
                pl
                - last_payload_per_id[cid]
            ).sum()

        else:
            delta = 0.0


        last_payload_per_id[cid] = pl


        mu, sigma = id_delta_stats.get(
            cid,
            (
                g_delta_mean,
                g_delta_std,
            ),
        )

        delta_zscore = (
            delta - mu
        ) / sigma


        if cid in id_byte_stats:

            bmu, bstd = id_byte_stats[cid]

            value_zscore = np.max(
                np.abs(
                    (pl - bmu)
                    / bstd
                )
            )

        else:
            value_zscore = 0.0


        norm_id = cid / 2048.0


        x = np.array(
            [
                freq_in_window,
                unique_ids_in_window,
                is_unknown_id,
                entropy,
                mean_byte,
                delta_zscore,
                value_zscore,
                norm_id,
            ],
            dtype=float,
        )


        diff = x - MEAN_

        dist = np.sqrt(
            diff
            @ INV_COV
            @ diff
        )


        if dist > THR:
            n_alert += 1


        latency_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        latencies_ms.append(
            latency_ms
        )

        n_msg += 1


finally:

    wall_end = time.time()

    try:
        bus.shutdown()
    except Exception:
        pass

    stop_monitor.set()
    time.sleep(0.3)


elapsed = wall_end - wall_start

arr = np.asarray(
    latencies_ms,
    dtype=float,
)


if len(arr):

    mean_latency = arr.mean()

    p50 = np.percentile(arr, 50)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)

    max_latency = arr.max()

else:

    mean_latency = float("nan")
    p50 = float("nan")
    p95 = float("nan")
    p99 = float("nan")
    max_latency = float("nan")


processing_fps = (
    n_msg / elapsed
    if elapsed > 0
    else 0.0
)

peak_mb = peak_rss / (
    1024 * 1024
)

avg_cpu = (
    np.mean(cpu_samples)
    if cpu_samples
    else 0.0
)

max_cpu = (
    np.max(cpu_samples)
    if cpu_samples
    else 0.0
)


lines = [
    "=== CAN Load Stress Detector ===",
    f"wall_start={wall_start:.9f}",
    f"wall_end={wall_end:.9f}",
    f"elapsed_sec={elapsed:.6f}",
    f"processed_frames={n_msg}",
    f"processing_fps={processing_fps:.3f}",
    f"alerts={n_alert}",
    f"latency_mean_ms={mean_latency:.6f}",
    f"latency_p50_ms={p50:.6f}",
    f"latency_p95_ms={p95:.6f}",
    f"latency_p99_ms={p99:.6f}",
    f"latency_max_ms={max_latency:.6f}",
    f"peak_rss_mb={peak_mb:.3f}",
    f"avg_cpu_percent={avg_cpu:.3f}",
    f"max_cpu_percent={max_cpu:.3f}",
]


text = "\n".join(lines)

print()
print(text)


os.makedirs(
    os.path.dirname(OUT_PATH),
    exist_ok=True,
)

with open(
    OUT_PATH,
    "w",
    encoding="utf-8",
) as f:
    f.write(
        text + "\n"
    )
