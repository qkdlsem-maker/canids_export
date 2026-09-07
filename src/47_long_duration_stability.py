"""
Long-duration host/vcan stability preflight.

Same ICSim Mahalanobis feature/inference path as scripts 23/46.
No per-frame CSV and no per-alert printing.

Records one row per reporting interval:
- processed frames / FPS
- latency mean/p95/p99/max
- CPU
- RSS
- cumulative frames

This is host-side stability evidence only.
It does NOT replace physical MCU long-duration validation.
"""

import can
import time
import pickle
import os
import sys
import csv
from array import array

import numpy as np
import psutil


HERE = os.path.dirname(__file__)

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 3600.0
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
CHANNEL = "vcan0"

STATS_PATH = os.path.join(HERE, "..", "models", "icsim_stats.pkl")
MODEL_PATH = os.path.join(HERE, "..", "models", "maha_detector_icsim.npz")

CSV_PATH = os.path.join(
    HERE, "..", "results", "f5_long_duration_stability.csv"
)
TXT_PATH = os.path.join(
    HERE, "..", "results", "f5_long_duration_stability.txt"
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
    _, c = np.unique(pl, return_counts=True)
    p = c / c.sum()
    return -np.sum(p * np.log2(p + 1e-12))


os.makedirs(
    os.path.dirname(CSV_PATH),
    exist_ok=True,
)


proc = psutil.Process()
proc.cpu_percent(interval=None)


bus = can.interface.Bus(
    channel=CHANNEL,
    interface="socketcan",
)


id_window = []
last_payload_per_id = {}

total_frames = 0
total_alerts = 0

interval_frames = 0
interval_alerts = 0

# compact float32 buffer; reset every interval
lat_buf = array("f")

peak_rss_mb = 0.0

rows = []


start_wall = time.time()
start_mono = time.perf_counter()

interval_start = start_mono
next_report = start_mono + INTERVAL
end_mono = start_mono + DURATION


print(
    f"[long stability] channel={CHANNEL}, "
    f"duration={DURATION:.0f}s, interval={INTERVAL:.0f}s"
)


with open(CSV_PATH, "w", newline="") as cf:

    writer = csv.writer(cf)

    writer.writerow([
        "interval_index",
        "elapsed_sec",
        "interval_sec",
        "frames",
        "fps",
        "alerts",
        "latency_mean_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "latency_max_ms",
        "cpu_percent",
        "rss_mb",
        "cumulative_frames",
    ])

    interval_index = 0

    try:

        while time.perf_counter() < end_mono:

            msg = bus.recv(timeout=0.2)

            now = time.perf_counter()

            if msg is not None:

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
                        pl - last_payload_per_id[cid]
                    ).sum()

                else:
                    delta = 0.0


                last_payload_per_id[cid] = pl


                mu, sigma = id_delta_stats.get(
                    cid,
                    (g_delta_mean, g_delta_std),
                )

                delta_zscore = (
                    delta - mu
                ) / sigma


                if cid in id_byte_stats:

                    bmu, bstd = id_byte_stats[cid]

                    value_zscore = np.max(
                        np.abs(
                            (pl - bmu) / bstd
                        )
                    )

                else:
                    value_zscore = 0.0


                norm_id = cid / 2048.0


                x = np.array([
                    freq_in_window,
                    unique_ids_in_window,
                    is_unknown_id,
                    entropy,
                    mean_byte,
                    delta_zscore,
                    value_zscore,
                    norm_id,
                ])


                diff = x - MEAN_

                dist = np.sqrt(
                    diff @ INV_COV @ diff
                )


                if dist > THR:
                    total_alerts += 1
                    interval_alerts += 1


                latency_ms = (
                    time.perf_counter() - t0
                ) * 1000.0

                lat_buf.append(latency_ms)

                total_frames += 1
                interval_frames += 1


            now = time.perf_counter()

            if now >= next_report or now >= end_mono:

                interval_index += 1

                elapsed = now - start_mono
                interval_elapsed = now - interval_start

                fps = (
                    interval_frames / interval_elapsed
                    if interval_elapsed > 0
                    else 0.0
                )


                if len(lat_buf):

                    a = np.frombuffer(
                        lat_buf,
                        dtype=np.float32,
                    )

                    lat_mean = float(a.mean())
                    lat_p95 = float(np.percentile(a, 95))
                    lat_p99 = float(np.percentile(a, 99))
                    lat_max = float(a.max())

                else:

                    lat_mean = float("nan")
                    lat_p95 = float("nan")
                    lat_p99 = float("nan")
                    lat_max = float("nan")


                cpu = proc.cpu_percent(interval=None)

                rss_mb = (
                    proc.memory_info().rss
                    / (1024 * 1024)
                )

                peak_rss_mb = max(
                    peak_rss_mb,
                    rss_mb,
                )


                row = [
                    interval_index,
                    elapsed,
                    interval_elapsed,
                    interval_frames,
                    fps,
                    interval_alerts,
                    lat_mean,
                    lat_p95,
                    lat_p99,
                    lat_max,
                    cpu,
                    rss_mb,
                    total_frames,
                ]

                rows.append(row)
                writer.writerow(row)
                cf.flush()


                print(
                    f"[{interval_index:03d}] "
                    f"t={elapsed/60:.1f} min | "
                    f"fps={fps:,.1f} | "
                    f"p99={lat_p99:.4f} ms | "
                    f"cpu={cpu:.1f}% | "
                    f"rss={rss_mb:.1f} MB"
                )


                interval_frames = 0
                interval_alerts = 0
                lat_buf = array("f")

                interval_start = now

                while next_report <= now:
                    next_report += INTERVAL


    except KeyboardInterrupt:
        pass

    finally:

        try:
            bus.shutdown()
        except Exception:
            pass


end_wall = time.time()
elapsed_total = end_wall - start_wall


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

if rows:

    fps = np.array([r[4] for r in rows], dtype=float)
    p99 = np.array([r[8] for r in rows], dtype=float)
    cpu = np.array([r[10] for r in rows], dtype=float)
    rss = np.array([r[11] for r in rows], dtype=float)

    x = np.arange(len(rows), dtype=float)

    if len(rows) >= 2:
        rss_slope_per_interval = float(
            np.polyfit(x, rss, 1)[0]
        )
        fps_slope_per_interval = float(
            np.polyfit(x, fps, 1)[0]
        )
    else:
        rss_slope_per_interval = 0.0
        fps_slope_per_interval = 0.0

    rss_growth = float(rss[-1] - rss[0])

    first5 = fps[:min(5, len(fps))]
    last5 = fps[-min(5, len(fps)):]

    first5_mean = float(first5.mean())
    last5_mean = float(last5.mean())

    throughput_change_pct = (
        (last5_mean - first5_mean)
        / first5_mean * 100.0
        if first5_mean > 0
        else float("nan")
    )

else:

    fps = np.array([])
    p99 = np.array([])
    cpu = np.array([])
    rss = np.array([])

    rss_slope_per_interval = float("nan")
    fps_slope_per_interval = float("nan")
    rss_growth = float("nan")
    first5_mean = float("nan")
    last5_mean = float("nan")
    throughput_change_pct = float("nan")


summary = f"""=== Long-Duration Host/vcan Stability ===

Protocol
--------
Duration requested     : {DURATION:.1f} s
Reporting interval     : {INTERVAL:.1f} s
Environment            : Linux SocketCAN / vcan0
Detector               : ICSim Mahalanobis real-time path
Per-frame disk logging : NO
Physical MCU FIFO test : NO

Results
-------
Elapsed time           : {elapsed_total:.3f} s
Total processed frames : {total_frames:,}
Overall processing FPS : {total_frames / elapsed_total if elapsed_total else 0:.3f}

FPS mean               : {fps.mean() if len(fps) else float('nan'):.3f}
FPS minimum            : {fps.min() if len(fps) else float('nan'):.3f}
FPS maximum            : {fps.max() if len(fps) else float('nan'):.3f}

p99 latency mean       : {p99.mean() if len(p99) else float('nan'):.6f} ms
p99 latency maximum    : {p99.max() if len(p99) else float('nan'):.6f} ms

CPU mean               : {cpu.mean() if len(cpu) else float('nan'):.3f} %
RSS first              : {rss[0] if len(rss) else float('nan'):.3f} MB
RSS last               : {rss[-1] if len(rss) else float('nan'):.3f} MB
RSS peak               : {rss.max() if len(rss) else float('nan'):.3f} MB
RSS growth             : {rss_growth:.3f} MB

First-5 interval FPS   : {first5_mean:.3f}
Last-5 interval FPS    : {last5_mean:.3f}
Throughput change      : {throughput_change_pct:.6f} %

RSS slope / interval   : {rss_slope_per_interval:.6f} MB
FPS slope / interval   : {fps_slope_per_interval:.6f}

Interpretation
--------------
This is a host-side long-duration stability preflight.
It evaluates throughput, latency and process-resource drift.
It does NOT establish physical MCU CAN peripheral/FIFO long-duration stability.
"""


print()
print(summary)


with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write(summary)


print(f"[saved] {CSV_PATH}")
print(f"[saved] {TXT_PATH}")
