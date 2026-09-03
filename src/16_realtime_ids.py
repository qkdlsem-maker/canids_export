"""
vcan0을 실시간으로 리스닝하며, 메시지가 들어올 때마다 즉시 하이브리드 모델(LightGBM+Mahalanobis)로
판정. 판정 결과와 지연시간을 predictions.csv로 기록.

사용법: python3 16_realtime_ids.py [지속시간(초), 기본 90]
"""
import can, time, pickle, os, sys, csv
import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(__file__)
DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 90
CHANNEL = "vcan0"
WINDOW = 20

with open(os.path.join(HERE, "..", "models", "car_hacking_full_stats.pkl"), "rb") as f:
    STATS = pickle.load(f)
known_ids = STATS["known_ids"]
id_delta_stats = STATS["id_delta_stats"]
id_byte_stats = STATS["id_byte_stats"]
g_delta_mean, g_delta_std = STATS["global_delta_mean"], STATS["global_delta_std"]

lgbm = lgb.Booster(model_file=os.path.join(HERE, "..", "models", "lgbm_full.txt"))
npz = np.load(os.path.join(HERE, "..", "models", "maha_detector_full.npz"))
MAHA_MEAN, MAHA_INV_COV, MAHA_THR = npz["mean"], npz["inv_cov"], float(npz["thr"])

CLASSES = ["DoS", "Fuzzy", "R", "RPM", "gear"]  # sorted() 순서와 동일해야 함(학습때 c2i 기준)

def payload_entropy(pl):
    v, c = np.unique(pl, return_counts=True)
    p = c / c.sum()
    return -np.sum(p * np.log2(p + 1e-12))

def maha_dist(x):
    diff = x - MAHA_MEAN
    return np.sqrt(diff @ MAHA_INV_COV @ diff)

id_window = []
last_payload_per_id = {}

pred_path = os.path.join(HERE, "..", "results", "realtime_predictions.csv")
os.makedirs(os.path.dirname(pred_path), exist_ok=True)
pred_file = open(pred_path, "w", newline="")
writer = csv.writer(pred_file)
writer.writerow(["timestamp", "can_id", "final_label", "latency_ms",
                  "freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy",
                  "mean_byte", "delta_zscore", "value_zscore", "norm_id"])

bus = can.interface.Bus(channel=CHANNEL, bustype="socketcan")
print(f"[realtime_ids] {CHANNEL} 리스닝 시작, {DURATION}초간 실행")

t_end = time.time() + DURATION
n_msg, n_alert = 0, 0
try:
    while time.time() < t_end:
        msg = bus.recv(timeout=1.0)
        if msg is None:
            continue
        t0 = time.time()
        cid = msg.arbitration_id
        pl = np.array(list(msg.data) + [0] * 8)[:8].astype(float)

        id_window.append(cid)
        if len(id_window) > WINDOW:
            id_window.pop(0)

        freq_in_window = id_window.count(cid) / len(id_window)
        unique_ids_in_window = len(set(id_window))
        is_unknown_id = 0.0 if cid in known_ids else 1.0
        entropy = payload_entropy(pl)
        mean_byte = pl.mean()

        if cid in last_payload_per_id:
            delta = np.abs(pl - last_payload_per_id[cid]).sum()
        else:
            delta = 0.0
        last_payload_per_id[cid] = pl
        mu, sigma = id_delta_stats.get(cid, (g_delta_mean, g_delta_std))
        delta_zscore = (delta - mu) / sigma

        if cid in id_byte_stats:
            bmu, bstd = id_byte_stats[cid]
            value_zscore = np.max(np.abs((pl - bmu) / bstd))
        else:
            value_zscore = 0.0

        norm_id = cid / 2048.0
        x = np.array([freq_in_window, unique_ids_in_window, is_unknown_id, entropy,
                       mean_byte, delta_zscore, value_zscore, norm_id])

        proba = lgbm.predict(x.reshape(1, -1))[0]
        lgbm_label = CLASSES[int(np.argmax(proba))]

        if lgbm_label != "R":
            final_label = lgbm_label
        elif maha_dist(x) > MAHA_THR:
            final_label = "unknown_anomaly"
        else:
            final_label = "R"

        latency_ms = (time.time() - t0) * 1000
        n_msg += 1
        if final_label != "R":
            n_alert += 1
            print(f"[ALERT] t={time.time():.3f} id={hex(cid)} -> {final_label} ({latency_ms:.3f}ms)")

        writer.writerow([time.time(), cid, final_label, f"{latency_ms:.4f}",
                          f"{freq_in_window:.4f}", unique_ids_in_window, is_unknown_id,
                          f"{entropy:.4f}", f"{mean_byte:.2f}", f"{delta_zscore:.3f}",
                          f"{value_zscore:.3f}", f"{norm_id:.4f}"])
except KeyboardInterrupt:
    pass
finally:
    pred_file.close()
    print(f"[realtime_ids] 종료. 총 {n_msg:,}건 처리, 알림 {n_alert:,}건")
    print(f"[realtime_ids] 저장: {pred_path}")
