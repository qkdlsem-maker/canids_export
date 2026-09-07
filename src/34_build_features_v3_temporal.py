
"""
V3 temporal feature builder

V2 6개 유지:
  freq_in_window
  unique_ids_in_window
  entropy
  mean_byte
  global_delta_zscore
  global_value_zscore

V3 temporal 5개 추가:
  rolling_delta_mean
  rolling_delta_std
  payload_change_rate
  payload_run_length
  interarrival_cv

중요:
- CAN ID 값 자체는 feature로 넣지 않음.
- ID는 동일 신호의 시간 이력을 추적하는 state key로만 사용.
- V2에서 HCRL train-normal로 fit한 idagnostic_stats.pkl을 그대로 사용.
- ROAD 데이터는 이 단계에서 전혀 사용하지 않음.
"""

import csv
import os
import time
import pickle
from collections import defaultdict, deque

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))

DATA_DIR = os.path.join(ROOT, "data_full")
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(DATA_DIR, "features_v3_parts")

os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "DoS": "DoS_dataset.csv",
    "Fuzzy": "Fuzzy_dataset.csv",
    "gear": "gear_dataset.csv",
    "RPM": "RPM_dataset.csv",
}

TRAIN_FRAC = 0.7
TEMP_WINDOW = 8

V2_STATS_PATH = os.path.join(MODEL_DIR, "idagnostic_stats.pkl")

if not os.path.exists(V2_STATS_PATH):
    raise FileNotFoundError(
        f"{V2_STATS_PATH} 없음. 먼저 python3 src/13_build_features_v2.py 실행"
    )

with open(V2_STATS_PATH, "rb") as f:
    STATS = pickle.load(f)

WINDOW = int(STATS["window"])
G_DELTA_MEAN = float(STATS["global_delta_mean"])
G_DELTA_STD = float(STATS["global_delta_std"])
BYTE_MEAN = np.asarray(STATS["byte_pos_mean"], dtype=float)
BYTE_STD = np.asarray(STATS["byte_pos_std"], dtype=float)

FEATS = [
    "freq_in_window",
    "unique_ids_in_window",
    "entropy",
    "mean_byte",
    "global_delta_zscore",
    "global_value_zscore",
    "rolling_delta_mean",
    "rolling_delta_std",
    "payload_change_rate",
    "payload_run_length",
    "interarrival_cv",
]


def payload_entropy(b):
    _, c = np.unique(b, return_counts=True)
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def parse_file(path):
    ts_list = []
    id_list = []
    data_list = []
    label_list = []

    with open(path, "r", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue

            try:
                ts = float(row[0])
                cid = int(row[1], 16)
                dlc = int(row[2])
                raw = row[3:3 + dlc]
                data = [int(x, 16) for x in raw]
            except (ValueError, IndexError):
                continue

            data = (data + [0] * 8)[:8]

            flag = row[-1].strip()
            label = "R" if flag == "R" else "__ATTACK__"

            ts_list.append(ts)
            id_list.append(cid)
            data_list.append(data)
            label_list.append(label)

    return (
        np.asarray(ts_list, dtype=np.float64),
        np.asarray(id_list, dtype=np.int64),
        np.asarray(data_list, dtype=np.int16),
        np.asarray(label_list, dtype=object),
    )


def build_features(ts, ids, data):
    n = len(ids)
    X = np.zeros((n, len(FEATS)), dtype=np.float32)

    # V2 window
    id_window = deque(maxlen=WINDOW)

    # Per-ID state.
    # ID 값을 feature로 출력하지 않고 시간 이력 추적용으로만 사용.
    last_payload = {}
    last_ts = {}

    delta_hist = defaultdict(lambda: deque(maxlen=TEMP_WINDOW))
    iat_hist = defaultdict(lambda: deque(maxlen=TEMP_WINDOW))

    run_length = defaultdict(int)

    for i in range(n):
        cid = int(ids[i])
        pl = data[i].astype(float)
        cur_ts = float(ts[i])

        # ---------- V2 ----------
        id_window.append(cid)

        freq = sum(x == cid for x in id_window) / len(id_window)
        unique_ids = len(set(id_window))

        ent = payload_entropy(pl)
        mean_byte = float(pl.mean())

        if cid in last_payload:
            delta = float(np.abs(pl - last_payload[cid]).sum())
            same_payload = bool(np.array_equal(pl, last_payload[cid]))
        else:
            delta = 0.0
            same_payload = False

        global_delta_z = (delta - G_DELTA_MEAN) / G_DELTA_STD
        global_value_z = float(
            np.max(np.abs((pl - BYTE_MEAN) / BYTE_STD))
        )

        # ---------- V3 temporal ----------
        dh = delta_hist[cid]
        dh.append(delta)

        rolling_delta_mean = float(np.mean(dh))
        rolling_delta_std = float(np.std(dh))

        # 최근 transition 중 payload가 변한 비율
        payload_change_rate = float(
            np.mean(np.asarray(dh) > 0.0)
        )

        # 같은 payload가 연속된 길이.
        # 다른 차량에서도 범위가 같도록 TEMP_WINDOW로 정규화.
        if cid not in last_payload:
            run_length[cid] = 1
        elif same_payload:
            run_length[cid] += 1
        else:
            run_length[cid] = 1

        payload_run_length = (
            min(run_length[cid], TEMP_WINDOW) / TEMP_WINDOW
        )

        # 동일 CAN ID의 inter-arrival variability
        if cid in last_ts:
            dt = max(cur_ts - last_ts[cid], 0.0)
            iat_hist[cid].append(dt)

        ih = iat_hist[cid]

        if len(ih) >= 2:
            m = float(np.mean(ih))
            s = float(np.std(ih))
            interarrival_cv = s / (m + 1e-9)
        else:
            interarrival_cv = 0.0

        # 너무 큰 outlier가 covariance를 지배하지 않도록
        # 물리적 의미 없는 극단 CV만 clipping.
        interarrival_cv = min(interarrival_cv, 20.0)

        X[i] = [
            freq,
            unique_ids,
            ent,
            mean_byte,
            global_delta_z,
            global_value_z,
            rolling_delta_mean,
            rolling_delta_std,
            payload_change_rate,
            payload_run_length,
            interarrival_cv,
        ]

        last_payload[cid] = pl.copy()
        last_ts[cid] = cur_ts

    return X


def main():
    t_all = time.time()

    for attack_name, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)

        print(f"\n[{attack_name}] {fname} 파싱...")
        t0 = time.time()

        ts, ids, data, labels = parse_file(path)

        # 실제 공격 label 명 복원
        labels = np.where(labels == "__ATTACK__", attack_name, "R")

        n = len(ids)
        split = int(n * TRAIN_FRAC)

        print(f"  rows={n:,}, split={split:,}")
        print("  V3 feature 계산...")

        X = build_features(ts, ids, data)

        df = pd.DataFrame(X, columns=FEATS)
        df["Label"] = labels

        is_test = np.zeros(n, dtype=bool)
        is_test[split:] = True
        df["is_test_region"] = is_test

        out = os.path.join(
            OUT_DIR,
            f"{attack_name}_features_v3.parquet"
        )
        df.to_parquet(out, index=False)

        print(f"  저장: {out}")
        print(f"  shape={df.shape}, {time.time()-t0:.1f}s")

    meta = {
        "feature_names": FEATS,
        "window": WINDOW,
        "temp_window": TEMP_WINDOW,
        "v2_stats_source": "models/idagnostic_stats.pkl",
    }

    with open(os.path.join(MODEL_DIR, "v3_feature_config.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("\n" + "=" * 70)
    print("V3 생성 완료")
    print(f"feature 수: {len(FEATS)}")
    print(f"총 시간: {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
