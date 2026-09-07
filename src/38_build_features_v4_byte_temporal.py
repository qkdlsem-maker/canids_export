"""
V4 = V2 + byte-level temporal summary

V2:
  freq_in_window
  unique_ids_in_window
  entropy
  mean_byte
  global_delta_zscore
  global_value_zscore

V4 추가:
  max_byte_abs_delta
  rolling_byte_std_max
  rolling_byte_std_mean
  rolling_byte_std_range

중요:
- CAN ID 자체는 feature가 아님.
- ID는 동일 신호의 시간 history 추적용 key로만 사용.
- ROAD 데이터 사용 없음.
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
OUT_DIR = os.path.join(DATA_DIR, "features_v4_parts")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    "DoS": "DoS_dataset.csv",
    "Fuzzy": "Fuzzy_dataset.csv",
    "gear": "gear_dataset.csv",
    "RPM": "RPM_dataset.csv",
}

TRAIN_FRAC = 0.7
BYTE_WINDOW = 16

V2_STATS_PATH = os.path.join(
    MODEL_DIR,
    "idagnostic_stats.pkl"
)

if not os.path.exists(V2_STATS_PATH):
    raise FileNotFoundError(
        "models/idagnostic_stats.pkl 없음"
    )

with open(V2_STATS_PATH, "rb") as f:
    STATS = pickle.load(f)

WINDOW = int(STATS["window"])

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


FEATS = [
    # V2
    "freq_in_window",
    "unique_ids_in_window",
    "entropy",
    "mean_byte",
    "global_delta_zscore",
    "global_value_zscore",

    # V4
    "max_byte_abs_delta",
    "rolling_byte_std_max",
    "rolling_byte_std_mean",
    "rolling_byte_std_range",
]


def payload_entropy(b):
    _, c = np.unique(
        b,
        return_counts=True
    )

    p = c / c.sum()

    return float(
        -np.sum(
            p * np.log2(p + 1e-12)
        )
    )


def parse_file(path):
    ts_list = []
    id_list = []
    data_list = []
    label_list = []

    with open(
        path,
        "r",
        newline=""
    ) as f:

        for row in csv.reader(f):

            if len(row) < 4:
                continue

            try:
                ts = float(row[0])
                cid = int(row[1], 16)
                dlc = int(row[2])

                raw = row[
                    3:3 + dlc
                ]

                data = [
                    int(x, 16)
                    for x in raw
                ]

            except (
                ValueError,
                IndexError
            ):
                continue

            data = (
                data + [0] * 8
            )[:8]

            flag = row[-1].strip()

            label = (
                "R"
                if flag == "R"
                else "__ATTACK__"
            )

            ts_list.append(ts)
            id_list.append(cid)
            data_list.append(data)
            label_list.append(label)

    return (
        np.asarray(
            ts_list,
            dtype=np.float64
        ),
        np.asarray(
            id_list,
            dtype=np.int64
        ),
        np.asarray(
            data_list,
            dtype=np.float64
        ),
        np.asarray(
            label_list,
            dtype=object
        ),
    )


def build_features(
    ts,
    ids,
    data
):
    n = len(ids)

    X = np.zeros(
        (n, len(FEATS)),
        dtype=np.float32
    )

    id_window = deque(
        maxlen=WINDOW
    )

    last_payload = {}

    # 동일 ID별 최근 payload history
    payload_hist = defaultdict(
        lambda: deque(
            maxlen=BYTE_WINDOW
        )
    )

    for i in range(n):

        cid = int(ids[i])
        pl = data[i]

        # ----------------------
        # V2
        # ----------------------
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

        ent = payload_entropy(pl)

        mean_byte = float(
            pl.mean()
        )

        if cid in last_payload:

            byte_delta = np.abs(
                pl - last_payload[cid]
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

        global_delta_z = (
            total_delta - G_DELTA_MEAN
        ) / (
            G_DELTA_STD + 1e-12
        )

        global_value_z = float(
            np.max(
                np.abs(
                    (
                        pl - BYTE_MEAN
                    )
                    / BYTE_STD
                )
            )
        )

        # ----------------------
        # V4 byte temporal
        # ----------------------
        hist = payload_hist[cid]

        hist.append(
            pl.copy()
        )

        if len(hist) >= 2:

            H = np.stack(
                list(hist),
                axis=0
            )

            # byte별 최근 temporal std
            byte_stds = np.std(
                H,
                axis=0
            )

            rolling_std_max = float(
                byte_stds.max()
            )

            rolling_std_mean = float(
                byte_stds.mean()
            )

            rolling_std_range = float(
                byte_stds.max()
                - byte_stds.min()
            )

        else:

            rolling_std_max = 0.0
            rolling_std_mean = 0.0
            rolling_std_range = 0.0

        X[i] = [
            freq,
            unique_ids,
            ent,
            mean_byte,
            global_delta_z,
            global_value_z,

            max_byte_abs_delta,
            rolling_std_max,
            rolling_std_mean,
            rolling_std_range,
        ]

        last_payload[cid] = (
            pl.copy()
        )

    return X


def main():

    t_all = time.time()

    for attack_name, fname in FILES.items():

        path = os.path.join(
            DATA_DIR,
            fname
        )

        print(
            f"\n[{attack_name}] "
            f"{fname}"
        )

        t0 = time.time()

        ts, ids, data, labels = (
            parse_file(path)
        )

        labels = np.where(
            labels == "__ATTACK__",
            attack_name,
            "R"
        )

        n = len(ids)

        split = int(
            n * TRAIN_FRAC
        )

        print(
            f"rows={n:,}, "
            f"split={split:,}"
        )

        X = build_features(
            ts,
            ids,
            data
        )

        df = pd.DataFrame(
            X,
            columns=FEATS
        )

        df["Label"] = labels

        is_test = np.zeros(
            n,
            dtype=bool
        )

        is_test[split:] = True

        df[
            "is_test_region"
        ] = is_test

        out = os.path.join(
            OUT_DIR,
            f"{attack_name}"
            "_features_v4.parquet"
        )

        df.to_parquet(
            out,
            index=False
        )

        print(
            f"saved: {out}"
        )

        print(
            f"time: "
            f"{time.time()-t0:.1f}s"
        )

    config = {
        "feature_names": FEATS,
        "window": WINDOW,
        "byte_window": BYTE_WINDOW,
        "description":
            "V2 + byte-level temporal summary",
    }

    with open(
        os.path.join(
            MODEL_DIR,
            "v4_feature_config.pkl"
        ),
        "wb"
    ) as f:
        pickle.dump(
            config,
            f
        )

    print("\n" + "=" * 70)
    print(
        f"V4 feature 생성 완료 "
        f"({len(FEATS)} features)"
    )
    print(
        f"총 시간 "
        f"{time.time()-t_all:.1f}s"
    )


if __name__ == "__main__":
    main()
