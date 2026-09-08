from collections import deque
import pickle
from pathlib import Path

import numpy as np


FEATURE_NAMES = [
    "freq_in_window",
    "unique_ids_in_window",
    "entropy",
    "mean_byte",
    "global_delta_zscore",
    "global_value_zscore",
]


class V2StreamFeatureExtractor:
    """
    Streaming implementation of the frozen final V2 feature semantics.

    Semantics follow src/13_build_features_v2.py:
      - 20-frame CAN-ID window
      - payload zero-padded/truncated to 8 bytes
      - per-ID previous payload only for delta computation
      - global train-normal statistics from idagnostic_stats.pkl
      - six ID-agnostic output features
    """

    def __init__(self, stats_path):
        stats_path = Path(stats_path)

        with stats_path.open("rb") as f:
            stats = pickle.load(f)

        self.window_size = int(stats["window"])
        self.global_delta_mean = float(stats["global_delta_mean"])
        self.global_delta_std = float(stats["global_delta_std"])

        self.byte_pos_mean = np.asarray(
            stats["byte_pos_mean"], dtype=np.float64
        )
        self.byte_pos_std = np.asarray(
            stats["byte_pos_std"], dtype=np.float64
        )

        self.reset()

    def reset(self):
        self.id_window = deque(maxlen=self.window_size)
        self.last_payload_per_id = {}

    @staticmethod
    def normalize_payload(payload):
        if isinstance(payload, str):
            hexdata = "".join(payload.split())

            if len(hexdata) % 2 != 0:
                raise ValueError("payload hex string must contain full bytes")

            values = [
                int(hexdata[i:i + 2], 16)
                for i in range(0, min(len(hexdata), 16), 2)
            ]
        else:
            values = [int(x) for x in payload[:8]]

        if any(x < 0 or x > 255 for x in values):
            raise ValueError("CAN payload byte outside 0..255")

        values = (values + [0] * 8)[:8]

        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def payload_entropy(payload):
        _, counts = np.unique(payload, return_counts=True)
        p = counts / counts.sum()
        return float(-np.sum(p * np.log2(p + 1e-12)))

    def process(self, can_id, payload):
        cid = int(can_id)

        if cid < 0 or cid > 0x7FF:
            raise ValueError(
                "final V2 demo supports standard 11-bit CAN IDs only"
            )

        pl = self.normalize_payload(payload)

        # Same ordering as src/13_build_features_v2.py:
        # append current ID first, then calculate window features.
        self.id_window.append(cid)

        freq_in_window = (
            sum(1 for x in self.id_window if x == cid)
            / len(self.id_window)
        )

        unique_ids_in_window = len(set(self.id_window))

        entropy = self.payload_entropy(pl)
        mean_byte = float(pl.mean())

        previous = self.last_payload_per_id.get(cid)

        if previous is None:
            delta = 0.0
        else:
            delta = float(np.abs(pl - previous).sum())

        self.last_payload_per_id[cid] = pl.copy()

        global_delta_zscore = (
            delta - self.global_delta_mean
        ) / self.global_delta_std

        global_value_zscore = float(
            np.max(
                np.abs(
                    (pl - self.byte_pos_mean)
                    / self.byte_pos_std
                )
            )
        )

        values = np.asarray(
            [
                freq_in_window,
                unique_ids_in_window,
                entropy,
                mean_byte,
                global_delta_zscore,
                global_value_zscore,
            ],
            dtype=np.float64,
        )

        return {
            "can_id": cid,
            "can_id_hex": f"{cid:03X}",
            "payload": [int(x) for x in pl],
            "payload_hex": "".join(f"{int(x):02X}" for x in pl),
            "features": values,
            "feature_dict": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, values)
            },
        }
