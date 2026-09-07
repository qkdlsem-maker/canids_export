
"""
Independent injector-side ground-truth logger for ICSim experiments.

Ground truth is created ONLY from attack injection events.
No IDS feature, model prediction, anomaly score, or threshold is used.
"""

import csv
import os
import time
import threading


class AttackGTLogger:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.seq = 0

        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        self.fp = open(
            path,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.writer(self.fp)

        self.writer.writerow([
            "seq",
            "timestamp_ns",
            "timestamp",
            "can_id",
            "can_id_hex",
            "dlc",
            "payload_hex",
            "attack_type",
        ])

        self.fp.flush()

    def log(self, can_id, payload, attack_type):
        """
        Call immediately before/after the actual CAN send().

        can_id:
            integer arbitration ID

        payload:
            bytes / bytearray / iterable[int]

        attack_type:
            DoS / Fuzzy / spoof / etc.
        """

        payload = bytes(payload)

        ts_ns = time.time_ns()
        ts = ts_ns / 1_000_000_000.0

        with self.lock:
            self.seq += 1

            self.writer.writerow([
                self.seq,
                ts_ns,
                f"{ts:.9f}",
                int(can_id),
                f"0x{int(can_id):03X}",
                len(payload),
                payload.hex().upper(),
                str(attack_type),
            ])

            # 공격 중 프로세스가 비정상 종료돼도
            # 가능한 한 GT를 보존한다.
            self.fp.flush()

        return self.seq, ts_ns

    def close(self):
        with self.lock:
            if not self.fp.closed:
                self.fp.flush()
                self.fp.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):
        self.close()
