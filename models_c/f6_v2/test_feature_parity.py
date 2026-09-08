import ctypes
import math
import numpy as np
from pathlib import Path

LIB = Path(__file__).with_name("libfeature_v2_stream.so")
lib = ctypes.CDLL(str(LIB))

lib.feature_v2_reset.argtypes = []
lib.feature_v2_reset.restype = None

lib.feature_v2_process.argtypes = [
    ctypes.c_uint16,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_double),
]
lib.feature_v2_process.restype = ctypes.c_int

WINDOW = 20

GLOBAL_DELTA_MEAN = 38.177002161911595
GLOBAL_DELTA_STD = 73.15781974453239

BYTE_POS_MEAN = np.array([
    59.83983448956544,
    43.563562607886,
    36.51474799623342,
    63.44021874034496,
    49.585436009696956,
    64.42127572524171,
    25.779640930089347,
    45.624118274161454
], dtype=np.float64)

BYTE_POS_STD = np.array([
    93.74467133819213,
    53.9960477801128,
    57.805561856829684,
    93.09714231996038,
    75.4460376368916,
    78.58607603834241,
    56.210402274924256,
    69.76311117028881
], dtype=np.float64)


def entropy_python(b):
    b = np.asarray(b, dtype=np.uint8)
    _, c = np.unique(b, return_counts=True)
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


class PythonReference:
    def __init__(self):
        self.id_window = []
        self.last_payload_per_id = {}

    def process(self, cid, payload):
        pl = np.asarray(payload, dtype=np.uint8)

        self.id_window.append(cid)
        if len(self.id_window) > WINDOW:
            self.id_window.pop(0)

        freq = self.id_window.count(cid) / len(self.id_window)
        unique_ids = len(set(self.id_window))

        entropy = entropy_python(pl)
        mean_byte = float(pl.mean())

        if cid in self.last_payload_per_id:
            prev = self.last_payload_per_id[cid]
            delta = float(
                np.abs(
                    pl.astype(np.int16) -
                    prev.astype(np.int16)
                ).sum()
            )
        else:
            delta = 0.0

        self.last_payload_per_id[cid] = pl.copy()

        delta_z = (
            delta - GLOBAL_DELTA_MEAN
        ) / GLOBAL_DELTA_STD

        value_z = float(
            np.max(
                np.abs(
                    (pl.astype(np.float64) - BYTE_POS_MEAN)
                    / BYTE_POS_STD
                )
            )
        )

        return np.array([
            freq,
            float(unique_ids),
            entropy,
            mean_byte,
            delta_z,
            value_z,
        ], dtype=np.float64)


def c_process(cid, payload):
    payload = [int(x) for x in payload]

    p = (ctypes.c_uint8 * 8)(*payload)
    out = (ctypes.c_double * 6)()

    rc = lib.feature_v2_process(cid, p, out)
    if rc != 0:
        raise RuntimeError(
            f"feature_v2_process failed: cid={cid:#x}, rc={rc}"
        )

    return np.array(list(out), dtype=np.float64)


# Deterministic sequence deliberately covers:
# - repeated ID
# - changing payload
# - all-equal payload
# - all-distinct payload
# - zero payload
# - max-byte payload
# - >20 frames to exercise circular-window replacement
# - boundary CAN IDs 0x000 and 0x7FF
sequence = [
    (0x123, [0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88]),
    (0x123, [0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88]),
    (0x123, [0x12,0x22,0x30,0x44,0x50,0x66,0x70,0x88]),
    (0x000, [0,0,0,0,0,0,0,0]),
    (0x7FF, [255,255,255,255,255,255,255,255]),
    (0x100, [0,1,2,3,4,5,6,7]),
]

for i in range(40):
    cid = (i * 97 + 0x55) & 0x7FF
    payload = [
        (i * 17 + j * 29 + (i ^ j)) & 0xFF
        for j in range(8)
    ]
    sequence.append((cid, payload))

# Revisit IDs after the window has wrapped.
sequence += [
    (0x123, [1,2,3,4,5,6,7,8]),
    (0x000, [8,7,6,5,4,3,2,1]),
    (0x7FF, [0,255,0,255,0,255,0,255]),
]

ref = PythonReference()
lib.feature_v2_reset()

names = [
    "freq_in_window",
    "unique_ids_in_window",
    "entropy",
    "mean_byte",
    "global_delta_zscore",
    "global_value_zscore",
]

max_abs = np.zeros(6, dtype=np.float64)
worst_frame = np.zeros(6, dtype=np.int64)

all_ok = True

for frame_no, (cid, payload) in enumerate(sequence, start=1):
    py = ref.process(cid, payload)
    cc = c_process(cid, payload)

    diff = np.abs(py - cc)

    for k in range(6):
        if diff[k] > max_abs[k]:
            max_abs[k] = diff[k]
            worst_frame[k] = frame_no

    if not np.allclose(py, cc, rtol=1e-12, atol=1e-12):
        all_ok = False
        print(f"\nMISMATCH frame={frame_no} id=0x{cid:03X}")
        print("Python:", repr(py))
        print("C     :", repr(cc))
        print("AbsDiff:", repr(diff))

print()
print("=== FEATURE PARITY SUMMARY ===")
print("frames =", len(sequence))

for k, name in enumerate(names):
    print(
        f"{name:24s} "
        f"max_abs_diff={max_abs[k]:.17g} "
        f"worst_frame={worst_frame[k]}"
    )

print()
print("ALLCLOSE_1E-12 =", all_ok)

if not all_ok:
    raise SystemExit(1)

print("PASS")
