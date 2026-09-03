"""
Car-Hacking(학습 데이터)에서 얻은 통계량(known_ids, per-ID delta/byte 통계)을
pickle로 저장. -> 09_road_crossdataset_test.py에서 재사용해서
"완전히 다른 차량 + 완전히 새로운 공격유형(ROAD)"에 대한 진짜 zero-day 테스트를 함.
"""
import pandas as pd, numpy as np, pickle, os

here = os.path.dirname(__file__)
DATA_PATH = os.path.join(here, "..", "data", "Car_Hacking_5pct.csv")

df = pd.read_csv(DATA_PATH)
n = len(df)
ids = df["CAN ID"].values
data_cols = [f"DATA[{i}]" for i in range(8)]
payloads = df[data_cols].values
normal_mask = (df["Label"] == "R").values

is_test_region = np.zeros(n, dtype=bool)
for lbl in df["Label"].unique():
    idxs = np.where(df["Label"].values == lbl)[0]
    cut = int(len(idxs) * 0.7)
    is_test_region[idxs[cut:]] = True
train_normal_idx = np.where(normal_mask & (~is_test_region))[0]

known_ids = set(np.unique(ids[train_normal_idx]))
per_id_delta, per_id_bytes = {}, {}
last_payload = {}
for i in train_normal_idx:
    cid, pl = ids[i], payloads[i]
    if cid in last_payload:
        d = np.abs(pl.astype(int) - last_payload[cid].astype(int)).sum()
        per_id_delta.setdefault(cid, []).append(d)
    last_payload[cid] = pl
    per_id_bytes.setdefault(cid, []).append(pl.astype(float))

id_delta_stats = {c: (float(np.mean(v)), float(np.std(v) + 1e-3)) for c, v in per_id_delta.items() if len(v) >= 5}
id_byte_stats = {}
for c, arr in per_id_bytes.items():
    if len(arr) >= 5:
        a = np.array(arr)
        id_byte_stats[c] = (a.mean(axis=0), a.std(axis=0) + 1e-3)

stats = {
    "known_ids": known_ids,
    "id_delta_stats": id_delta_stats,
    "id_byte_stats": id_byte_stats,
    "global_delta_mean": 2.0,
    "global_delta_std": 5.0,
    "window": 20,
}
out_path = os.path.join(here, "..", "models", "car_hacking_stats.pkl")
with open(out_path, "wb") as f:
    pickle.dump(stats, f)
print("저장됨:", out_path, f"(known_ids: {len(known_ids)}개)")
