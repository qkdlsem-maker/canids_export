"""
features_v3: delta_zscore(변화량) + value_zscore(절대값 z-score, ID별 byte 통계 기반) 추가
value_zscore가 핵심: gear/RPM 스푸핑은 '변화량'보다 '절대값 자체'가 물리적으로 말이 안 되는 경우가 많음
(예: RPM=0인데 갑자기 매우 높은 고정값으로 유지) -> delta만으론 못 잡던 케이스 보완 목적
"""
import pandas as pd, numpy as np, time

import os
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Car_Hacking_5pct.csv")
WINDOW = 20

def payload_entropy(b):
    v, c = np.unique(b, return_counts=True)
    p = c / c.sum()
    return -np.sum(p * np.log2(p + 1e-12))

def build_features_v3(df, window=WINDOW, train_frac=0.7):
    n = len(df)
    ids = df["CAN ID"].values
    data_cols = [f"DATA[{i}]" for i in range(8)]
    payloads = df[data_cols].values
    normal_mask = (df["Label"] == "R").values

    # ---- 중요: leakage 방지 (item7) ----
    # 주의: 이 5% subset은 실제로는 DoS/Fuzzy/RPM/gear 각 공격의 원본 캡처 파일이
    # '통째로' 이어붙여진 구조 (전역 위치로 자르면 gear 공격 전체가 test에만 몰림).
    # -> Label(공격유형)별로 "그 유형 내에서" 앞 70%/뒤 30%로 나눠 각 공격이
    #    train/test 양쪽에 다 존재하게 함 (시간순서는 유지, leakage는 방지).
    is_test_region = np.zeros(n, dtype=bool)
    for lbl in df["Label"].unique():
        idxs = np.where(df["Label"].values == lbl)[0]
        cut = int(len(idxs) * train_frac)
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

    id_delta_stats = {c: (np.mean(v), np.std(v) + 1e-3) for c, v in per_id_delta.items() if len(v) >= 5}
    id_byte_stats = {}
    for c, arr in per_id_bytes.items():
        if len(arr) >= 5:
            a = np.array(arr)
            id_byte_stats[c] = (a.mean(axis=0), a.std(axis=0) + 1e-3)
    g_delta_mean, g_delta_std = 2.0, 5.0

    feats = np.zeros((n, 9), dtype=np.float32)
    id_window, last_payload_per_id = [], {}
    t0 = time.time()
    for i in range(n):
        cid, pl = ids[i], payloads[i]
        id_window.append(cid)
        if len(id_window) > window:
            id_window.pop(0)

        freq_in_window = id_window.count(cid) / len(id_window)
        unique_ids_in_window = len(set(id_window))
        is_unknown_id = 0.0 if cid in known_ids else 1.0
        entropy = payload_entropy(pl)
        mean_byte = pl.mean()

        if cid in last_payload_per_id:
            delta = np.abs(pl.astype(int) - last_payload_per_id[cid].astype(int)).sum()
        else:
            delta = 0.0
        last_payload_per_id[cid] = pl
        mu, sigma = id_delta_stats.get(cid, (g_delta_mean, g_delta_std))
        delta_zscore = (delta - mu) / sigma

        if cid in id_byte_stats:
            bmu, bstd = id_byte_stats[cid]
            value_zscore = np.max(np.abs((pl.astype(float) - bmu) / bstd))
        else:
            value_zscore = 0.0  # 통계 없는 ID는 중립

        norm_id = cid / 2048.0
        feats[i] = [freq_in_window, unique_ids_in_window, is_unknown_id, entropy,
                    mean_byte, delta_zscore, value_zscore, norm_id, float(is_test_region[i])]

    print(f"[v3] {n:,} rows, {time.time()-t0:.1f}s")
    cols = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy",
            "mean_byte", "delta_zscore", "value_zscore", "norm_id", "is_test_region"]
    fd = pd.DataFrame(feats, columns=cols)
    fd["label_bin"] = df["label_bin"].values
    fd["Label"] = df["Label"].values
    return fd

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df["label_bin"] = (df["Label"] != "R").astype(int)
    fd = build_features_v3(df)
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "features_v3.parquet")
    fd.to_parquet(out_path)
    print("saved", fd.shape, "->", out_path)
