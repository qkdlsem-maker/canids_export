"""
HCRL Car-Hacking 원본 전체 데이터셋(DoS/Fuzzy/gear/RPM_dataset.csv)을 파싱하고
기존과 동일한 8종 피처를 계산. 5% 서브셋과 달리 실제 타임스탬프가 있어
진짜 시간순(chronological) train/test split이 가능함 (재현성/신뢰도 강화).

사용법: python3 10_build_features_full.py
전제: canids_export/data_full/ 안에 DoS_dataset.csv, Fuzzy_dataset.csv,
      gear_dataset.csv, RPM_dataset.csv 가 있어야 함
"""
import csv, os, time, pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data_full")
OUT_PARQUET = os.path.join(HERE, "..", "data_full", "features_full.parquet")
OUT_STATS = os.path.join(HERE, "..", "models", "car_hacking_full_stats.pkl")
WINDOW = 20
TRAIN_FRAC = 0.7

FILES = {
    "DoS": "DoS_dataset.csv",
    "Fuzzy": "Fuzzy_dataset.csv",
    "gear": "gear_dataset.csv",
    "RPM": "RPM_dataset.csv",
}


def parse_file(path, attack_name):
    """(timestamp, CAN_ID(int), data[8](int), Label) 리스트로 파싱.
       Label: 'R'이면 정상, 'T'(injected)면 해당 attack_name."""
    ts_list, id_list, data_list, label_list = [], [], [], []
    with open(path, "r", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            try:
                ts = float(row[0])
                cid = int(row[1], 16)
                dlc = int(row[2])
                data_fields = row[3:3 + dlc]
                data = [int(x, 16) for x in data_fields]
            except ValueError:
                continue
            data = (data + [0] * 8)[:8]
            flag = row[-1].strip()
            label = "R" if flag == "R" else attack_name
            ts_list.append(ts); id_list.append(cid); data_list.append(data); label_list.append(label)
    return (np.array(ts_list), np.array(id_list), np.array(data_list, dtype=np.int64), np.array(label_list))


def payload_entropy(b):
    v, c = np.unique(b, return_counts=True)
    p = c / c.sum()
    return -np.sum(p * np.log2(p + 1e-12))


def main():
    t_start = time.time()

    # ---------- 1단계: 4개 파일 파싱 + 파일별 시간순 70/30 split ----------
    parsed = {}
    for name, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        print(f"[파싱] {fname} ...")
        t0 = time.time()
        ts, ids, data, labels = parse_file(path, name)
        n = len(ts)
        is_test = np.zeros(n, dtype=bool)
        cut = int(n * TRAIN_FRAC)
        is_test[cut:] = True  # 실제 타임스탬프로 이미 정렬되어 있음 -> 진짜 시간순 split
        parsed[name] = dict(ts=ts, ids=ids, data=data, labels=labels, is_test=is_test)
        print(f"  -> {n:,}행, {time.time()-t0:.1f}s, R={np.sum(labels=='R'):,} {name}={np.sum(labels==name):,}")

    # ---------- 2단계: 학습통계(known_ids, per-ID delta/byte stats) 산출 (train 구간 R만) ----------
    print("[통계 산출] train 구간 정상(R) 데이터 기준")
    known_ids = set()
    per_id_delta, per_id_bytes = {}, {}
    for name, d in parsed.items():
        mask = (~d["is_test"]) & (d["labels"] == "R")
        idxs = np.where(mask)[0]
        known_ids.update(np.unique(d["ids"][idxs]))
        last_payload = {}
        for i in idxs:
            cid, pl = d["ids"][i], d["data"][i]
            if cid in last_payload:
                delta = np.abs(pl.astype(int) - last_payload[cid].astype(int)).sum()
                per_id_delta.setdefault(cid, []).append(delta)
            last_payload[cid] = pl
            per_id_bytes.setdefault(cid, []).append(pl.astype(float))

    id_delta_stats = {c: (float(np.mean(v)), float(np.std(v) + 1e-3)) for c, v in per_id_delta.items() if len(v) >= 5}
    id_byte_stats = {}
    for c, arr in per_id_bytes.items():
        if len(arr) >= 5:
            a = np.array(arr)
            id_byte_stats[c] = (a.mean(axis=0), a.std(axis=0) + 1e-3)
    g_delta_mean, g_delta_std = 2.0, 5.0
    print(f"  known_ids: {len(known_ids)}개")

    stats = {
        "known_ids": known_ids, "id_delta_stats": id_delta_stats, "id_byte_stats": id_byte_stats,
        "global_delta_mean": g_delta_mean, "global_delta_std": g_delta_std, "window": WINDOW,
    }
    with open(OUT_STATS, "wb") as f:
        pickle.dump(stats, f)
    print(f"[저장] {OUT_STATS}")

    # ---------- 3단계: 파일별로 순서를 유지하며 8종 피처 계산 ----------
    all_feats = []
    FEATS_COLS = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy",
                  "mean_byte", "delta_zscore", "value_zscore", "norm_id"]

    for name, d in parsed.items():
        print(f"[피처공학] {name} ...")
        t0 = time.time()
        n = len(d["ids"])
        feats = np.zeros((n, 8), dtype=np.float32)
        id_window, last_payload_per_id = [], {}
        ids_arr, data_arr = d["ids"], d["data"]
        for i in range(n):
            cid = ids_arr[i]; pl = data_arr[i]
            id_window.append(cid)
            if len(id_window) > WINDOW:
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
                value_zscore = 0.0

            norm_id = cid / 2048.0
            feats[i] = [freq_in_window, unique_ids_in_window, is_unknown_id, entropy,
                        mean_byte, delta_zscore, value_zscore, norm_id]

        fd = pd.DataFrame(feats, columns=FEATS_COLS)
        fd["Label"] = d["labels"]
        fd["is_test_region"] = d["is_test"]
        fd["source_file"] = name
        all_feats.append(fd)
        print(f"  -> {time.time()-t0:.1f}s")

    full = pd.concat(all_feats, ignore_index=True)
    full.to_parquet(OUT_PARQUET)
    print(f"\n[완료] {OUT_PARQUET} 저장, shape={full.shape}, 총 소요시간 {time.time()-t_start:.1f}s")
    print(full["Label"].value_counts())


if __name__ == "__main__":
    main()
