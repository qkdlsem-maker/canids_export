"""
기존 8개 피처 중 is_unknown_id, norm_id, delta_zscore, value_zscore는
학습 차량의 CAN ID를 그대로 키(key)로 쓰기 때문에 차종이 바뀌면 무용지물이 된다
(ROAD 교차검증에서 확인된 문제). 이를 해결하기 위해 CAN ID 자체에 의존하지 않는
6개 ID-agnostic 피처로 재설계한다.

[V1 (기존, ID-dependent)]          [V2 (신규, ID-agnostic)]
freq_in_window                 ->  freq_in_window (유지)
unique_ids_in_window           ->  unique_ids_in_window (유지)
is_unknown_id (ID 키 사용)      ->  제거
entropy                        ->  entropy (유지)
mean_byte                      ->  mean_byte (유지)
delta_zscore (ID별 통계)        ->  global_delta_zscore (전역 통계로 대체)
value_zscore (ID별 통계)        ->  global_value_zscore (바이트 위치별 전역 통계로 대체)
norm_id (ID 값 그 자체)         ->  제거

사용법: python3 13_build_features_v2.py
"""
import csv, os, time, pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data_full")
OUT_PARQUET = os.path.join(HERE, "..", "data_full", "features_v2_idagnostic.parquet")
OUT_STATS = os.path.join(HERE, "..", "models", "idagnostic_stats.pkl")
WINDOW = 20
TRAIN_FRAC = 0.7

FILES = {
    "DoS": "DoS_dataset.csv",
    "Fuzzy": "Fuzzy_dataset.csv",
    "gear": "gear_dataset.csv",
    "RPM": "RPM_dataset.csv",
}


def parse_file(path, attack_name):
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

    # ---------- 1단계: 파싱 + 시간순 split (기존과 동일) ----------
    parsed = {}
    for name, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        print(f"[파싱] {fname} ...")
        t0 = time.time()
        ts, ids, data, labels = parse_file(path, name)
        n = len(ts)
        is_test = np.zeros(n, dtype=bool)
        is_test[int(n * TRAIN_FRAC):] = True
        parsed[name] = dict(ids=ids, data=data, labels=labels, is_test=is_test)
        print(f"  -> {n:,}행, {time.time()-t0:.1f}s")

    # ---------- 2단계: 전역(global) 통계 산출 (ID 키 없이, train 구간 정상만) ----------
    print("[전역 통계 산출] ID에 의존하지 않는 전역 delta/byte-position 통계")
    all_train_deltas = []
    byte_pos_values = [[] for _ in range(8)]  # 바이트 위치(0~7)별 전역 값 모음
    for name, d in parsed.items():
        mask = (~d["is_test"]) & (d["labels"] == "R")
        idxs = np.where(mask)[0]
        last_payload = {}
        for i in idxs:
            cid, pl = d["ids"][i], d["data"][i]
            if cid in last_payload:
                delta = np.abs(pl.astype(int) - last_payload[cid].astype(int)).sum()
                all_train_deltas.append(delta)
            last_payload[cid] = pl
            for bi in range(8):
                byte_pos_values[bi].append(pl[bi])

    global_delta_mean = float(np.mean(all_train_deltas))
    global_delta_std = float(np.std(all_train_deltas) + 1e-3)
    byte_pos_mean = np.array([np.mean(v) for v in byte_pos_values])
    byte_pos_std = np.array([np.std(v) + 1e-3 for v in byte_pos_values])
    print(f"  global_delta: mean={global_delta_mean:.2f} std={global_delta_std:.2f}")
    print(f"  byte_pos_mean={byte_pos_mean.round(1)}")

    stats = {"global_delta_mean": global_delta_mean, "global_delta_std": global_delta_std,
              "byte_pos_mean": byte_pos_mean, "byte_pos_std": byte_pos_std, "window": WINDOW}
    with open(OUT_STATS, "wb") as f:
        pickle.dump(stats, f)
    print(f"[저장] {OUT_STATS}")

    # ---------- 3단계: ID-agnostic 6개 피처 계산 ----------
    FEATS_COLS = ["freq_in_window", "unique_ids_in_window", "entropy", "mean_byte",
                  "global_delta_zscore", "global_value_zscore"]
    all_feats = []
    for name, d in parsed.items():
        print(f"[피처공학 V2] {name} ...")
        t0 = time.time()
        n = len(d["ids"])
        feats = np.zeros((n, 6), dtype=np.float32)
        id_window, last_payload_per_id = [], {}
        ids_arr, data_arr = d["ids"], d["data"]
        for i in range(n):
            cid = ids_arr[i]; pl = data_arr[i]
            id_window.append(cid)
            if len(id_window) > WINDOW:
                id_window.pop(0)

            freq_in_window = id_window.count(cid) / len(id_window)
            unique_ids_in_window = len(set(id_window))
            entropy = payload_entropy(pl)
            mean_byte = pl.mean()

            if cid in last_payload_per_id:
                delta = np.abs(pl.astype(int) - last_payload_per_id[cid].astype(int)).sum()
            else:
                delta = 0.0
            last_payload_per_id[cid] = pl
            global_delta_zscore = (delta - global_delta_mean) / global_delta_std

            global_value_zscore = np.max(np.abs((pl.astype(float) - byte_pos_mean) / byte_pos_std))

            feats[i] = [freq_in_window, unique_ids_in_window, entropy, mean_byte,
                        global_delta_zscore, global_value_zscore]

        fd = pd.DataFrame(feats, columns=FEATS_COLS)
        fd["Label"] = d["labels"]
        fd["is_test_region"] = d["is_test"]
        all_feats.append(fd)
        print(f"  -> {time.time()-t0:.1f}s")

    full = pd.concat(all_feats, ignore_index=True)
    full.to_parquet(OUT_PARQUET)
    print(f"\n[완료] {OUT_PARQUET} 저장, shape={full.shape}, 총 소요시간 {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
