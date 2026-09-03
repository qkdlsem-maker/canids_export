"""
Car-Hacking으로 학습한 Mahalanobis 탐지기를, 완전히 다른 차량(ROAD)의
완전히 새로운 공격유형(correlated_signal, max_speedometer, max_engine_coolant_temp,
reverse_light_on/off, accelerator)에 그대로 적용해서 진짜 zero-day 일반화 성능을 검증.

사용법:
    python3 09_road_crossdataset_test.py /path/to/road/data/attacks /path/to/road/data/ambient

- attacks 폴더: *.log (마스커레이드 없는 원본 .log만 사용, _masquerade 제외)
- ambient 폴더: 정상 오탐율(FPR) 측정용 (원하면 생략 가능, 없으면 FPR은 스킵)
"""
import sys, os, re, glob, pickle
import numpy as np
import pandas as pd

here = os.path.dirname(__file__)

with open(os.path.join(here, "..", "models", "car_hacking_stats.pkl"), "rb") as f:
    STATS = pickle.load(f)
npz = np.load(os.path.join(here, "..", "models", "maha_detector.npz"))
MEAN_, INV_COV, THR = npz["mean"], npz["inv_cov"], float(npz["thr"])
WINDOW = STATS["window"]

LINE_RE = re.compile(r"\(([\d.]+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)")

def parse_log(path, max_lines=200000):
    """candump 형식 .log 파싱 -> (CAN ID(int), 8바이트 payload(int array)) 리스트"""
    rows = []
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            m = LINE_RE.search(line)
            if not m:
                continue
            cid = int(m.group(2), 16)
            hexdata = m.group(3)
            byts = [int(hexdata[j:j+2], 16) for j in range(0, min(len(hexdata), 16), 2)]
            byts = (byts + [0] * 8)[:8]
            rows.append((cid, byts))
    return rows

def payload_entropy(b):
    v, c = np.unique(b, return_counts=True)
    p = c / c.sum()
    return -np.sum(p * np.log2(p + 1e-12))

def extract_features(rows):
    """저장된 Car-Hacking 통계 기준으로 8개 피처 계산 (학습 때와 완전히 동일한 로직)"""
    known_ids = STATS["known_ids"]
    id_delta_stats = STATS["id_delta_stats"]
    id_byte_stats = STATS["id_byte_stats"]
    g_delta_mean, g_delta_std = STATS["global_delta_mean"], STATS["global_delta_std"]

    feats = []
    id_window, last_payload_per_id = [], {}
    for cid, pl in rows:
        pl = np.array(pl, dtype=float)
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
        feats.append([freq_in_window, unique_ids_in_window, is_unknown_id, entropy, mean_byte, delta_zscore, value_zscore, norm_id])
    return np.array(feats)

def maha_flag(X):
    diff = X - MEAN_
    dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, INV_COV, diff))
    return dist > THR

def evaluate_file(path):
    rows = parse_log(path)
    if len(rows) < WINDOW + 1:
        return None
    X = extract_features(rows)
    flags = maha_flag(X)
    return flags.mean(), len(rows)

def main():
    attacks_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    ambient_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print("=== ROAD 공격 (Car-Hacking엔 없던 완전히 새로운 공격유형) ===")
    for path in sorted(glob.glob(os.path.join(attacks_dir, "*.log"))):
        name = os.path.basename(path)
        if "_masquerade" in name:
            continue
        result = evaluate_file(path)
        if result is None:
            continue
        recall, n = result
        print(f"[{name}] 탐지율: {recall:.1%}  (메시지 {n:,}개)")

    if ambient_dir:
        print("\n=== ROAD 정상(ambient) 데이터 - 오탐율(FPR) ===")
        for path in sorted(glob.glob(os.path.join(ambient_dir, "*.log")))[:3]:
            name = os.path.basename(path)
            result = evaluate_file(path)
            if result is None:
                continue
            fpr, n = result
            print(f"[{name}] 오탐율: {fpr:.2%}  (메시지 {n:,}개)")

if __name__ == "__main__":
    main()
