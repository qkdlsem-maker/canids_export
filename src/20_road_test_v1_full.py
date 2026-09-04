"""
F3 재검증: V1(ID-dependent)과 V2(ID-agnostic)를 "둘 다 전체 데이터셋(1,657만 행) 기준"으로
학습한 뒤 ROAD에 적용해서 공정하게 비교.
"""
import sys, os, re, glob, pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)

with open(os.path.join(HERE, "..", "models", "car_hacking_full_stats.pkl"), "rb") as f:
    STATS = pickle.load(f)
npz = np.load(os.path.join(HERE, "..", "models", "maha_detector_full.npz"))
MEAN_, INV_COV, THR = npz["mean"], npz["inv_cov"], float(npz["thr"])
WINDOW = STATS["window"]
known_ids = STATS["known_ids"]
id_delta_stats = STATS["id_delta_stats"]
id_byte_stats = STATS["id_byte_stats"]
g_delta_mean, g_delta_std = STATS["global_delta_mean"], STATS["global_delta_std"]

print(f"[V1-전체데이터셋] known_ids: {len(known_ids)}개, threshold={THR:.3f}")

LINE_RE = re.compile(r"\(([\d.]+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)")

def parse_log(path, max_lines=200000):
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

def extract_features_v1(rows):
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
        feats.append([freq_in_window, unique_ids_in_window, is_unknown_id, entropy,
                       mean_byte, delta_zscore, value_zscore, norm_id])
    return np.array(feats)

def maha_flag(X):
    diff = X - MEAN_
    dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, INV_COV, diff))
    return dist > THR

def evaluate_file(path):
    rows = parse_log(path)
    if len(rows) < WINDOW + 1:
        return None
    X = extract_features_v1(rows)
    flags = maha_flag(X)
    return flags.mean(), len(rows)

def main():
    road_dir = os.path.join(HERE, "..", "road_data")
    attacks_dir = os.path.join(road_dir, "attacks")
    ambient_dir = os.path.join(road_dir, "ambient")

    print("\n=== [V1-전체데이터셋] ROAD 공격 ===")
    attack_results = []
    for path in sorted(glob.glob(os.path.join(attacks_dir, "*.log"))):
        name = os.path.basename(path)
        if "_masquerade" in name:
            continue
        result = evaluate_file(path)
        if result is None:
            continue
        recall, n = result
        attack_results.append(recall)
        print(f"[{name}] 탐지율: {recall:.1%} ({n:,}건)")

    print("\n=== [V1-전체데이터셋] ROAD 정상(ambient) FPR ===")
    fpr_results = []
    for path in sorted(glob.glob(os.path.join(ambient_dir, "*.log")))[:5]:
        name = os.path.basename(path)
        result = evaluate_file(path)
        if result is None:
            continue
        fpr, n = result
        fpr_results.append(fpr)
        print(f"[{name}] 오탐율: {fpr:.2%} ({n:,}건)")

    print("\n" + "=" * 60)
    if fpr_results:
        print(f"[V1-전체데이터셋] 평균 FPR={np.mean(fpr_results):.2%}, 평균 탐지율={np.mean(attack_results):.1%}")

    out_path = os.path.join(HERE, "..", "results", "f3_v1_full_road_test.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== V1(ID-dependent), 전체 데이터셋 기준, ROAD 교차검증 ===\n")
        f.write(f"평균 FPR: {np.mean(fpr_results):.4%}\n")
        f.write(f"평균 탐지율: {np.mean(attack_results):.4%}\n")
    print(f"\n[완료] {out_path} 저장됨")

if __name__ == "__main__":
    main()
