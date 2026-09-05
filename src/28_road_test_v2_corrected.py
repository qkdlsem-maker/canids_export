"""
14_train_v2_and_road_test.py의 ROAD 평가(recall = flags.mean() over 전체 파일)는
ICSim에서 발견한 것과 같은 '시간구간 희석' 문제가 있음. capture_metadata.json의
injection_id + injection_interval로 실제 공격 메시지만 정밀하게 걸러서 재평가.

전제: models/idagnostic_stats.pkl, models/maha_v2.npz (13/14번 결과물) 존재해야 함
사용법: python3 28_road_test_v2_corrected.py
"""
import os, re, glob, json, pickle
import numpy as np

HERE = os.path.dirname(__file__)
ROAD_ATTACKS = os.path.join(HERE, "..", "road_data", "attacks")

with open(os.path.join(HERE, "..", "models", "idagnostic_stats.pkl"), "rb") as f:
    STATS = pickle.load(f)
WINDOW = STATS["window"]
g_delta_mean, g_delta_std = STATS["global_delta_mean"], STATS["global_delta_std"]
byte_pos_mean, byte_pos_std = STATS["byte_pos_mean"], STATS["byte_pos_std"]

npz = np.load(os.path.join(HERE, "..", "models", "maha_v2.npz"))
MEAN_, INV_COV, THR = npz["mean"], npz["inv_cov"], float(npz["thr"])

with open(os.path.join(ROAD_ATTACKS, "capture_metadata.json")) as f:
    METADATA = json.load(f)

LINE_RE = re.compile(r"\(([\d.]+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)")

def parse_log(path):
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            ts = float(m.group(1))
            cid = int(m.group(2), 16)
            hexdata = m.group(3)
            byts = [int(hexdata[j:j+2], 16) for j in range(0, min(len(hexdata), 16), 2)]
            byts = (byts + [0] * 8)[:8]
            rows.append((ts, cid, byts))
    return rows

def payload_entropy(b):
    v, c = np.unique(b, return_counts=True)
    p = c / c.sum()
    return -np.sum(p * np.log2(p + 1e-12))

def extract_features_v2(rows):
    feats = []
    id_window, last_payload_per_id = [], {}
    for _, cid, pl in rows:
        pl = np.array(pl, dtype=float)
        id_window.append(cid)
        if len(id_window) > WINDOW:
            id_window.pop(0)
        freq_in_window = id_window.count(cid) / len(id_window)
        unique_ids_in_window = len(set(id_window))
        entropy = payload_entropy(pl)
        mean_byte = pl.mean()
        if cid in last_payload_per_id:
            delta = np.abs(pl - last_payload_per_id[cid]).sum()
        else:
            delta = 0.0
        last_payload_per_id[cid] = pl
        global_delta_zscore = (delta - g_delta_mean) / g_delta_std
        global_value_zscore = np.max(np.abs((pl - byte_pos_mean) / byte_pos_std))
        feats.append([freq_in_window, unique_ids_in_window, entropy, mean_byte,
                       global_delta_zscore, global_value_zscore])
    return np.array(feats)

def maha_flag(X):
    diff = X - MEAN_
    dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, INV_COV, diff))
    return dist > THR

print(f"{'파일':45s} | {'기존(전체파일)':>14s} | {'정정(구간+ID 필터)':>18s} | 필터된 메시지 수")
print("-" * 100)

naive_results, fixed_results = [], []
for path in sorted(glob.glob(os.path.join(ROAD_ATTACKS, "*.log"))):
    name = os.path.basename(path)
    if "_masquerade" in name:
        continue
    meta = METADATA.get(name.replace(".log", ""), {})
    injection_id = meta.get("injection_id")
    interval = meta.get("injection_interval")

    rows = parse_log(path)
    if len(rows) < WINDOW + 1:
        continue
    X = extract_features_v2(rows)
    flags = maha_flag(X)
    naive_recall = flags.mean()
    naive_results.append(naive_recall)

    if interval is None:
        print(f"{name:45s} | {naive_recall:>13.1%} | {'메타데이터 없음(행동기반 공격)':>18s} | -")
        continue

    t0 = rows[0][0]
    rel_ts = np.array([r[0] - t0 for r in rows])
    ids = np.array([r[1] for r in rows])
    in_interval = (rel_ts >= interval[0]) & (rel_ts <= interval[1])

    if injection_id and injection_id != "XXX":
        target_id = int(injection_id, 16)
        mask = in_interval & (ids == target_id)
    else:
        mask = in_interval

    n_filtered = mask.sum()
    if n_filtered == 0:
        print(f"{name:45s} | {naive_recall:>13.1%} | {'필터링 결과 0건':>18s} | 0")
        continue
    fixed_recall = flags[mask].mean()
    fixed_results.append(fixed_recall)
    print(f"{name:45s} | {naive_recall:>13.1%} | {fixed_recall:>17.1%} | {n_filtered:,}")

print("-" * 100)
if naive_results:
    print(f"\n[기존 방식] 평균 탐지율(전체파일 기준): {np.mean(naive_results):.1%}")
if fixed_results:
    print(f"[정정 방식] 평균 탐지율(실제 공격메시지 기준): {np.mean(fixed_results):.1%}")

out_path = os.path.join(HERE, "..", "results", "f3_road_v2_corrected.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("=== ROAD V2 평가 정정 (capture_metadata.json 기반 시간구간+ID 필터링) ===\n\n")
    if naive_results:
        f.write(f"기존 방식(전체파일 기준) 평균: {np.mean(naive_results):.1%}\n")
    if fixed_results:
        f.write(f"정정 방식(실제 공격메시지 기준) 평균: {np.mean(fixed_results):.1%}\n")
print(f"\n[완료] {out_path} 저장됨")
