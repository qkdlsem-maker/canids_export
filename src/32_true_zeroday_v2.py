"""
31번(V1, ID-dependent)은 ROAD에서 FPR도 100%로 나와 "전부 이상치로 찍는" 무의미한 결과였음
(F3에서 이미 확인한 V1의 근본 한계가 재현된 것). ID-agnostic(V2) 모델로 같은 실험을 재실행해서
진짜 판별력 있는 zero-day 결과를 확인.

HCRL로 학습(1회) -> V2(ID-agnostic) Mahalanobis 완전 동결 -> ROAD에 그대로 적용.

전제: 13_build_features_v2.py(idagnostic_stats.pkl), 14_train_v2_and_road_test.py(maha_v2.npz)
      실행 완료 상태여야 함.
"""
import os, re, glob, json, pickle
import numpy as np

HERE = os.path.dirname(__file__)
ROAD_ATTACKS = os.path.join(HERE, "..", "road_data", "attacks")
ROAD_AMBIENT = os.path.join(HERE, "..", "road_data", "ambient")

with open(os.path.join(HERE, "..", "models", "idagnostic_stats.pkl"), "rb") as f:
    STATS = pickle.load(f)
WINDOW = STATS["window"]
g_delta_mean, g_delta_std = STATS["global_delta_mean"], STATS["global_delta_std"]
byte_pos_mean, byte_pos_std = STATS["byte_pos_mean"], STATS["byte_pos_std"]

npz = np.load(os.path.join(HERE, "..", "models", "maha_v2.npz"))
MEAN_, INV_COV, THR = npz["mean"], npz["inv_cov"], float(npz["thr"])

print("[동결된 HCRL V2(ID-agnostic) 모델 로드 완료] 재학습/threshold조정 없음\n")

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

def evaluate_attack_file(path, meta):
    injection_id = meta.get("injection_id")
    interval = meta.get("injection_interval")
    rows = parse_log(path)
    if len(rows) < WINDOW + 1 or interval is None:
        return None
    X = extract_features_v2(rows)
    flags = maha_flag(X)

    t0 = rows[0][0]
    rel_ts = np.array([r[0] - t0 for r in rows])
    ids = np.array([r[1] for r in rows])
    in_interval = (rel_ts >= interval[0]) & (rel_ts <= interval[1])
    if injection_id and injection_id != "XXX":
        mask = in_interval & (ids == int(injection_id, 16))
    else:
        mask = in_interval

    n = mask.sum()
    if n == 0:
        return None
    return flags[mask].mean(), n

def evaluate_ambient_file(path):
    rows = parse_log(path)
    if len(rows) < WINDOW + 1:
        return None
    X = extract_features_v2(rows)
    flags = maha_flag(X)
    return flags.mean(), len(rows)

print("=" * 80)
print("[진짜 Zero-day 검증 V2] HCRL 학습 ID-agnostic 모델(동결) -> ROAD 공격 탐지")
print("=" * 80)
attack_results = []
for path in sorted(glob.glob(os.path.join(ROAD_ATTACKS, "*.log"))):
    name = os.path.basename(path)
    if "_masquerade" in name:
        continue
    meta = METADATA.get(name.replace(".log", ""), {})
    result = evaluate_attack_file(path, meta)
    if result is None:
        continue
    recall, n = result
    attack_results.append((name, recall, n))
    print(f"[{name}] 탐지율={recall:.1%}  n={n:,}")

print("\n" + "=" * 80)
print("[정상구간 오탐율] ROAD ambient")
print("=" * 80)
ambient_results = []
for path in sorted(glob.glob(os.path.join(ROAD_AMBIENT, "*.log")))[:5]:
    name = os.path.basename(path)
    result = evaluate_ambient_file(path)
    if result is None:
        continue
    fpr, n = result
    ambient_results.append((name, fpr))
    print(f"[{name}] FPR={fpr:.2%}  n={n:,}")

print("\n" + "=" * 80)
if attack_results:
    print(f"[요약] 평균 탐지율={np.mean([r[1] for r in attack_results]):.1%}")
if ambient_results:
    print(f"[요약] 평균 FPR={np.mean([r[1] for r in ambient_results]):.2%}")
print("(FPR이 낮으면서 탐지율이 유의미하게 나와야 '진짜 판별력 있는 zero-day 검증'으로 인정 가능)")

out_path = os.path.join(HERE, "..", "results", "true_zeroday_v2_hcrl_to_road.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("=== 진짜 Zero-day 검증(V2, ID-agnostic): HCRL 학습(동결) -> ROAD 적용 ===\n\n")
    f.write("공격 탐지:\n")
    for name, r, n in attack_results:
        f.write(f"  [{name}] 탐지율={r:.1%} (n={n:,})\n")
    f.write("\n정상구간 오탐율:\n")
    for name, fpr in ambient_results:
        f.write(f"  [{name}] FPR={fpr:.2%}\n")

print(f"\n[완료] {out_path} 저장됨")
