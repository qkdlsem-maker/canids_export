"""
피드백 반영: 기존 30_f4_hybrid_zeroday_split.py는 HCRL 내에서 공격 유형 1개만 학습에서
제외한 leave-one-attack-out이라, "같은 차량·같은 정상분포·같은 feature 생성 방식" 안에서의
비교일 뿐 강한 의미의 zero-day 증거는 아님.

이 스크립트는:
  HCRL로 학습(1회) -> LightGBM+Mahalanobis 완전 동결(재학습/threshold 조정 절대 안 함)
  -> ROAD Dataset(전혀 다른 차량, 전혀 다른 공격 유형)에 그대로 적용
  -> 미학습 공격이 실제로 탐지되는지 확인

ROAD의 capture_metadata.json(injection_id+injection_interval)로 실제 공격 메시지만
정밀 필터링(F3에서 확립한 방법론과 동일하게 적용, naive 시간구간 희석 문제 회피).

전제: 11_train_full.py(lgbm_full.txt, maha_detector_full.npz), 10_build_features_full.py
      (car_hacking_full_stats.pkl) 실행 완료 상태여야 함.
"""
import os, re, glob, json, pickle
import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(__file__)
ROAD_ATTACKS = os.path.join(HERE, "..", "road_data", "attacks")
ROAD_AMBIENT = os.path.join(HERE, "..", "road_data", "ambient")
FEATS = ["freq_in_window", "unique_ids_in_window", "is_unknown_id", "entropy",
          "mean_byte", "delta_zscore", "value_zscore", "norm_id"]

with open(os.path.join(HERE, "..", "models", "car_hacking_full_stats.pkl"), "rb") as f:
    STATS = pickle.load(f)
WINDOW = STATS["window"]
known_ids = STATS["known_ids"]
id_delta_stats = STATS["id_delta_stats"]
id_byte_stats = STATS["id_byte_stats"]
g_delta_mean, g_delta_std = STATS["global_delta_mean"], STATS["global_delta_std"]

lgbm = lgb.Booster(model_file=os.path.join(HERE, "..", "models", "lgbm_full.txt"))
CLASSES = ["DoS", "Fuzzy", "R", "RPM", "gear"]

npz = np.load(os.path.join(HERE, "..", "models", "maha_detector_full.npz"))
MEAN_, INV_COV, THR = npz["mean"], npz["inv_cov"], float(npz["thr"])

print(f"[동결된 HCRL 모델 로드 완료] known_ids={len(known_ids)}개(전부 HCRL 차량 기준, ROAD와 무관)")
print("재학습 없음, threshold 조정 없음 — 이 스크립트에서 학습 관련 코드는 전혀 실행되지 않음\n")

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

def extract_features_v1(rows):
    feats = []
    id_window, last_payload_per_id = [], {}
    for _, cid, pl in rows:
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

def lgbm_predict_labels(X):
    proba = lgbm.predict(X)
    idx = np.argmax(proba, axis=1)
    return np.array([CLASSES[i] for i in idx])

def hybrid_decide(X):
    lgbm_labels = lgbm_predict_labels(X)
    maha_flags = maha_flag(X)
    final = []
    for lbl, mf in zip(lgbm_labels, maha_flags):
        if lbl != "R":
            final.append(lbl)
        elif mf:
            final.append("unknown_anomaly")
        else:
            final.append("R")
    return np.array(final), lgbm_labels, maha_flags

def evaluate_attack_file(path, meta):
    injection_id = meta.get("injection_id")
    interval = meta.get("injection_interval")
    rows = parse_log(path)
    if len(rows) < WINDOW + 1 or interval is None:
        return None

    X = extract_features_v1(rows)
    final, lgbm_labels, maha_flags = hybrid_decide(X)

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

    hybrid_recall = (final[mask] != "R").mean()
    maha_recall = maha_flags[mask].mean()
    lgbm_nonR_rate = (lgbm_labels[mask] != "R").mean()
    return hybrid_recall, maha_recall, lgbm_nonR_rate, n

def evaluate_ambient_file(path):
    rows = parse_log(path)
    if len(rows) < WINDOW + 1:
        return None
    X = extract_features_v1(rows)
    final, lgbm_labels, maha_flags = hybrid_decide(X)
    fpr_hybrid = (final != "R").mean()
    fpr_maha = maha_flags.mean()
    return fpr_hybrid, fpr_maha, len(rows)

print("=" * 90)
print("[진짜 Zero-day 검증] HCRL 학습 하이브리드(동결) -> ROAD(전혀 다른 차량) 공격 탐지")
print("=" * 90)
attack_results = []
for path in sorted(glob.glob(os.path.join(ROAD_ATTACKS, "*.log"))):
    name = os.path.basename(path)
    if "_masquerade" in name:
        continue
    meta = METADATA.get(name.replace(".log", ""), {})
    result = evaluate_attack_file(path, meta)
    if result is None:
        continue
    hybrid_recall, maha_recall, lgbm_nonR, n = result
    attack_results.append((name, hybrid_recall, maha_recall, lgbm_nonR, n))
    print(f"[{name}] Hybrid={hybrid_recall:.1%}  (2단계 Maha={maha_recall:.1%}, 1단계 LGBM비R비율={lgbm_nonR:.1%})  n={n:,}")

print("\n" + "=" * 90)
print("[정상구간 오탐율] ROAD ambient(전혀 다른 차량의 정상 운행)")
print("=" * 90)
ambient_results = []
for path in sorted(glob.glob(os.path.join(ROAD_AMBIENT, "*.log")))[:5]:
    name = os.path.basename(path)
    result = evaluate_ambient_file(path)
    if result is None:
        continue
    fpr_hybrid, fpr_maha, n = result
    ambient_results.append((name, fpr_hybrid, fpr_maha))
    print(f"[{name}] Hybrid FPR={fpr_hybrid:.1%}  (2단계 Maha FPR={fpr_maha:.1%})  n={n:,}")

print("\n" + "=" * 90)
if attack_results:
    print(f"[요약] 평균 Hybrid 공격탐지율={np.mean([r[1] for r in attack_results]):.1%}, "
          f"평균 2단계(Maha)탐지율={np.mean([r[2] for r in attack_results]):.1%}")
if ambient_results:
    print(f"[요약] 평균 Hybrid FPR={np.mean([r[1] for r in ambient_results]):.1%}, "
          f"평균 2단계(Maha) FPR={np.mean([r[2] for r in ambient_results]):.1%}")

out_path = os.path.join(HERE, "..", "results", "true_zeroday_hcrl_to_road.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("=== 진짜 Zero-day 검증: HCRL 학습(동결) -> ROAD 적용, 재학습/threshold조정 없음 ===\n\n")
    f.write("공격 탐지:\n")
    for name, hy, mh, lg, n in attack_results:
        f.write(f"  [{name}] Hybrid={hy:.1%} Maha2단계={mh:.1%} LGBM1단계비R={lg:.1%} (n={n:,})\n")
    f.write("\n정상구간 오탐율:\n")
    for name, fh, fm in ambient_results:
        f.write(f"  [{name}] Hybrid_FPR={fh:.1%} Maha_FPR={fm:.1%}\n")

print(f"\n[완료] {out_path} 저장됨")
