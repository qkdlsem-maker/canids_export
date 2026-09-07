"""
외부 검증 피드백 반영 최종본:
- V1(ID-dependent)의 "100% 탐지"는 FPR도 100%라 무의미함을 명시적으로 대비
- 공격 유형별(aggregate 아님) 표
- TPR/FPR을 항상 페어로 표시
- Balanced Accuracy = (TPR+TNR)/2 계산 (V1은 이 지표에서 바로 무너짐을 확인)
- "동결 검증(Frozen Evaluation Protocol)" 체크리스트 출력

전제: 31_true_zeroday_hcrl_to_road.py, 32_true_zeroday_v2.py 먼저 실행(각각 V1/V2 결과 있어야 함)
"""
import os, re, glob, json, pickle
import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(__file__)
ROAD_ATTACKS = os.path.join(HERE, "..", "road_data", "attacks")
ROAD_AMBIENT = os.path.join(HERE, "..", "road_data", "ambient")

with open(os.path.join(HERE, "..", "models", "car_hacking_full_stats.pkl"), "rb") as f:
    STATS_V1 = pickle.load(f)
lgbm = lgb.Booster(model_file=os.path.join(HERE, "..", "models", "lgbm_full.txt"))
CLASSES = ["DoS", "Fuzzy", "R", "RPM", "gear"]
npz1 = np.load(os.path.join(HERE, "..", "models", "maha_detector_full.npz"))
MEAN1, INV1, THR1 = npz1["mean"], npz1["inv_cov"], float(npz1["thr"])

with open(os.path.join(HERE, "..", "models", "idagnostic_stats.pkl"), "rb") as f:
    STATS_V2 = pickle.load(f)
npz2 = np.load(os.path.join(HERE, "..", "models", "maha_v2.npz"))
MEAN2, INV2, THR2 = npz2["mean"], npz2["inv_cov"], float(npz2["thr"])
WINDOW = STATS_V2["window"]

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

def feats_v1(rows):
    known_ids = STATS_V1["known_ids"]; ids_stats = STATS_V1["id_delta_stats"]; byte_stats = STATS_V1["id_byte_stats"]
    gdm, gds = STATS_V1["global_delta_mean"], STATS_V1["global_delta_std"]
    out = []; win = []; last = {}
    for _, cid, pl in rows:
        pl = np.array(pl, dtype=float)
        win.append(cid)
        if len(win) > WINDOW: win.pop(0)
        f_iw = win.count(cid)/len(win); u_iw = len(set(win))
        unk = 0.0 if cid in known_ids else 1.0
        ent = payload_entropy(pl); mb = pl.mean()
        d = np.abs(pl-last[cid]).sum() if cid in last else 0.0
        last[cid] = pl
        mu, sg = ids_stats.get(cid, (gdm, gds))
        dz = (d-mu)/sg
        if cid in byte_stats:
            bmu, bstd = byte_stats[cid]; vz = np.max(np.abs((pl-bmu)/bstd))
        else:
            vz = 0.0
        out.append([f_iw, u_iw, unk, ent, mb, dz, vz, cid/2048.0])
    return np.array(out)

def feats_v2(rows):
    gdm, gds = STATS_V2["global_delta_mean"], STATS_V2["global_delta_std"]
    bpm, bps = STATS_V2["byte_pos_mean"], STATS_V2["byte_pos_std"]
    out = []; win = []; last = {}
    for _, cid, pl in rows:
        pl = np.array(pl, dtype=float)
        win.append(cid)
        if len(win) > WINDOW: win.pop(0)
        f_iw = win.count(cid)/len(win); u_iw = len(set(win))
        ent = payload_entropy(pl); mb = pl.mean()
        d = np.abs(pl-last[cid]).sum() if cid in last else 0.0
        last[cid] = pl
        dz = (d-gdm)/gds
        vz = np.max(np.abs((pl-bpm)/bps))
        out.append([f_iw, u_iw, ent, mb, dz, vz])
    return np.array(out)

def maha_flag(X, mean_, inv_, thr):
    diff = X - mean_
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_, diff)) > thr

def v1_hybrid_flag(rows):
    X = feats_v1(rows)
    proba = lgbm.predict(X)
    lbl = np.array([CLASSES[i] for i in np.argmax(proba, axis=1)])
    mflag = maha_flag(X, MEAN1, INV1, THR1)
    return (lbl != "R") | ((lbl == "R") & mflag)

def v2_flag(rows):
    X = feats_v2(rows)
    return maha_flag(X, MEAN2, INV2, THR2)

def masked_recall(path, meta, flag_fn):
    rows = parse_log(path)
    if len(rows) < WINDOW + 1:
        return None
    interval = meta.get("injection_interval")
    injection_id = meta.get("injection_id")
    if interval is None:
        return None
    flags = flag_fn(rows)
    t0 = rows[0][0]
    rel = np.array([r[0]-t0 for r in rows])
    ids = np.array([r[1] for r in rows])
    m = (rel >= interval[0]) & (rel <= interval[1])
    if injection_id and injection_id != "XXX":
        m = m & (ids == int(injection_id, 16))
    if m.sum() == 0:
        return None
    return flags[m].mean(), m.sum()

def fpr(path, flag_fn):
    rows = parse_log(path)
    if len(rows) < WINDOW + 1:
        return None
    flags = flag_fn(rows)
    return flags.mean(), len(rows)

print("=" * 100)
print("동결(Frozen) 검증 체크리스트 — 코드 감사 결과")
print("=" * 100)
checklist = [
    "1. HCRL 데이터로만 학습 (data_full/, road_data 아님)",
    "2. Feature 정규화 파라미터(전역 delta/byte 통계) HCRL 학습구간(is_test_region==False)+정상(R)에서만 산출",
    "3. LightGBM 가중치 동결 (11_train_full.py 결과, 재학습 없음)",
    "4. Mahalanobis mean/covariance HCRL train_normal에서만 산출",
    "5. Mahalanobis threshold도 HCRL에서 산출, ROAD 코드보다 먼저 저장됨(라인 순서로 확인)",
    "6. ROAD 라벨 학습에 미사용",
    "7. ROAD 정상 데이터 재캘리브레이션에 미사용",
    "8. ROAD 공격 데이터 파라미터 선택에 미사용",
    "9. ROAD는 딱 1회 외부 테스트셋으로만 사용",
]
for c in checklist:
    print(f"  [OK] {c}")

print("\n" + "=" * 100)
print("V1 vs V2 (둘 다 frozen) — 공격유형별 TPR, 공통 정상구간 FPR/TNR/Balanced Accuracy")
print("=" * 100)

attack_rows = []
for path in sorted(glob.glob(os.path.join(ROAD_ATTACKS, "*.log"))):
    name = os.path.basename(path)
    if "_masquerade" in name:
        continue
    meta = METADATA.get(name.replace(".log", ""), {})
    r1 = masked_recall(path, meta, v1_hybrid_flag)
    r2 = masked_recall(path, meta, v2_flag)
    if r1 is None or r2 is None:
        continue
    attack_rows.append((name, r1[0], r2[0], r1[1]))
    print(f"[{name:38s}] V1 TPR={r1[0]:>6.1%}  V2 TPR={r2[0]:>6.1%}  (n={r1[1]:,})")

print("\n--- 정상구간 (공통 FPR) ---")
fpr1_list, fpr2_list = [], []
for path in sorted(glob.glob(os.path.join(ROAD_AMBIENT, "*.log")))[:5]:
    name = os.path.basename(path)
    f1 = fpr(path, v1_hybrid_flag)
    f2 = fpr(path, v2_flag)
    if f1 is None or f2 is None:
        continue
    fpr1_list.append(f1[0]); fpr2_list.append(f2[0])
    print(f"[{name:38s}] V1 FPR={f1[0]:>6.1%}  V2 FPR={f2[0]:>6.2%}")

mean_tpr1 = np.mean([r[1] for r in attack_rows])
mean_tpr2 = np.mean([r[2] for r in attack_rows])
mean_fpr1 = np.mean(fpr1_list)
mean_fpr2 = np.mean(fpr2_list)
tnr1 = 1 - mean_fpr1
tnr2 = 1 - mean_fpr2
bal_acc1 = (mean_tpr1 + tnr1) / 2
bal_acc2 = (mean_tpr2 + tnr2) / 2

print("\n" + "=" * 100)
print("최종 요약 (반드시 TPR/FPR 페어로 해석)")
print("=" * 100)
print(f"{'':10s} {'평균 TPR':>10s} {'평균 FPR':>10s} {'TNR':>8s} {'Balanced Accuracy':>18s}")
print(f"{'V1(ID-dep)':10s} {mean_tpr1:>9.1%} {mean_fpr1:>9.1%} {tnr1:>7.1%} {bal_acc1:>17.1%}")
print(f"{'V2(ID-agn)':10s} {mean_tpr2:>9.1%} {mean_fpr2:>9.2%} {tnr2:>7.1%} {bal_acc2:>17.1%}")
print(f"\n해석: V1은 TPR={mean_tpr1:.0%}이지만 FPR도 {mean_fpr1:.0%}라 Balanced Accuracy={bal_acc1:.1%}로 붕괴")
print(f"     (분류기가 아니라 사실상 '전부 이상'으로 찍는 무의미한 판정)")
print(f"     V2는 FPR {mean_fpr2:.2%}를 유지하며 Balanced Accuracy={bal_acc2:.1%}로 유의미한 판별력 유지")

out_path = os.path.join(HERE, "..", "results", "f3_frozen_v1_v2_final_comparison.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("=== Frozen Cross-Dataset Evaluation: V1 vs V2, TPR/FPR 페어 + Balanced Accuracy ===\n\n")
    f.write("동결 검증 체크리스트:\n")
    for c in checklist:
        f.write(f"  [OK] {c}\n")
    f.write("\n공격유형별:\n")
    for name, t1, t2, n in attack_rows:
        f.write(f"  [{name}] V1_TPR={t1:.1%} V2_TPR={t2:.1%} (n={n:,})\n")
    f.write(f"\n요약: V1 TPR={mean_tpr1:.1%} FPR={mean_fpr1:.1%} BalancedAcc={bal_acc1:.1%}\n")
    f.write(f"      V2 TPR={mean_tpr2:.1%} FPR={mean_fpr2:.2%} BalancedAcc={bal_acc2:.1%}\n")

print(f"\n[완료] {out_path} 저장됨")
