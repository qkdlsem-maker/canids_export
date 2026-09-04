"""
ICSim이 만든 '정상' 트래픽(candump 로그)을 파싱해서, 우리 하이브리드 파이프라인과
동일한 방식(known_ids, ID별 delta/byte 통계)으로 통계를 학습.

이건 "완전히 새로운 차량에 IDS를 처음 배포할 때, 그 차량의 idle 상태 트래픽으로
먼저 캘리브레이션한다"는 실제 배포 시나리오를 그대로 재현한 것.

사용법: python3 21_fit_icsim_stats.py <candump 로그 경로>
"""
import sys, os, re, pickle
import numpy as np

HERE = os.path.dirname(__file__)
LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "candump.log"
WINDOW = 20

LINE_RE = re.compile(r"\(([\d.]+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)")

def parse_log(path):
    ids, payloads = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            cid = int(m.group(2), 16)
            hexdata = m.group(3)
            byts = [int(hexdata[j:j+2], 16) for j in range(0, min(len(hexdata), 16), 2)]
            byts = (byts + [0] * 8)[:8]
            ids.append(cid)
            payloads.append(byts)
    return np.array(ids), np.array(payloads, dtype=np.int64)

def main():
    print(f"[파싱] {LOG_PATH}")
    ids, payloads = parse_log(LOG_PATH)
    n = len(ids)
    print(f"  -> {n:,}개 메시지")

    known_ids = set(np.unique(ids))
    print(f"  고유 ID: {len(known_ids)}개")

    per_id_delta, per_id_bytes = {}, {}
    last_payload = {}
    for i in range(n):
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

    all_deltas = [d for vals in per_id_delta.values() for d in vals]
    g_delta_mean, g_delta_std = float(np.mean(all_deltas)), float(np.std(all_deltas) + 1e-3)

    stats = {
        "known_ids": known_ids, "id_delta_stats": id_delta_stats, "id_byte_stats": id_byte_stats,
        "global_delta_mean": g_delta_mean, "global_delta_std": g_delta_std, "window": WINDOW,
    }
    stats_path = os.path.join(HERE, "..", "models", "icsim_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"[저장] {stats_path}")

    def payload_entropy(b):
        v, c = np.unique(b, return_counts=True)
        p = c / c.sum()
        return -np.sum(p * np.log2(p + 1e-12))

    feats = np.zeros((n, 8), dtype=np.float32)
    id_window, last_payload_per_id = [], {}
    for i in range(n):
        cid, pl = ids[i], payloads[i]
        id_window.append(cid)
        if len(id_window) > WINDOW:
            id_window.pop(0)

        freq_in_window = id_window.count(cid) / len(id_window)
        unique_ids_in_window = len(set(id_window))
        is_unknown_id = 0.0
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

    mean_ = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False) + np.eye(8) * 1e-6
    inv_cov = np.linalg.inv(cov)
    diff = feats - mean_
    dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))
    thr = np.percentile(dist, 99.9)

    maha_path = os.path.join(HERE, "..", "models", "maha_detector_icsim.npz")
    np.savez(maha_path, mean=mean_, inv_cov=inv_cov, thr=thr)
    print(f"[저장] {maha_path}")
    print(f"\n[캘리브레이션 완료] threshold={thr:.3f}")
    print("이제 21_realtime_icsim.py로 실시간 탐지 + 공격 주입 테스트 진행")

if __name__ == "__main__":
    main()
