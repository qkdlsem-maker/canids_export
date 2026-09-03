"""
vcan0에 정상 트래픽을 지속적으로 흘려보내면서, 주기적으로 라벨링된 공격 버스트를 주입.
학습된 모델의 known_ids/id_byte_stats를 그대로 재사용해 "그럴듯한 정상값" 생성.

사용법: python3 15_traffic_gen.py [지속시간(초), 기본 90]
출력: road_data와 별개로 results/realtime_ground_truth.csv (start_ts,end_ts,label)
"""
import can, time, random, pickle, os, sys, csv
import numpy as np

HERE = os.path.dirname(__file__)
DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 90
CHANNEL = "vcan0"

with open(os.path.join(HERE, "..", "models", "car_hacking_full_stats.pkl"), "rb") as f:
    STATS = pickle.load(f)
known_ids = sorted(STATS["known_ids"])
id_byte_stats = STATS["id_byte_stats"]

# 정상 트래픽에 쓸 ID 풀 (학습된 known_ids 중 일부 샘플)
normal_ids = random.sample(known_ids, min(15, len(known_ids)))
# 공격 대상으로 쓸 ID (스푸핑용, known_ids 중에서 별도 선정)
gear_id = random.choice(known_ids)
rpm_id = random.choice([i for i in known_ids if i != gear_id])
dos_id = min(known_ids)  # 최우선순위 ID로 DoS 흉내

bus = can.interface.Bus(channel=CHANNEL, bustype="socketcan")

last_normal_payload = {}

def gen_normal_payload(cid):
    mu, std = id_byte_stats.get(cid, ([128.0]*8, [50.0]*8))
    mu = np.array(mu); std = np.array(std)
    phi = 0.85
    is_const = std < 1.0  # 학습 데이터에서 사실상 고정값이었던 바이트 위치(분산≈0, 1e-3 플로어)

    if cid in last_normal_payload:
        prev = last_normal_payload[cid]
        noise = np.random.normal(0, std * np.sqrt(1 - phi**2), size=8)
        pl = mu + phi * (prev - mu) + noise
    else:
        pl = np.random.normal(mu, std)

    pl = np.where(is_const, np.round(mu), pl)  # 고정 바이트는 노이즈 없이 학습된 평균값 그대로 사용
    pl = np.clip(pl, 0, 255)
    pl = [int(v) for v in pl]
    last_normal_payload[cid] = np.array(pl, dtype=float)
    return pl

def send(cid, payload):
    msg = can.Message(arbitration_id=cid, data=bytes(payload[:8]), is_extended_id=False)
    bus.send(msg)

def inject_dos(t_end):
    while time.time() < t_end:
        send(dos_id, [0, 0, 0, 0, 0, 0, 0, 0])
        time.sleep(0.001)

def inject_fuzzy(t_end):
    while time.time() < t_end:
        cid = random.randint(0, 0x7FF)
        pl = [random.randint(0, 255) for _ in range(8)]
        send(cid, pl)
        time.sleep(0.003)

def inject_spoof(t_end, cid):
    while time.time() < t_end:
        mu, std = id_byte_stats.get(cid, ([128]*8, [50]*8))
        # 정상 범위에서 크게 벗어난(수십 표준편차) 값으로 스푸핑
        pl = [int(max(0, min(255, mu[i] + random.choice([-1, 1]) * std[i] * 15))) for i in range(8)]
        send(cid, pl)
        time.sleep(0.01)

ATTACKS = [
    ("DoS", inject_dos, None),
    ("Fuzzy", inject_fuzzy, None),
    ("gear", inject_spoof, gear_id),
    ("RPM", inject_spoof, rpm_id),
]

# ---- ID별 고유 주기 스케줄러 (실제 CAN 버스: 빠른 센서 ~10-30ms, 느린 상태정보 ~100-500ms) ----
id_periods = {}
id_next_send = {}
now0 = time.time()
for i, cid in enumerate(normal_ids):
    period = random.choice([0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5])
    id_periods[cid] = period
    id_next_send[cid] = now0 + random.uniform(0, period)

def send_due_normal_messages():
    now = time.time()
    for cid, next_t in list(id_next_send.items()):
        if now >= next_t:
            send(cid, gen_normal_payload(cid))
            id_next_send[cid] = now + id_periods[cid] + random.uniform(-0.001, 0.001)

print(f"[traffic_gen] 시작. 정상 ID 풀: {[hex(i) for i in normal_ids]}")
print(f"[traffic_gen] gear_id={hex(gear_id)} rpm_id={hex(rpm_id)} dos_id={hex(dos_id)}")

gt_path = os.path.join(HERE, "..", "results", "realtime_ground_truth.csv")
os.makedirs(os.path.dirname(gt_path), exist_ok=True)
gt_file = open(gt_path, "w", newline="")
gt_writer = csv.writer(gt_file)
gt_writer.writerow(["start_ts", "end_ts", "label"])

t_start = time.time()
t_finish = t_start + DURATION
next_attack_at = t_start + random.uniform(5, 10)

while time.time() < t_finish:
    now = time.time()
    if now >= next_attack_at:
        label, fn, arg = random.choice(ATTACKS)
        burst_dur = random.uniform(2.0, 4.0)
        a_start = time.time()
        a_end = a_start + burst_dur
        print(f"[traffic_gen] >>> {label} 공격 주입 시작 ({burst_dur:.1f}s)")
        if arg is None:
            fn(a_end)
        else:
            fn(a_end, arg)
        gt_writer.writerow([a_start, a_end, label])
        gt_file.flush()
        print(f"[traffic_gen] <<< {label} 공격 종료")
        next_attack_at = time.time() + random.uniform(5, 10)
    else:
        send_due_normal_messages()
        time.sleep(0.002)

gt_file.close()
print(f"[traffic_gen] 완료. ground truth 저장: {gt_path}")
