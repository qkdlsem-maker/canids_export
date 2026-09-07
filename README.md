# CAN IDS — 하이브리드(LightGBM + Mahalanobis) 침입탐지 시스템

2026 K-DS 해커톤 본선 진출작. 전력변환/로봇제어 ECU 보호를 위한 CAN/CAN-FD 실시간 침입탐지 시스템.
팀 자비스(J.A.R.V.I.S). 산업체 과제: 에스엘(주).

> ⚠️ 이 README는 **본선 심사 피드백(F1~F7) 대응이 전부 반영된 최신 버전**입니다.
> 예선 제출본(5% 서브셋 기준)과는 데이터 규모·검증 방식이 다릅니다 — 4절 참고.

---

## 1. 개요

CAN 버스는 발신자 인증·암호화가 없어 DoS/Fuzzing/Spoofing 공격에 취약합니다. 본 프로젝트는:

- **1단계 LightGBM**: 알려진 공격(DoS/Fuzzy/gear/RPM)을 즉시 분류
- **2단계 Mahalanobis 거리 기반 이상탐지**: 1단계가 놓친 미학습(zero-day) 공격을 통계적으로 탐지

두 단계를 결합한 하이브리드 구조로, "알려진 공격은 유형까지, 미지 공격은 이상 여부까지" 판정합니다.

---

## 2. 본선 심사 피드백 대응표 (F1~F7)

| 코드 | 심사위원 요청 | 대응 결과 | 상태 |
|---|---|---|---|
| F1 | ICSim·can-utils 실시간 주입, 탐지율/오탐률/지연 정량화, Peak RAM/CPU | 기존 실시간 검증에 더해 **탐지기와 독립된 per-frame GT 체인**으로 재검증. GT→candump 9,485/9,485(100%), 처리 프레임 attack recall 100%, end-to-end recall 99.9789%, 별도 normal-only FPR 0.053538% | ✅ 완료 |
| F2 | 5%서브셋→전체 데이터 재현성 검증 | HCRL 전체(1,657만 행) 재학습, 시드 3회 반복(F1=1.0000±0, FPR=0.0192%±0) | ✅ 완료 |
| F3 | CAN ID 의존 낮은 feature set 재설계 | ID-agnostic 6피처 설계, V1/V2 모두 전체데이터 기준 ROAD 공정비교(V1 FPR 100%실패 vs V2 FPR 0.07%) | ✅ 완료 |
| F4 | LightGBM단독 vs Mahalanobis단독 vs Hybrid 비교 | 4축 비교 + 시드 3회 반복 + 탐지/유형정확 지표 분리(zero-day RPM: LGBM 0% vs Hybrid 100%) | ✅ 완료 |
| F5 | 실제 MCU Flash/RAM/CPU/WCET/탐지지연 | ARM Cortex-M4 크로스컴파일 + Renode 에뮬레이션. Renode 공식 `ElapsedCycles`(진짜 사이클 카운터)로 실측: Flash 254KB, 정적RAM 1.7KB, 최악(Fuzzy) WCET 908,880cycles=5.41ms(168MHz) | 🟢 완료(시뮬레이션 기준) |
| F6 | 기존 ECU 기능과 동시 동작(장시간·최대부하) | SysTick 동시성 + Renode bxCAN FIFO 대조실험에 더해 host SocketCAN/vcan 부하·장시간 검증. **약 17.2k observed fps에서 3/3회 100% processing coverage**, 약 20k fps에서 59분 steady-state 운용 시 throughput·latency·CPU·RSS의 유의한 열화 없음. 실제 MCU CAN FIFO/장시간 silicon 검증은 별도 필요 | 🟢 Host/시뮬레이션 검증 완료 |
| F7 | (선행 결함) C 변환기 정상 클래스 인덱스 오류 | NORMAL_IDX/CODE_MAP 도입, x86+ARM 회귀테스트로 검증 | ✅ 완료 |

**F5/F6 남은 한계**: formal WCET는 대표 입력에 대한 실측이며 모든 입력의 수학적 최악값 증명은 아니다. Renode bxCAN 및 Linux SocketCAN/vcan에서 FIFO·최대부하·약 1시간 장시간 안정성을 보강 검증했지만, **실제 target MCU silicon에서의 CAN-controller FIFO overflow immunity, CPU/resource behavior, 장시간 연속 운용은 아직 검증하지 않았다.** 따라서 Renode/host-side 결과는 실제 하드웨어 검증과 구분하여 보고한다.

---

## 3. 최종 시스템 아키텍처
CAN 버스 실시간 스트림
-> 특징공학 (8개 피처: 빈도/엔트로피/ID별 z-score 등)
-> 1단계 LightGBM (시그니처 기반, 알려진 공격 즉시 분류)
-> 알려진 공격이면: 유형(DoS/Fuzzy/gear/RPM)과 함께 즉시 경보
-> 정상으로 분류되면: 2단계로
-> 2단계 Mahalanobis 거리 기반 이상탐지 (정상 분포 기반)
-> 이상이면: "미지 이상패턴(Zero-day)"으로 경보
-> 정상이면: 통과

핵심 성능(전체 데이터셋, 5% 서브셋 아님):

| 지표 | 결과 |
|---|---|
| Macro F1 (알려진 공격) | 1.0000 ± 0.0000 (시드 3회) |
| Zero-day 탐지율(Hybrid) | 99.9~100% (LightGBM 단독은 0~100%로 불안정) |
| 하이브리드 e2e 지연 | 0.11~0.31ms (PC 기준) |
| MCU Flash / 정적 RAM | 254KB / 1.7KB |
| MCU WCET(5클래스, 168MHz, Renode ElapsedCycles 실측) | 2.18~5.41ms (최악: Fuzzy) |

---

## 4. 데이터 히스토리 (중요 — 여러 버전이 공존)

| 버전 | 데이터 | 용도 |
|---|---|---|
| `data/` | HCRL 5% 서브셋 | 예선 제출본, 참고용으로만 유지 |
| `data_full/` | HCRL 전체(DoS/Fuzzy/gear/RPM_dataset.csv, 1,657만 행, 실제 타임스탬프) | **F2 이후 모든 실험의 기준 데이터** |
| `road_data/` | ORNL ROAD Dataset(공격/정상 각각) | F3 교차검증 전용 |

**모델도 2가지가 공존합니다**: `models_c/can_ids_embedded.c`(서브셋 학습, 구버전) vs `models_c/can_ids_embedded_full.c`(전체데이터 학습, **최종 채택**). MCU 검증(F5/F6)은 `can_ids_embedded_full.c` 기준입니다.

---

## 5. 재현 가이드

### 5.1 전체 데이터 기반 하이브리드 재학습 (F2)
```bash
python3 src/10_build_features_full.py      # data_full/ 원본 파싱 + 피처공학
python3 src/11_train_full.py               # LightGBM+Mahalanobis 재학습
python3 src/18_train_full_repeated.py      # 시드 3회 반복(안정성 검증)
```

### 5.2 Ablation Study (F4)
```bash
python3 src/12_ablation_study.py           # LightGBM/Mahalanobis/Hybrid 4축 비교
python3 src/26_ablation_repeated.py        # 시드 3회 반복
python3 src/29_f4_hybrid_split_metrics.py  # 알려진 공격: 탐지 vs 유형정확 분리
python3 src/30_f4_hybrid_zeroday_split.py  # zero-day: 탐지 vs 유형정확 분리
```

### 5.3 ID-agnostic 재설계 + ROAD 교차검증 (F3)
```bash
python3 src/13_build_features_v2.py            # ID-agnostic 6피처 생성
python3 src/14_train_v2_and_road_test.py       # V2 학습 + ROAD 테스트(참고용 naive recall)
python3 src/20_road_test_v1_full.py            # V1도 전체데이터로 재학습 후 ROAD 재비교(공정비교)
python3 src/28_road_test_v2_corrected.py       # capture_metadata.json 기반 정밀 재평가(권장, 최종 수치)
```
⚠️ `14_train_v2_and_road_test.py`의 ROAD recall은 "공격 파일 전체"를 분모로 삼아 저평가되어 있습니다.
**최종 수치는 반드시 `28_road_test_v2_corrected.py` 결과를 사용하세요.**

### 5.4 MCU 검증 (F5/F6)
```bash
cd models_c
python3 07_export_to_c.py                  # data_full 기준 C 코드 생성 (can_ids_embedded_full.c)
cp can_ids_embedded_full.c fw_test/can_ids_embedded.c
cd fw_test
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=soft -O2 -nostartfiles \
  -T linker.ld -o can_ids_fw.elf startup.c main.c -lm --specs=nosys.specs
cd ../renode_portable
./renode --console -e "include @../run_ids3.resc"
```

### 5.4.1 F6 — 실제 bxCAN FIFO 오버런 재현
```bash
sudo ip link add dev vcan1 type vcan 2>/dev/null; sudo ip link set up vcan1

cd models_c/can_fw
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=soft -O2 -nostartfiles \
  -T linker.ld -o can_test_fw.elf startup.c main.c -lm --specs=nosys.specs
cd ..
./renode_portable/renode --console -e "include @run_can_test.resc"
```

### 5.5 실시간 스트리밍 검증 — ICSim (F1)
```bash
cd ICSim && make && cd ..
Xvfb :98 -screen 0 1024x768x16 &
export DISPLAY=:98
sudo ip link add dev vcan0 type vcan 2>/dev/null; sudo ip link set up vcan0
cd ICSim && ./icsim vcan0 & ./controls vcan0 & cd ..

candump -l vcan0 &  # 60초 후 Ctrl+C
python3 src/21_fit_icsim_stats.py ICSim/candump-*.log

python3 src/23_icsim_realtime_detect.py 90     # 터미널 1
python3 src/22_icsim_attack_inject.py 90       # 터미널 2 (1~2초 후)

python3 src/27_evaluate_icsim_fixed.py 0x39 0x294 0x143
```
⚠️ `24_evaluate_icsim.py`는 저평가된 결과를 냅니다. **최종 수치는 반드시 `27_evaluate_icsim_fixed.py`를 사용하세요.**

---

## 추가 Robustness Validation

### 1. Independent ICSim Ground Truth

ICSim 실시간 평가의 ground truth가 IDS 내부 feature 또는 prediction에 의존하지 않도록 독립적인 per-frame injection ground-truth chain을 구축하였다.

관련 스크립트:

- `src/42_icsim_gt_logger.py`
- `src/43_match_icsim_ground_truth.py`
- `src/44_evaluate_icsim_independent_gt.py`

#### Results

- Injected GT frames: 9,485
- GT -> candump matched: 9,485 / 9,485 = 100%
- Prediction matched: 9,483 / 9,485 = 99.9789%
- Processed-frame attack recall: 100%
- End-to-end attack recall: 99.9789%
- Normal-only frames: 117,673
- False positives: 63
- Normal-only FPR: 0.053538%

Ground-truth construction does NOT depend on:

- `value_zscore`
- `is_unknown_id`
- Mahalanobis distance
- model prediction
- attack-window heuristic

---

### 2. Host-Side Sustained CAN Load

관련 스크립트:

- `src/46_can_load_stress_detector.py`

Linux SocketCAN/vcan 환경에서 `candump`를 독립적인 입력 관측기로 사용하였다.

따라서 아래 FPS는 `cangen` requested rate가 아니라 candump에서 실제 관측된 입력률이다.

| Observed input | Processing coverage | Observed loss |
| ---: | ---: | ---: |
| 17,159.1 fps | 100.000000% | 0.000000% |
| 20,596.5 fps | 99.854861% | 0.145139% |
| 22,892.5 fps | 99.724704% | 0.275296% |
| 25,778.9 fps | 99.629726% | 0.370274% |
| ~939,871 fps extreme burst | 3.608642% | 96.391358% |

### Reproducibility at ~17.2k fps

| Run | Observed FPS | Coverage | p99 latency |
| ---: | ---: | ---: | ---: |
| 1 | 17,141.0 | 100.000000% | 0.0595 ms |
| 2 | 17,161.1 | 100.000000% | 0.0527 ms |
| 3 | 17,175.2 | 100.000000% | 0.0523 ms |
| Mean | 17,159.1 | 100.000000% | 0.054831 ms |

세 번의 독립 반복시험 모두 100% processing coverage를 기록하였다.

약 20.6k observed frames/s부터 작은 end-to-end processing loss가 관측되기 시작하였다.

이 loss를 Mahalanobis 계산시간 하나의 원인으로 해석하지 않는다. SocketCAN receive queue, Linux scheduling 및 observer socket을 포함한 host-side receive path 전체의 saturation 결과로 해석한다.

---

### 3. One-Hour Host-Side Stability

관련 스크립트:

- `src/47_long_duration_stability.py`

약 20k frames/s의 지속 입력으로 1시간 시험을 수행하였다.

시작/종료 경계구간을 제외한 interval 2-60, 약 59분을 steady-state 구간으로 평가하였다.

| Metric | Result |
| --- | ---: |
| Steady-state duration | ~59 min |
| Mean throughput | 19,980.069 fps |
| FPS standard deviation | 6.346 fps |
| FPS minimum | 19,965.066 fps |
| FPS maximum | 19,992.933 fps |
| First-5 -> Last-5 throughput | +0.002953% |
| FPS slope | -0.003015 fps/min |
| Mean p99 latency | 0.045725 ms |
| Maximum p99 latency | 0.056169 ms |
| First-5 -> Last-5 p99 | -0.001088 ms |
| Mean CPU | 54.375% |
| Maximum CPU | 56.500% |
| RSS first | 75.051 MB |
| RSS last | 75.121 MB |
| RSS growth | +0.070312 MB |

시험 중 progressive throughput degradation, latency escalation, crash 또는 hang은 관찰되지 않았다.

59분 동안 RSS 변화는 +0.070312 MB로 작았으며, 본 시험 시간 범위에서 material memory-growth behavior는 관찰되지 않았다.

---

### 4. V2 Mahalanobis Threshold Sensitivity

관련 스크립트:

- `src/45_mahalanobis_threshold_sensitivity.py`

V2 feature set, mean 및 covariance를 동결한 상태에서 threshold percentile만 변경하였다.

모든 threshold candidate는 HCRL train-normal distance만으로 계산하였다.

ROAD는 threshold 계산 또는 선택/tuning에 사용하지 않았다.

| Percentile | Threshold | HCRL FPR | HCRL Attack TPR | ROAD FPR | ROAD TPR | ROAD BA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 99.00 | 6.876150 | 0.276050% | 76.612382% | 1.060749% | 20.280361% | 59.609806% |
| 99.50 | 8.198411 | 0.142978% | 57.491468% | 0.375003% | 19.281762% | 59.453379% |
| 99.70 | 8.899195 | 0.087958% | 46.444491% | 0.226788% | 17.634661% | 58.703936% |
| 99.80 | 9.153870 | 0.057671% | 45.156786% | 0.189936% | 16.442160% | 58.126112% |
| 99.90 FINAL | 10.104021 | 0.028183% | 32.951678% | 0.079932% | 12.398709% | 56.159388% |
| 99.95 | 11.164184 | 0.012598% | 21.712455% | 0.048472% | 5.960758% | 52.956143% |
| 99.99 | 14.232271 | 0.002166% | 5.239631% | 0.000238% | 1.334591% | 50.667177% |

Saved final threshold:

`10.1040212650`

Recomputed HCRL train-normal 99.90 percentile:

`10.1040209354`

Absolute difference:

`3.29681745e-7`

따라서 numerical tolerance 내에서 기존 frozen threshold가 재현되었다.

Threshold sweep에서는 명확한 FPR-TPR trade-off가 나타났다. 따라서 99.90% 주변의 성능이 threshold-insensitive하다고 주장하지 않는다.

99.90%는 ROAD 성능을 보고 재선택한 최적점이 아니라 사전에 HCRL train-normal에서 정의한 보수적인 frozen operating point이다.

---

### 5. Final Model Status

최종 모델은 계속 V2 ID-agnostic Mahalanobis이다.

V3 temporal 및 V4 byte-temporal 실험은 post-hoc ablation이며 최종 모델 교체 근거로 사용하지 않는다.

---

### 6. Scope and Remaining Hardware Limitation

본 문서의 CAN load 및 long-duration 결과는 Linux SocketCAN/vcan 기반 host-side validation이다.

`vcan`은 실제 CAN arbitration 및 physical bit timing을 재현하지 않는다.

따라서 본 결과는 다음을 직접 입증하지 않는다.

- 실제 target MCU CAN-controller hardware FIFO overflow immunity
- 실제 MCU silicon에서의 장시간 안정성
- 실제 MCU에서의 CPU/resource behavior

Renode 및 host-side 결과는 실제 target hardware validation과 구분하여 보고한다.

---

## 6. 저장소 구조
canids_export/
├── data/ # 5% 서브셋 (구버전)
├── data_full/ # HCRL 전체 데이터셋 (F2 이후 기준)
├── road_data/ # ORNL ROAD Dataset + capture_metadata.json
├── ICSim/ # zombieCraig ICSim 클론 (F1)
├── src/
│ ├── 01~09 # 예선 제출본(5%서브셋) 파이프라인
│ ├── 10_build_features_full.py
│ ├── 11_train_full.py
│ ├── 12_ablation_study.py
│ ├── 13_build_features_v2.py
│ ├── 14_train_v2_and_road_test.py
│ ├── 15~17 # F1: 초기 실시간 검증(구버전)
│ ├── 18_train_full_repeated.py
│ ├── 20_road_test_v1_full.py
│ ├── 21_fit_icsim_stats.py
│ ├── 22_icsim_attack_inject.py
│ ├── 23_icsim_realtime_detect.py
│ ├── 24_evaluate_icsim.py # ⚠️ 구버전, 저평가됨
│ ├── 25_icsim_threshold_sweep.py
│ ├── 26_ablation_repeated.py
│ ├── 27_evaluate_icsim_fixed.py # ✅ 최종 평가
│ ├── 28_road_test_v2_corrected.py # ✅ 최종 ROAD 평가
│ ├── 29_f4_hybrid_split_metrics.py
│ └── 30_f4_hybrid_zeroday_split.py
├── models_c/
│ ├── 07_export_to_c.py
│ ├── can_ids_embedded.c
│ ├── can_ids_embedded_full.c
│ ├── test_main*.c
│ ├── renode_portable/
│ ├── fw_test/ # F5 ElapsedCycles 측정용
│ ├── can_fw/ # 실제 bxCAN 드라이버(F6 FIFO 오버런 실측용)
│ └── run_can_test.resc # vcan1↔canHub↔CAN1 연결 스크립트
└── results/

---

## 7. 알려진 한계 (Known Limitations)

1. **F5 CPU 점유율**: 베어메탈 단일 태스크 환경이라 측정 안 함. RTOS 환경에서 재정의 필요.
2. **F5 formal WCET**: Renode `ElapsedCycles`(진짜 사이클 카운터)로 5클래스 대표 샘플 기준 실측(최악 Fuzzy 5.41ms)했으나, 모든 가능한 입력에 대한 수학적 최악값(formal WCET)은 아님. 실제 실리콘 실측도 아직 아님(시뮬레이션 기준).
3. **F6 장시간 연속운용**: 실제 bxCAN 페리페럴로 5,000건 집중 전송 시 FIFO 오버런 0건까지 실측했으나, 수 시간 단위 연속 운용은 미검증(시뮬레이션 시간 제약).
4. **F3 ROAD 일부 공격 유형**: `max_speedometer`/`reverse_light` 계열(값-고정형 스푸핑)은 정밀 재평가해도 탐지율 0% — ID-agnostic 피처의 구조적 한계로 판단됨.
5. **평가 방법론 일반**: 배경 트래픽 밀도가 높은 환경에서는 "시간구간 기준" recall이 아니라 "실제 공격 메시지 기준" recall을 써야 함 — F1/F3 양쪽에서 반복 확인된 교훈.

---

## 8. 팀

자비스(J.A.R.V.I.S) — 국립부경대학교. 노신비(기획/발표), 최혜림(개발/실험), 김준상(개발/발표준비), 이유민(개발/디자인).
지도교수: 김태국 (futurenetworklab).
