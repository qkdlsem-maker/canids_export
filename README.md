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
| F1 | ICSim·can-utils 실시간 주입, 탐지율/오탐률/지연 정량화, Peak RAM/CPU | 기존 실시간 검증에 더해 탐지기와 독립된 per-frame GT 체인으로 재검증. GT→candump 9,485/9,485 (100%), 처리 프레임 attack recall 100%, end-to-end recall 99.9789%, 별도 normal-only FPR 0.053538% | ✅ 완료 |
| F2 | 5%서브셋→전체 데이터 재현성 검증 | HCRL 전체 약 1,657만 행 재학습 및 시드 3회 반복 검증. F1 1.0000±0, FPR 0.0192%±0 | ✅ 완료 |
| F3 | CAN ID 의존 낮은 feature set 재설계 | ID-agnostic 6-feature Frozen V2 설계. 전체 데이터 기준 ROAD 공정 비교에서 V1 FPR 100% 실패 → V2 FPR **0.079932%**로 개선 | ✅ 완료 |
| HCRL 내부 zero-day 탐지(Hybrid) | 기존 동일-domain/held-out 평가에서 99.9~100% 수준 | HCRL 내부 held-out/zero-day 조건에서 99.9~100% 수준 확인. 단, 동일 데이터셋 계열 내부 평가이므로 외부 일반화 성능과 구분 | ✅ 완료 |
| ROAD 외부 일반화 — V2 frozen threshold | TPR 12.398709%, FPR 0.079932%, Balanced Accuracy 56.159388% | ORNL ROAD 외부 데이터셋에서 Frozen V2 기준 TPR 12.398709%, FPR 0.079932%, Balanced Accuracy 56.159388%. 외부 미지 공격 탐지 성능의 한계를 포함해 그대로 보고 | ✅ 완료 |
| F5 | 실제 MCU Flash/RAM/CPU/WCET/탐지지연 | NUCLEO-F446RE (STM32F446RE, Cortex-M4, 84 MHz)에서 Frozen V2 실제 구동. 8개 검증 벡터 × 12,500회 = 100,000회 physical inference, 판정 불일치 0회. DWT 기준 최소 1.574 ms, 평균 1.835 ms, 관측 최대 1.977 ms. Static RAM 524 B, observed stack high-water 368 B, 합산 관측치 892 B. 관측 최대값은 formal WCET가 아님 | 🟢 Renode/pre-hardware + physical MCU F5 완료 |
| F6 | 기존 ECU 기능과 동시 동작(장시간·최대부하) | Renode에서 CAN1 FIFO0→RX IRQ→SW queue→V2 feature→Frozen Hybrid + 100 Hz synthetic control task 동시 실행. 843.75 fps 3/3 internal lossless, 847.65625 fps에서 3/3 SW-queue loss 발생. 60분 nominal 500 fps에서 CAN1 수신 프레임 전부 처리, SW overflow 0, control deadline miss 0. 실제 CAN-controller 기반 physical 장시간·최대부하 검증은 미완료이며 Renode 결과와 구분 | 🟡 Renode 정량 검증 완료 / Physical F6 미완료 |
| F7 | (선행 결함) C 변환기 정상 클래스 인덱스 오류 | NORMAL_IDX/CODE_MAP 도입 및 x86 + ARM 회귀테스트로 검증 | ✅ 완료 |

**F5/F6 검증 범위와 남은 한계:** F5는 실제 **NUCLEO-F446RE (STM32F446RE, Cortex-M4, 84 MHz)**에서 Frozen V2 IDS를 구동하여 추론 지연, RAM/stack 사용량 및 반복 안정성을 실측했다. 8개의 고정 검증 벡터를 12,500회 반복하여 총 **100,000회 physical inference**를 수행했으며, expected decision 대비 **parity failure 0회**를 확인했다. DWT cycle 기준 최소 **132,195 cycles (1.574 ms)**, 평균 **154,103.096 cycles (1.835 ms)**, 관측 최대 **166,086 cycles (1.977 ms)**였다. RAM 계측에서는 static RAM **524 B**, observed stack high-water **368 B**, 합산 관측치 **892 B**였으며 reserved stack 1,024 B 초과는 발생하지 않았다.

단, **1.977 ms는 100,000회 시험에서 관측된 최대값(observed maximum)이며 formal WCET를 의미하지 않는다.** 또한 100,000회 시험은 서로 다른 100,000개 CAN 샘플이 아니라 **8개 고정 검증 벡터에 대한 반복 안정성 시험**이다. 재현 코드와 원본 측정 결과는 `models_c/f5_physical_100k/`에 보존한다.

**F6는 Renode 기반 정량 검증과 실제 하드웨어 검증을 구분한다.** Renode 환경에서 CAN1 FIFO0→RX IRQ→SW queue→V2 feature→Frozen Hybrid와 100 Hz synthetic control task의 동시 동작을 검증했으며, **843.75 fps에서는 3/3 internal lossless**, **847.65625 fps에서는 3/3 SW-queue loss**가 발생했다. 또한 nominal **500 fps 조건에서 60분** 동안 CAN1 수신 프레임 전부 처리, SW overflow **0**, control deadline miss **0**을 확인했다. 다만 이는 Renode 기반 결과이며, **실제 target MCU의 CAN-controller를 이용한 장시간·최대부하 physical validation은 아직 완료되지 않았다.** 따라서 F6의 Renode 결과를 실제 하드웨어의 최대 처리율 또는 장시간 안정성 결과로 주장하지 않는다.

---

## 3. 최종 시스템 아키텍처
CAN 버스 실시간 스트림
- **Mahalanobis V2 최종 입력 피처 (6개, ID-agnostic)**: `freq_in_window`, `unique_ids_in_window`, `entropy`, `mean_byte`, `global_delta_zscore`, `global_value_zscore`
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
| HCRL 내부 zero-day 탐지(Hybrid) | 기존 동일-domain/held-out 평가에서 99.9~100% 수준 |
| ROAD 외부 일반화 — V2 frozen threshold | TPR 12.398709%, FPR 0.079932%, Balanced Accuracy 56.159388% |
| 하이브리드 e2e 지연 | 0.11~0.31ms (PC 기준) |
| Physical MCU Flash / RAM | Flash proxy 247.66 KiB / static RAM 524 B, observed stack high-water 368 B (STM32F446RE, 84 MHz) |
| Physical MCU inference latency / stability | 100,000회 physical inference, parity failure 0회 / 평균 154,103.096 cycles (1.835 ms), 관측 최대 166,086 cycles (1.977 ms) |

---

## 4. 데이터 히스토리 (중요 — 여러 버전이 공존)

| 버전 | 데이터 | 용도 |
|---|---|---|
| `data/` | HCRL 5% 서브셋 | 예선 제출본, 참고용으로만 유지 |
| `data_full/` | HCRL 전체(DoS/Fuzzy/gear/RPM_dataset.csv, 1,657만 행, 실제 타임스탬프) | **F2 이후 모든 실험의 기준 데이터** |
| `road_data/` | ORNL ROAD Dataset(공격/정상 각각) | F3 교차검증 전용 |

**최종 embedded 모델은 `models_c/can_ids_embedded_v2.c`입니다.** Frozen V2는 6개의 CAN-ID-independent feature를 사용하며, `NORMAL_IDX=2`, `CODE_MAP={1,2,0,3,4}`, Mahalanobis threshold `10.104021265036314`를 사용합니다. 기존 can_ids_embedded.c 와 can_ids_embedded_full.c 는 이전 실험 계열로만 유지하며, 현재 Frozen V2 및 F5/F6 검증의 기준 모델이 아닙니다.

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
python3 08_export_v2_to_c.py
# Final embedded model: models_c/can_ids_embedded_v2.c
# Renode F5 benchmark: f5_benchmark_v2/
# Physical MCU F5 100K validation: f5_physical_100k/
```


F5/F6 최종 증거 파일:

- `models_c/f5_v2_final_results.txt` — Renode/pre-hardware F5 결과
- `models_c/f5_physical_100k/f5_physical_100k_results.txt` — 실제 STM32F446RE 100K F5 결과
- `models_c/f6_v2/f6_v2_final_results.txt` — Renode F6 결과

> F5는 실제 NUCLEO-F446RE(STM32F446RE, 84 MHz)에서 100,000회 physical inference 검증을 완료했으며 parity failure 0회, 평균 1.835 ms, 관측 최대 1.977 ms를 확인했다. F6의 실제 CAN-controller 기반 장시간·최대부하 physical validation은 아직 완료되지 않았으며 Renode 결과와 구분한다.

### 5.5 실시간 스트리밍 검증 — ICSim (F1)
ICSim 및 `vcan0` 환경을 준비합니다.

```bash
cd ICSim && make && cd ..
Xvfb :98 -screen 0 1024x768x16 &
export DISPLAY=:98
sudo ip link add dev vcan0 type vcan 2>/dev/null; sudo ip link set up vcan0
cd ICSim && ./icsim vcan0 & ./controls vcan0 & cd ..
```

기존 실시간 탐지 및 공격 주입:

```bash
python3 src/23_icsim_realtime_detect.py 90     # 터미널 1
python3 src/22_icsim_attack_inject.py 90       # 터미널 2 (1~2초 후)

python3 src/27_evaluate_icsim_fixed.py 0x39 0x294 0x143
```

`27_evaluate_icsim_fixed.py`는 기존 ICSim 평가 로직을 보정한 fixed evaluation입니다.

**최종 F1 결과는 탐지기와 독립된 per-frame Ground Truth 체인으로 추가 재검증했습니다.**

```text
ICSim / attack injection
        │
        ├── CAN traffic ──────────> candump
        │
        └── independent GT logger ─> per-frame Ground Truth
                                      │
                                      ▼
                         GT ↔ candump matching
                                      │
                                      ▼
                         prediction ↔ GT evaluation
```

최종 독립 GT 재검증에 사용한 스크립트:

```text
src/42_icsim_gt_logger.py
src/43_match_icsim_ground_truth.py
src/44_evaluate_icsim_independent_gt.py
```

최종 재검증 결과:

- Injected GT frames: **9,485**
- GT → candump matched: **9,485 / 9,485 (100%)**
- Prediction matched: **9,483 / 9,485 (99.9789%)**
- 처리된 공격 프레임 attack recall: **100%**
- End-to-end recall: **99.9789%**
- 별도 normal-only FPR: **0.053538%**

> ⚠️ `src/24_evaluate_icsim.py`는 구버전 평가입니다. `src/27_evaluate_icsim_fixed.py`에서 기존 평가 로직을 보정했으며, **최종 F1 성능 근거는 `42 → 43 → 44`의 독립 per-frame GT 재검증 결과를 기준으로 합니다.**

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

## 6. Scope and Remaining Hardware Limitation

본 문서의 CAN load 및 long-duration 결과는 **Linux SocketCAN/vcan 기반 host-side validation**이며, 실제 MCU의 CAN-controller 최대부하 시험과 구분한다.

한편 F5에서는 실제 **NUCLEO-F446RE (STM32F446RE, Cortex-M4, 84 MHz)**에서 Frozen V2 IDS를 구동하여 **100,000회 physical inference**를 수행했고, parity failure 0회와 추론 지연 및 RAM/stack 사용량을 실측했다.

다만 이 physical F5 시험은 8개의 고정 검증 벡터를 반복 실행한 추론 안정성 시험이며, 실제 CAN-controller를 통한 장시간·최대부하 F6 시험은 아니다.

따라서 현재 결과는 다음을 직접 입증하지 않는다.

- 실제 target MCU CAN-controller의 최대 지속 처리율 및 FIFO overflow immunity
- 실제 CAN-controller 부하 상태에서의 장시간 physical 안정성
- 실제 CAN 통신과 기존 ECU 제어 태스크가 결합된 조건에서의 최대부하 동작

F6의 부하·장시간 동작은 Renode 및 host-side에서 정량 검증했으며, **실제 CAN-controller 기반 장시간·최대부하 physical validation은 아직 완료되지 않았다.** Renode/host-side 결과와 실제 target hardware 결과는 구분하여 보고한다.
---

## 6. 저장소 구조

```text
canids_export/
├── data/                          # 5% 서브셋 (구버전)
├── data_full/                     # HCRL 전체 데이터셋 (F2 이후 기준)
├── road_data/                     # ORNL ROAD Dataset + capture_metadata.json
├── ICSim/                         # zombieCraig ICSim 클론 (F1)
│
├── src/
│   ├── 01~09                      # 예선 제출본(5% 서브셋) 파이프라인
│   ├── 10_build_features_full.py
│   ├── 11_train_full.py
│   ├── 12_ablation_study.py
│   ├── 13_build_features_v2.py
│   ├── 14_train_v2_and_road_test.py
│   ├── 15~17                      # F1 초기 실시간 검증(구버전)
│   ├── 18_train_full_repeated.py
│   ├── 20_road_test_v1_full.py
│   ├── 21_fit_icsim_stats.py
│   ├── 22_icsim_attack_inject.py
│   ├── 23_icsim_realtime_detect.py
│   ├── 24_evaluate_icsim.py       # ⚠ 구버전, 재평가됨
│   ├── 25_icsim_threshold_sweep.py
│   ├── 26_ablation_repeated.py
│   ├── 27_evaluate_icsim_fixed.py # 최종 평가
│   ├── 28_road_test_v2_corrected.py # 최종 ROAD 평가
│   ├── 29_f4_hybrid_split_metrics.py
│   └── 30_f4_hybrid_zeroday_split.py
│
├── models_c/
│   ├── 07_export_to_c.py
│   ├── can_ids_embedded.c
│   ├── can_ids_embedded_v2.c      # Frozen V2 최종 embedded model
│   ├── test_main.c
│   ├── renode_portable/
│   ├── fw_test/                   # F5 ElapsedCycles 측정용
│   └── f5_physical_100k/          # 실제 STM32F446RE 100K 검증
│
└── results/
```

## 6.1 심사용 Offline Dashboard

본선 시연용 Flask dashboard는 인터넷 연결 없이 `localhost`에서 동작한다. 저장된 판정 결과를 단순 표시하는 방식이 아니라, 사전에 기록된 CAN frame을 Frozen V2 pipeline에 입력하여 feature와 prediction을 실행 시점에 계산한다.

- `NORMAL`: ICSim baseline
- `DoS`: known attack
- `FUZZY`: known attack
- `SPOOF`: unknown-style anomaly scenario
- ROAD: Live Demo와 분리된 External Validation evidence

Windows 실행: `run_demo.bat`

실행 전 `dashboard/preflight.py`가 Python package, frozen model, 6-feature structure, Mahalanobis threshold, demo bundle integrity를 검사한다. 실제 본선용 Windows 노트북에서 Wi-Fi를 끈 상태로 브라우저 자동 실행과 4개 replay scenario 동작을 검증하였다.

> Windows LightGBM native loader의 비-ASCII path 문제가 확인되어 심사용 package는 `C:\JARVIS\jarvis_demo`와 같은 ASCII-only path에서 실행하도록 검증하였다.

---

## 7. 알려진 한계 (Known Limitations)

1. **F5 CPU 점유율**: 베어메탈 단일 태스크 환경이라 측정 안 함. RTOS 환경에서 재정의 필요.
2. **F5 timing limitation**: Renode 실행 수치는 instruction-equivalent count이며 실제 MCU의 cycle-accurate WCET 또는 latency로 환산하지 않는다. Physical MCU timing 측정은 pending이다.
3. **F6 physical validation**: Renode에서 60분 concurrent test와 부하 경계시험을 완료했으나, 실제 MCU silicon에서의 CAN-controller behavior 및 장시간 연속운용은 pending이다.
4. **F3 ROAD 일부 공격 유형**: `max_speedometer`/`reverse_light` 계열(값-고정형 스푸핑)은 정밀 재평가해도 탐지율 0% — ID-agnostic 피처의 구조적 한계로 판단됨.
5. **평가 방법론 일반**: 배경 트래픽 밀도가 높은 환경에서는 "시간구간 기준" recall이 아니라 "실제 공격 메시지 기준" recall을 써야 함 — F1/F3 양쪽에서 반복 확인된 교훈.

---

## 8. 팀

자비스(J.A.R.V.I.S) — 국립부경대학교. 노신비(기획/발표), 최혜림(개발/실험), 김준상(개발/발표준비), 이유민(개발/디자인).
