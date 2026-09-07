# Reviewer Feedback Robustness Validation

## 1. Independent ICSim Ground Truth

ICSim 실시간 평가의 ground truth가 IDS 내부 feature 또는 prediction에 의존하지 않도록 독립적인 per-frame injection ground-truth chain을 구축하였다.

관련 스크립트:

- `src/42_icsim_gt_logger.py`
- `src/43_match_icsim_ground_truth.py`
- `src/44_evaluate_icsim_independent_gt.py`

### Results

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

## 2. Host-Side Sustained CAN Load

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

## 3. One-Hour Host-Side Stability

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

## 4. V2 Mahalanobis Threshold Sensitivity

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

## 5. Final Model Status

최종 모델은 계속 V2 ID-agnostic Mahalanobis이다.

V3 temporal 및 V4 byte-temporal 실험은 post-hoc ablation이며 최종 모델 교체 근거로 사용하지 않는다.

---

## 6. Scope and Remaining Hardware Limitation

본 문서의 CAN load 및 long-duration 결과는 Linux SocketCAN/vcan 기반 host-side validation이다.

`vcan`은 실제 CAN arbitration 및 physical bit timing을 재현하지 않는다.

따라서 본 결과는 다음을 직접 입증하지 않는다.

- 실제 target MCU CAN-controller hardware FIFO overflow immunity
- 실제 MCU silicon에서의 장시간 안정성
- 실제 MCU에서의 CPU/resource behavior

Renode 및 host-side 결과는 실제 target hardware validation과 구분하여 보고한다.
