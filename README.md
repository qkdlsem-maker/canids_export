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
| F1 | ICSim·can-utils 실시간 주입, 탐지율/오탐률/지연 정량화, Peak RAM/CPU | 실제 ICSim + vcan0 3회 반복 검증. DoS 98.3%, Fuzzy 100%, 스푸핑 100%/100%, FPR 0.05%, Peak RSS 65.6~65.8MB | ✅ 완료 |
| F2 | 5%서브셋→전체 데이터 재현성 검증 | HCRL 전체(1,657만 행) 재학습, 시드 3회 반복(F1=1.0000±0, FPR=0.0192%±0) | ✅ 완료 |
| F3 | CAN ID 의존 낮은 feature set 재설계 | ID-agnostic 6피처 설계, V1/V2 모두 전체데이터 기준 ROAD 공정비교(V1 FPR 100%실패 vs V2 FPR 0.07%) | ✅ 완료 |
| F4 | LightGBM단독 vs Mahalanobis단독 vs Hybrid 비교 | 4축 비교 + 시드 3회 반복 + 탐지/유형정확 지표 분리(zero-day RPM: LGBM 0% vs Hybrid 100%) | ✅ 완료 |
| F5 | 실제 MCU Flash/RAM/CPU/WCET/탐지지연 | ARM Cortex-M4 크로스컴파일 + Renode 에뮬레이션. Flash 254KB, 정적RAM 1.7KB, 5클래스 WCET 0.31~0.83ms(명령어 수 기반 추정) | 🟡 부분완료 |
| F6 | 기존 ECU 기능과 동시 동작(장시간·최대부하) | SysTick 인터럽트 동시성 검증(정적분석+300회 반복, 인터럽트 비활성화 명령 0건) | 🟡 부분완료 |
| F7 | (선행 결함) C 변환기 정상 클래스 인덱스 오류 | NORMAL_IDX/CODE_MAP 도입, x86+ARM 회귀테스트로 검증 | ✅ 완료 |

**F5/F6 남은 한계**: CPU 점유율(베어메탈 단일태스크라 개념 재정의 필요), 실제 실리콘 대신 명령어 수 기반 시간 추정(Renode 사이클카운터 미지원), 장시간(수시간)/최대버스부하/FIFO오버런 카운터 미검증.

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
| MCU WCET(5클래스, 168MHz 가정) | 0.31~0.83ms (명령어 수 기반 추정) |

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
python3 src/20_road_test_v1_full.py            # V1도 전체데이터로 재학습 후 ROAD 재비교
python3 src/28_road_test_v2_corrected.py       # 메타데이터 기반 정밀 재평가(권장, 최종 수치)
```
⚠️ `14_train_v2_and_road_test.py`의 ROAD recall은 "공격 파일 전체"를 분모로 삼아 저평가되어 있습니다.
**최종 수치는 반드시 `28_road_test_v2_corrected.py` 결과를 사용하세요.**

### 5.4 MCU 검증 (F5/F6)
```bash
cd models_c
python3 07_export_to_c.py                  # data_full 기준 C 코드 생성
cp can_ids_embedded_full.c fw_test/can_ids_embedded.c
cd fw_test
arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfloat-abi=soft -O2 -nostartfiles \
  -T linker.ld -o can_ids_fw.elf startup.c main.c -lm --specs=nosys.specs
cd ../renode_portable
./renode --console -e "include @../run_ids3.resc"
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
│ ├── 15~17 # F1: 초기 실시간 검증(합성 트래픽, 구버전)
│ ├── 18_train_full_repeated.py
│ ├── 20_road_test_v1_full.py
│ ├── 21_fit_icsim_stats.py
│ ├── 22_icsim_attack_inject.py
│ ├── 23_icsim_realtime_detect.py
│ ├── 24_evaluate_icsim.py # ⚠️ 구버전, 저평가됨
│ ├── 25_icsim_threshold_sweep.py # ⚠️ 불필요했음, 참고용
│ ├── 26_ablation_repeated.py
│ ├── 27_evaluate_icsim_fixed.py # ✅ 최종 평가
│ ├── 28_road_test_v2_corrected.py # ✅ 최종 ROAD 평가
│ ├── 29_f4_hybrid_split_metrics.py
│ └── 30_f4_hybrid_zeroday_split.py
├── models_c/
│ ├── 07_export_to_c.py # C 변환기 (F7 버그 수정 완료)
│ ├── can_ids_embedded.c # 서브셋 모델 (구버전)
│ ├── can_ids_embedded_full.c # 전체데이터 모델 (최종)
│ ├── test_main*.c
│ ├── renode_portable/
│ └── fw_test/


---

## 7. 알려진 한계 (Known Limitations)

1. **F5 CPU 점유율**: 베어메탈 단일 태스크 환경이라 측정 안 함. RTOS 환경에서 재정의 필요.
2. **F5 WCET**: Renode가 사이클 카운터를 지원하지 않아 명령어 수×CPI 가정으로 추정한 값. 실리콘 실측 아님.
3. **F6 장시간/최대부하**: 수 초 단위 반복 테스트만 수행. 수 시간 연속 운용, 최대 버스 부하, FIFO 오버런 카운터는 미검증.
4. **F3 ROAD 일부 공격 유형**: `max_speedometer`/`reverse_light` 계열(값-고정형 스푸핑)은 정밀 재평가해도 탐지율 0% — 평가 방법론 문제가 아니라 ID-agnostic 피처의 구조적 한계로 판단됨.
5. **평가 방법론 일반**: 배경 트래픽 밀도가 높은 환경에서는 "시간구간 기준" recall이 아니라 "실제 공격 메시지 기준" recall을 써야 함 — F1/F3 양쪽에서 반복 확인된 교훈.

---

## 8. 팀

자비스(J.A.R.V.I.S) — 국립부경대학교. 노신비(기획/발표), 최혜림(개발/실험), 김준상(개발/발표준비), 이유민(개발/디자인).
지도교수: 김태국 (futurenetworklab).
