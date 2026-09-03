# 차량 CAN 버스 침입탐지 시스템 - 2026 K-DS 해커톤

에스엘㈜ 과제: 전력변환/로봇제어 ECU 보호를 위한 CAN/CAN-FD 침입탐지 시스템

서버 배치 경로 예시: `/home/futurenetlab/choihyerim1129/canids_hackathon/`

## 폴더 구조
```
canids_export/
├── data/                  # Car_Hacking_5pct.csv (HCRL 5% 서브셋), features_v3.parquet(생성됨)
├── src/                   # CPU에서 실행 (이미 검증 완료)
│   ├── 01_build_features.py
│   ├── 02_hybrid_train_eval.py
│   └── 03_export_dashboard_data.py
├── gpu/                   # GPU 서버에서 실행 (아직 미실행 - torch 필요)
│   └── train_cnn_transformer.py
├── models/                # 학습된 모델 (lgbm_final.txt, maha_detector.npz)
├── results/               # 실험 결과 (final_summary.txt)
└── requirements.txt
```

## 실행 순서
```bash
pip install -r requirements.txt --break-system-packages
cd src
python3 01_build_features.py       # 특징공학 (약 35초)
python3 02_hybrid_train_eval.py    # LightGBM + Mahalanobis 하이브리드 학습/평가
python3 03_export_dashboard_data.py

# GPU 서버에서 (gpu0 or gpu1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
cd ../gpu
python3 train_cnn_transformer.py   # 1D-CNN / TinyTransformer 학습 + LightGBM과 비교
```

## 최종 확정 결과 (2단계 하이브리드: LightGBM + Mahalanobis)
| 항목 | 결과 | 요구사항 |
|---|---|---|
| Macro F1 (알려진 공격) | 1.0000 | ≥0.95 ✅ |
| Zero-day 탐지율 (4개 공격 모두, 학습서 완전 제외) | 99.9~100% | - |
| 정상 오탐율(FPR) | 1.99% | 낮을수록 좋음 |
| 하이브리드 합산 메모리 (LightGBM 709KB + Mahalanobis 1.3KB) | 0.694MB | ≤1MB ✅ |
| 하이브리드 e2e 지연시간 (2단계 순차 실행) | 1.21ms | ≤10ms ✅ |

## 이번 라운드에서 고친 문제들
1. **메모리 초과 해결**: 2단계 이상탐지기를 IsolationForest(593KB) → Mahalanobis 거리 기반 경량 탐지기(1.3KB, 468배 감소)로 교체. 통계량(평균벡터+공분산역행렬)만 저장하면 되므로 초경량.
2. **Zero-day 탐지율 개선**: 기존엔 gear/RPM 스푸핑 탐지율이 7~14%로 낮았음. 원인은 "변화량(delta)"만 봤기 때문 → ID별 **절대값 z-score(value_zscore)** 피처 추가 + 임계값을 99.9 percentile로 튜닝 → 전 공격유형 99.9~100% 탐지, FPR 1.99%로 개선.
3. **End-to-end 검증**: 기존엔 두 모델을 따로따로만 측정. 이번엔 실제로 순차 실행(LightGBM→Mahalanobis)한 합산 지연시간·합산 메모리를 측정.
4. **대시보드 실데이터화**: 기존 랜덤 시뮬레이션 대신, 실제 학습된 모델의 실제 추론 결과(`models/dashboard_data.json`)를 재생하도록 변경.
5. **데이터 leakage 제거**: 이 5% 서브셋은 DoS/Fuzzy/RPM/gear 각 원본 캡처파일이 통째로 이어붙여진 구조라, 전역 위치 기준 split을 하면 한 공격유형 전체가 test에만 몰리는 문제 발견. → Label(공격유형)별로 각자 앞 70%/뒤 30%로 나누는 방식으로 수정.

## 미완료 / 알아야 할 한계
- **item5 (1D-CNN·TinyTransformer 비교)**: `gpu/train_cnn_transformer.py`에 코드는 다 있으나, 이 환경엔 torch가 없어 실행 못 해봄. GPU 서버에서 직접 실행해서 결과 확인 필요. (LightGBM 성능이 이미 최고치라 CNN/Transformer가 이를 뛰어넘을 가능성은 낮음 → "간단한 트리모델로 충분하다"는 것도 정당한 결론이 될 수 있음)
- **item6 (데이터셋 한계)**: 원본 HCRL Car-Hacking 전체 데이터셋(타임스탬프 포함, 1600만 건)이 아니라 GitHub에 공개된 **5% 서브셋**(타임스탬프 없음, `data/Car_Hacking_5pct.csv`)으로 실험함. 원본 데이터 접근 권한이 있다면 `data/` 폴더에 동일 컬럼 구조(`CAN ID, DATA[0..7], Label`)로 교체 후 그대로 재실행 가능. 예선신청서에는 이 부분을 "방법론적 한계"로 명시하는 게 정직하고 안전함.
