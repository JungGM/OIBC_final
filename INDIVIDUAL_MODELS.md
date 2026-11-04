# 개별 발전량 예측모델

2026 OIBC 배포 데이터의 2022~2023년 실적과 전날 10시·17시 예보로 학습한다.
2023년 1~6월 두 구간에서 Optuna TPE로 모델별 MAE를 탐색한다.
2023년 7~9월, 10~12월은 별도 외부 Walk-forward OOF다.
각 검증 구간의 학습 컷오프는 첫 대상일의 전날 17시 KST다.
모델별 상위 3개를 외부 OOF MAE로 정하고, 2023-12-31 17시까지 재학습한다.
2024-01-01 01시~2025-06-01 00시는 후보를 고정한 뒤에만 로컬 테스트로 평가한다.

배포 데이터 실적과 새 특징 캐시 `features_v2_20261104`를 사용한다.
캐시 manifest의 원본 경로와 특징 목록을 읽고 원본 파일 해시를 검증한다.

```powershell
uv venv --python 3.12 C:/Users/jgm20/OIBC_local/env_individual
uv pip install --python C:/Users/jgm20/OIBC_local/env_individual/Scripts/python.exe -r requirements-individual.txt
& C:/Users/jgm20/OIBC_local/env_individual/Scripts/python.exe train_individual.py --data <배포데이터폴더> --cache C:/Users/jgm20/OIBC_local/features_v2_20261104 --out C:/Users/jgm20/OIBC_local/individual_features_v2_20261104 --family all --trials 4
```

버전 폴더마다 `model.joblib` 또는 `model.pt`, `config.json`,
`metrics.json`, `oof.csv`를 저장한다. `top3.csv`는 집계 비교표다.
버전은 기존 숫자 다음부터 추가된다. 개별 예측과 모델은 로컬 폴더에만 보관한다.

개별 지표는 실적이 있는 발전소·시간의 MAE, RMSE, MBE, 용량정규화 nMAE다.
W와 S는 현재 그룹 A/B가 모두 20MW 이상이며 253개 발전소 실적이
완전한 시각에만 배포 README 식으로 계산한다.
W는 0.10 × Σ SMP × k × 집합 절대오차, S는 R−W다.
2023년 OOF에서 완전한 시각이 없으면 W/S는 null이다.
로컬 테스트의 완전한 시각 수와 평가 행 수를 함께 확인해야 한다.
