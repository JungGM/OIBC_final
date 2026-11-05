# OIBC T3 + GRU
고정 XGBoost T3 80%와 GRU 20%를 발전소별 kWh 예측 단계에서 혼합합니다.

## 재현
Python 3.12, CPU 4 threads로 실행했습니다. 원본 데이터는 별도로 준비합니다.
```sh
python -m venv .venv
# 환경을 활성화한 뒤
python -m pip install -r requirements.txt
python pipeline.py --data-root /path/to/oibc_data --output-dir 제출
python validate_submission.py --data-root /path/to/oibc_data --submission-dir 제출
```
프로그램은 비어 있는 출력 폴더만 허용하며 모델을 메모리에서 학습합니다.
필요한 원본은 sites.csv, smp.csv, train/generation.parquet, train/smp.csv,
train/forecast_wa.parquet, train/forecast_wb.parquet,
test/forecast_wa.parquet, test/forecast_wb.parquet입니다.
인터넷 연결은 예측에 필요하지 않습니다. 대용량 특징 결합을 위해 충분한 RAM이 필요합니다.

## 모델
T3는 89개 특징, 2021~2024년 달력 연도 학습, 시드 42, 400트리입니다.
GRU는 발전소별 우측닫힘 하루 24시간 예보 시퀀스, 1층, 은닉 크기 32,
AdamW(lr=0.001, weight_decay=0.001), 8 epoch, batch 256, gradient clipping 1을 사용합니다.
완전한 24시간 실적 시퀀스 중 모든 시각이 2024년 안에 있는 것만 GRU 학습에 사용합니다.
결측은 학습 자료 평균으로 대체하고 표준화합니다. 평균 대체값은 표준화 후 0이며
표준화 값은 [-20,20]으로 제한합니다. T3에는 NaN을 그대로 전달합니다.
GRU 출력/용량은 [0,1]로 제한합니다. T3는 음수만 0으로 제한합니다.
시작·중앙·끝 태양고도가 모두 -1도 이하인 1시간 구간은 두 모델 모두 0으로 처리합니다.
GRU 시드 42를 사용합니다. 향후 다른 시드를 선택하는 튜닝은 포함하지 않습니다.

## 검증
2021~2023년 학습→2024년 부분관측 검증에서 80:20은 다섯 GRU 시드 모두 개선됐습니다.
평균 벌금 감소율 0.882%, 상반기 1.618%, 하반기 0.368%입니다.
2025년 1~5월 253개소 완전관측 3,624시간에서 경계 수정 후 시드42의
T3 벌금 73,426,245원, 80:20 벌금 71,384,437원(2.781% 감소)이었습니다.
2025년은 반복 사용된 평가기간입니다. 이 프로젝트는 사용자가 요청한 개선 후보의 재현 패키지이며
미사용 테스트에서 성능이 확정됐다는 의미는 없습니다. 자세한 결과는 RESULTS.md를 확인하세요.

## 파일
features.py는 기존 89개 특징, train_gru.py는 GRU 학습, predict.py는 혼합 예측,
evaluate.py는 그룹 벌금과 발전소 오차, pipeline.py는 전체 재현,
validate_submission.py는 제출 형식을 처리합니다.
결과 CSV는 제출/에 로컬 저장하며 .gitignore에서 제외합니다.
원본 데이터, 모델 가중치, 캐시도 Git 추적 대상에서 제외합니다.
공개 업로드 전 git status로 CSV와 원본 자료가 포함되지 않았는지 확인하세요.

