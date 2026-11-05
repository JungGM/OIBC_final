# 예측 모델 후보: XGBoost / GRU / TCN

각 모델의 하이퍼파라미터 탐색, 두 후보 선정, 시간순 검증, 전체 이력 재학습과 발전소별 예측을 생성한다. `GM_Ver1.stacking`으로 Ridge 스태킹, `GM_Ver1.gradient_descent`로 대회 그룹 벌금 기반 가중치 최적화를 수행한다. VPP 구성 최적화는 후속 단계다. GRU/TCN 자체 가중치 학습에는 AdamW를 사용한다.

## 시간과 데이터

- 입력: 고정 학습 데이터 `fixed_training_v1`의 기본 78열 + 발전소 코드 1열 + 태양 위치 6열, 총 85열. 모든 trial에서 고정한다. 추가 특성의 개별 기여도를 검증한 ablation은 아니다.
- 전일 17시 발행 예보와 제공 메타데이터만 사용한다. 실측 기상·SMP·k·현재 그룹·타깃 파생치는 입력에 없다. 신경망은 무작위 초기화하며 외부 사전학습 가중치를 쓰지 않는다.
- GRU/TCN의 한 시퀀스는 한 발전소의 한 운영일 24시간이다. 00시는 전날 운영일 마지막 구간이다. 같은 시퀀스의 입력 예보는 모두 동일한 전일 마감에 알려진 값이다.
- 실제 0 타깃은 학습한다. 결측·음수 실적은 손실 마스크로 제외하며 결측을 실적으로 0 대체하지 않는다. 손실 계산을 위한 0 placeholder는 반드시 마스크와 함께 사용한다. 검증 정답은 원본 float64를 독립 재조회하고 음수 관측도 보존한다.
- 학습용 수치 연산은 float32다. GRU/TCN 결측 대체 평균·표준편차는 각 학습 fold의 유효 실적 행에서만 적합한다. 표준화 후 ±10으로 자르고 결측은 표준화된 평균 0으로 둔다.
- 최종 제출 모델은 **2021-01-01 01:00~2025-05-31 17:00 KST의 모든 유효 실적**으로 재학습한다. 첫 평가일 마감 후인 2025-05-31 18:00~2025-06-01 00:00 실적은 제외한다. 실제 수신시각 메타데이터는 없어 타깃 시각 기준 embargo만 보장한다.

## 후보 선택과 확인

1. 튜닝 학습: 2021년부터 2022-12-31 17시까지. CPU 탐색에서는 XGBoost 최대 500,000개 유효 행, GRU/TCN 최대 20,000개 유효 시퀀스를 고정 난수로 표본 추출한다.
2. 튜닝 검증: 2023년 운영일에서 고정 추출한 공통 12,000개 발전소·일 시퀀스의 관측·비음수 타깃. 지표는 **발전소별 원래 kWh 단위 MAE**다. 모든 모델이 동일한 검증 키를 쓴다. 2023년에는 253개소 실적이 완전한 시간이 없어 대회 그룹 벌금을 직접 최소화했다고 주장하지 않는다.
3. Optuna TPE로 XGBoost 10회, GRU 8회, TCN 8회를 탐색한다. 유한한 탐색 범위의 상위 두 설정을 Ver1·Ver2로 고정한다. 전역 최적해나 대회 우승 성능을 뜻하지 않는다. XGBoost는 최대 600 rounds·patience 45, 신경망은 최대 16 epochs·patience 4다. XGBoost의 선택 round는 재학습에서 고정한다. 신경망은 이력이 크게 늘 때 갱신 횟수를 비슷하게 유지하도록 `min(선택 epoch, max(3, ceil(선택 epoch×20000/전체 학습 시퀀스 수)))`번 **전체 데이터**를 반복한다. 이 규칙은 확인 점수를 보기 전에 고정했으며 실제 epoch는 `fit_summary.json`에 기록한다.
4. 튜닝의 마지막 타깃은 2024-01-01 00시이므로, 확인에 사용할 2024년 예측은 **1월 2일부터**다. 1월 1일 운영일은 점수와 스태킹 학습에서 제외한다. 해당 base 모델의 적합 타깃은 2023-12-31 17시까지로 보수적으로 제한한다. 2025년 1~5월 확인용 모델은 2024-12-31 17시까지 **각각 전체 유효 이력**으로 재학습한다. 확인 결과로 후보를 다시 고르지 않는다. 학습 경계가 다른 두 기간의 예측을 섞어 하나의 동일 모델 실적으로 해석하지 않는다.
5. 최종 모델은 위 최종 마감까지 다시 전체 이력으로 학습하고 평가 입력 253개소 × 8,760시간 = 2,216,280행을 예측한다. 평가 기간 정답이 없어 이 예측의 성능은 알 수 없다.

2024~2025년은 과거 연구에서 이미 확인한 기간이다. 이번 탐색의 선택과는 분리하지만 연구 전체에서 한 번도 보지 않은 테스트라고 부르지 않는다. 참조 그룹 벌금은 0.1 × Σ SMP × k × |그룹 예측합−그룹 실적합|이다. 전체 253개소 정답이 있는 시간만 계산한다. 2024년의 178시간 점수는 연간 대표 성능이나 공식 채점이 아니다.

## 모델

- **XGBoost:** 용량 정규화 타깃을 회귀하고 용량을 다시 곱한다. 목적함수, 깊이, 학습률, 표본·열 추출률, 정규화와 용량 가중 지수를 탐색한다. 음수 예측은 0으로 자른다. [공식 매개변수 문서](https://xgboost.readthedocs.io/en/stable/parameter.html).
- **GRU:** 발전소 임베딩과 24시간 수치 입력을 GRU에 넣어 시간별 발전 이용률을 출력한다. 은닉 크기·층 수·단방향/양방향·임베딩·dropout·학습률·규제·손실을 탐색한다. 양방향도 같은 마감에 알려진 예보만 참조한다. 출력은 softplus로 비음수를 보장한다. [PyTorch GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html).
- **TCN:** 왼쪽 패딩만 사용하는 팽창 인과 합성곱 두 개와 residual 연결을 각 block에 둔다. dilation은 1,2,4,(8)이다. 채널·층 수·kernel·임베딩·dropout·학습률·규제·손실을 탐색한다. 시퀀스 밖에서 숨은 상태를 전달하지 않는다. [TCN 원 논문](https://arxiv.org/abs/1803.01271).

## 실행

저장소 루트에서 실행한다. 출력은 Git·OneDrive 밖이어야 한다. 동일 출력 폴더를 다시 지정하면 완료된 단계를 재사용한다. 설정·데이터를 바꿀 때는 새 출력 폴더를 사용한다.

```powershell
python -m pip install -r GM_Ver1/requirements.txt
python -m unittest GM_Ver1.test_models train.test_dataset -v
python -m GM_Ver1.run --out C:\OIBC_local\OIBC_final\forecast_run
# 모델별 실행도 가능: --family XGBoost / GRU / TCN
python -m GM_Ver1.XGBoost.Ver1.model --artifact-root C:\OIBC_local\OIBC_final\forecast_run --output C:\OIBC_local\OIBC_final\forecast_run\reproduced.parquet
# Git에 들어 있는 가중치를 바로 사용할 때:
python -m GM_Ver1.XGBoost.Ver1.model --artifact-root GM_Ver1 --dataset C:\OIBC_local\fixed_training_v1 --output C:\OIBC_local\reproduced.parquet
```

이번 채팅의 소스·후보 설정·탐색 기록·집계 결과·학습 가중치·예측 파일은 `GM_Ver1/`에 파일 단위로 보관한다. 실제 학습 데이터와 재생성 가능한 입력 캐시는 업로드하지 않는다. 각 `Family/VerN/`에는 `config.json`, `metrics.json`, `confirm_2024/`, `check_2025/`, `final/`이 있다. 각 stage는 학습 모델, 학습 메타데이터와 전체 발전소별 예측 Parquet를 가진다. GRU/TCN에는 scaler와 epoch 이력도 있다. `final/`만 대회 평가 기간용 전체 이력 재학습 모델이다. `run_status.json`에서 완료·진행·대기 상태를 확인할 수 있다.

스태킹은 `confirm_2024/predictions.parquet`, `check_2025/predictions.parquet`의 시간 외 예측으로 적합한다. 2024년 1월 1일은 하이퍼파라미터 선택 가용 시점 때문에 제외한다. 학습 데이터에 대한 final 모델의 적합값을 OOF로 사용하면 안 된다. 실행은 `python -m GM_Ver1.stacking --out <기존 로컬 결과 폴더>`이며 결과는 `Stacking/`에 저장한다.
