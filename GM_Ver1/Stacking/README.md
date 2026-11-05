# Ridge 스태킹 Ver1

6개 base 모델의 시간 외 예측으로 학습한 선형 메타모델이다.

2024년 1월 2일~9월 30일 17시로 적합하고 10월 1일~12월 31일 17시 MAE로 정규화 강도 alpha를 선택했다. 해당 값으로 2024년 마감 이전 실적을 다시 학습해 2025년 1~5월을 평가했다. 2025년 점수로 설정을 바꾸지 않았다.

| 모델 | 2025.1~5 MAE(kWh) | 참조 그룹 벌금(원) |
|---|---:|---:|
| XGBoost/Ver1 | 5.8030 | 78,126,128 |
| XGBoost/Ver2 | 5.8333 | 78,268,947 |
| GRU/Ver1 | 5.4885 | 72,118,409 |
| GRU/Ver2 | 5.6753 | 71,322,817 |
| TCN/Ver1 | 5.7675 | 91,326,177 |
| TCN/Ver2 | 5.7798 | 79,674,809 |
| uniform_mean_6 | 5.3442 | 71,477,301 |
| Ridge_stacking | 5.4327 | 71,830,743 |

선택 alpha: 0.001. 최종 메타모델 학습은 2,795,734개 유효 시간 외 예측 행을 사용했다. Base 모델은 전체 유효 학습 이력을 사용하지만, 메타모델에는 시간 외 예측이 있는 2024~2025년 행만 넣는다.

최종 메타모델은 2024~2025년의 시간 외 예측과 첫 평가일 마감 이전 정답으로 재학습했다. 이를 6개 최종 base 모델의 평가 기간 예측에 적용했다. 최종 파일은 `Ver1/final/model.json`과 `Ver1/final/predictions.parquet`다.

계수는 음수도 가능한 Ridge 회귀계수이며 합이 1인 구성비가 아니다. 발전소 용량으로 정규화한 6개 예측에 적용하고, kWh로 복원한 뒤 음수만 0으로 자른다. 정규화 통계는 메타 학습 fold에서만 적합한다.

참조 그룹은 고정하며 2025년 벌금은 실적이 완전한 3,624시간만 계산한다. 2024~2025년은 과거 연구에서 확인한 기간으로 연구 전체의 미사용 테스트가 아니다. 평가 기간 정답은 없어 최종 예측의 성능은 알 수 없다. 별도의 경사하강법 가중치 최적화와 VPP 구성 최적화는 이 단계에 포함하지 않았다.

실행: `python -m GM_Ver1.stacking --out <기존 로컬 결과 폴더>`

[Ridge 공식 설명](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) · [Stacking의 학습 예측 주의사항](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
