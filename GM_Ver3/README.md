# GM_Ver3

2026 OIBC용 독립 프로젝트입니다. [AGENTS.md](AGENTS.md)에 목표와 개발 규칙을 기록했습니다.
원본 데이터로 실행한 계절·월별 검증과 최종 결과는 [RESULTS.md](RESULTS.md)에 있습니다.

## 검증 절차

- 개별 모델은 2021년부터 각 폴드 마감까지 확보된 발전량과 예보로 학습합니다. 전체 학습 LightGBM은 유효 라벨 전체를 사용하고, 계산비용이 높은 후보는 같은 이력에서 고정 시드로 추출한 표본을 사용합니다. 모델 입력은 제공 메타데이터와 전일 10시·17시 예보 및 해당 예보에서 계산한 특성으로 제한합니다.
- 2021년 12월부터 2024년 11월까지 12개 계절 폴드에서 유효한 발전소별 관측만 평가합니다. 계절 균등 평균과 최악 계절을 함께 반영해 후보 6개를 선정합니다. 잔차 상관관계는 두 발전소가 모두 관측된 시각이 최소 168개인 쌍에서만 계산합니다.
- VPP 비교는 2025년 1~4월 전체 관측 시각을 사용합니다. 스태커는 이전 블록의 기본 모델 OOF를 학습합니다. 그룹 탐색은 각 블록을 대상으로 그보다 더 이른 데이터에서 만든 스태커의 시점외 예측 잔차를 사용합니다. 2024년 12월의 짧은 초기 구간은 사전 정의한 평균으로 시작합니다. 1월 기본 모델 예측은 기존 12월 24일 시작 폴드의 동결 모델을 재사용합니다.
- VPP는 참조·균형·지역·잔차 상관·GA·GA와 이동/교환 탐색을 비교합니다. 새 GA는 시간당 평균 `W`와 최악 운영월의 시간당 `W`를 함께 줄입니다. 이전 달에 편성한 그룹을 이후 달에 그대로 적용해 안정성을 확인합니다. 개별 합산과 직접 예측의 결합, 과거 시점외 잔차 기반 시간대별 보정도 비교합니다.
- 5월 채점 전에 선택 조합, 모델, 그룹, 5월 예측과 제출 예측을 고정합니다. 5월은 한 번 채점하며 5월 실적을 학습·보정·편성에 사용하지 않습니다. 고정한 선택 기록이나 예측 지문이 달라지면 채점과 제출 파일 내보내기를 중단합니다.

## 실행

저장소 루트에서 실행합니다.

```powershell
python -m pip install -r GM_Ver3/requirements.txt
python -m unittest GM_Ver3.test_settlement GM_Ver3.test_validation -v

$Raw = 'C:/path/01_participant_dataset'
$Out = 'C:/path/local_fresh_run'
python -m GM_Ver3.fit --raw $Raw --out $Out --stage development
python -m GM_Ver3.run --raw $Raw --out $Out --stage select
python -m GM_Ver3.seasonal --raw $Raw --out $Out
python -m GM_Ver3.nested --raw $Raw --out $Out --stage select
python -m GM_Ver3.fit --raw $Raw --out $Out --stage may
python -m GM_Ver3.nested --raw $Raw --out $Out --stage freeze
python -m GM_Ver3.nested --raw $Raw --out $Out --stage holdout
python -m GM_Ver3.nested --raw $Raw --out $Out --stage export
python -m GM_Ver3.submission --raw $Raw --submission "$Out/validation_nested/submission"
```

`settlement`는 명시적으로 전달한 발전소 목록, 고정 배정, SMP, 발전소별 실적 및 예측을 읽습니다. SMP 파일의 KST 종료시각을 평가 범위로 사용합니다. 학습 구간 SMP처럼 `k`가 없으면 종료시각 11~15시에 3, 나머지에 1을 적용합니다. 그룹별 부호 있는 오차를 합한 뒤 절댓값을 취해 `0.1 × SMP × k`를 곱합니다. 실적이 없거나 음수인 발전소가 있는 시각은 두 그룹의 평가에서 함께 제외하고 제외 수를 기록합니다. 0 kWh 실적은 유효합니다. `issue_cutoff`는 종료시각별 예측 마감을 계산합니다.

`selection.json`은 2~4월 비교, `seasonal/summary.json`은 계절별 부분 관측 검증, `validation_nested/`는 개선된 선택·5월 평가·제출 파일을 보관합니다. 출력 경로는 저장소와 배포 폴더 바깥에 둡니다. 2021~2024년 부분 관측 결과에는 253개소 VPP 벌금이나 연간 공식 점수를 부여하지 않습니다.

## 다중 모델 전체 아키텍처

기존 실행 폴더와 제출 파일을 보존하면서 새로운 경로에 모델과 결과를 생성합니다. 트리 모델 세 종은 제한된 설정 그리드를 2024년의 시점외 관측으로 평가합니다. GRU·LSTM은 이전에 발행된 기상예보의 3일 또는 7일 요약과 대상일 24시간 예보를 분리해 처리합니다. 발전량 실측은 학습 타깃입니다. 평가기간에는 발전량 lag와 실측 기상 없이 실행합니다. 학습용 과거 발전량 평균을 넣는 트리 ablation은 각 행의 과거만 사용합니다.

```powershell
python -m unittest GM_Ver3.test_settlement GM_Ver3.test_validation GM_Ver3.test_architecture
python -m GM_Ver3.experts --raw $Raw --base $Out --out $Experts --stage all
python -m GM_Ver3.experts --raw $Raw --base $Out --out $Experts --stage ablation
python -m GM_Ver3.architecture --raw $Raw --base $Out --trained $Experts --out $Decision
python -m GM_Ver3.architecture_audit --raw $Raw --base $Out --decision $Decision
python -m GM_Ver3.full_refit --raw $Raw --base $Out --trained $Experts --decisions $Decision --out $FullRefit
python -m GM_Ver3.portfolio_challenge --raw $Raw --base $Out --decision $Decision --out $Portfolio
```

`$Experts`, `$Decision`, `$FullRefit`은 별도 로컬 경로입니다. 첫 두 명령은 후보별·폴드별로 체크포인트를 저장합니다. 같은 설정과 원본으로 중단한 실행을 재개합니다. 설정이나 원본 변경 시 새 출력 폴더를 사용합니다.

OOF 스태킹은 단순 평균, 그룹 절대오차 LP, Ridge, LightGBM·XGBoost 잔차 스태커, 일반 MLP 게이트와 경제 손실 MLP 게이트를 비교합니다. 게이트는 시간·기상에 따라 합이 1인 모델 가중치를 계산합니다. 시간순 내부 검증으로 폭과 학습률, 조기 종료를 고르고 이전 블록 전체로 다시 학습합니다.

VPP 편성은 과거에 이미 만들어 둔 시점외 스태커 잔차를 사용합니다. 참조, 용량균형, 위치 정렬, 잔차 상관, GA와 GA+이동·교환을 비교합니다. 직접 그룹 예측과 합산 예측의 결합 비율도 과거 블록 안에서 시간순 분리한 평가로 고릅니다. 기존 1~4월과 같은 시각에서 벌금 합계로 선택하고 5월은 재검증 결과로 별도 공개합니다. 최종 재학습은 선택한 개별 모델 또는 스태킹 구성원을 첫 평가일 마감까지 확장합니다.

연간 정산금 실험의 모의 실적은 같은 시나리오 안에서 기존 보존 제출본과 새 후보가 공유합니다. 5월 오차를 3일 블록으로 재표본한 가정에 따른 값입니다. 보고서에는 대회 서버 점수를 별도로 표시하며, 실제 점수를 확인하지 않은 경우 빈 값으로 저장합니다.

별도 `portfolio_challenge`는 1~4월에 선택한 예측 스태커의 발전소별 예측을 고정합니다. 평균 벌금 또는 평균과 최악 월을 사용하는 GA·이동 교환을 같은 예측에서 비교합니다. 5월을 이전 설정 변경에 사용한 이력이 있으므로 회귀 진단으로 봅니다.
