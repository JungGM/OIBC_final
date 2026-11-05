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
