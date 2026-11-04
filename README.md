# OIBC 고정 학습 데이터 v1

대회 제공 원본에서 모델 학습에 사용할 실제 Parquet를 생성한다.
학습 데이터 기준은 `dataset_lock.json`의 SHA256으로 고정한다.
원본·학습 행·평가 입력·좌표·개별 발전소 결과·모델은 로컬에만 보관한다.
이 저장소에는 재현 코드, 특성 사전, 시간 분할 및 집계 검증 결과만 둔다.

## 생성 파일과 사용법

- `training_ready.parquet`: 결측·음수 타깃을 제외한 **4,392,526행**, 타깃 `energy_kwh`(kWh). 실제 0 발전량 2,108,576행은 유지한다.
- `training_data.parquet`: 2021-01-01 01:00~2025-06-01 00:00 KST, 253개소 전체 격자 **9,788,064행**. 미관측 5,395,498행과 음수 40행을 원본 그대로 보존한다.
- `evaluation_input.parquet`: 2025-06-01 01:00~2026-06-01 00:00 KST, **2,216,280행**. 실적 타깃은 전부 결측이다.
- 후보 입력 270개 중 초기 기본 입력은 78개다. 특성별 수식·단위·가용성·결측 규칙은 [특성 사전](feature_dictionary.csv)을 따른다. 270개는 추가 실험 후보를 포함하며 성능 우수성이 검증된 최종 선택은 아니다.
- SMP와 `k`, 참조 그룹은 `training_weights.parquet`, `evaluation_weights.parquet`, `reference_composition.parquet`로 분리한다.

모든 시간은 KST, 1시간, 우측닫힘이다. `00:00`는 전날 운영일의 마지막 시간이다.
전일 17:00 cutoff는 `(ts - 1h).normalize() - 7h`로 계산한다.
`energy_per_capacity`, `target_*`, `issue_cutoff`, 키 등 메타 열을 모델 X에 임의로 넣지 않는다.

```python
from dataset import current_dataset, iter_fold

directory = current_dataset()  # 이 PC의 로컬 파일 위치
for X, y, keys in iter_fold(directory, "tune_2023", "train"):
    # X: manifest.default_features 78개, y: 원본 kWh
    # 결측 대체/스케일링/선택은 이 학습 fold 안에서만 fit
    pass
```

다른 PC에서는 `current_dataset()` 대신 아래 생성 결과의 로컬 디렉터리를 지정한다.
270개 전체 후보는 manifest의 `features` 목록을 `features=`로 명시한다.
한 번에 전체 270열을 메모리에 올리지 않도록 배치 로더를 제공한다.

## 재현

Python 3.12와 [고정 패키지 버전](requirements.txt)을 사용한다.
배포 폴더 안에 `README.md`, `sites.csv`, `smp.csv`, `train/`, `test/`가 있어야 한다.
이전 캐시나 모델의 특성 파일에 의존하지 않고 원본에서 다시 생성한다.
출력은 Git·OneDrive·원본 폴더 밖의 새 로컬 폴더여야 한다.

```powershell
python -m pip install -r requirements.txt
python audit_data.py --data <배포폴더> --out <새로컬결과폴더>
python build_dataset.py --data <배포폴더> --out <새로컬결과폴더>
python verify_dataset.py --data <배포폴더> --out <새로컬결과폴더>
python freeze_dataset.py --out <새로컬결과폴더> --publish-code .
python -m unittest test_dataset -v
```

`freeze_dataset.py`는 전체 격자가 독립 검사를 통과한 뒤에만 학습 가능 행을 추출한다.
원본·전체 격자·학습용 파일 해시와 키·타깃 지문을 기록한다. 기존 완성 파일은 덮어쓰지 않는다.

## 특성과 누수 경계

round17 A/B 예보 30열, 풍향 sin/cos, 입력 결측 표시자, 시각·계절·요일·주말·용량을 기본 입력으로 쓴다.
추가 후보에는 10→17시 예보 수정량, 대상일 안의 시간 변화·하루 요약,
A/B 비교, 비식별 좌표를 사용한 근사 태양 위치, 강설·결빙 지표가 있다.
대상일의 이후 시각 예보도 전일 발행된 값만 사용하며, 차분은 정확한 1시간 인접·같은 운영일 안에서만 계산한다.
요약값에는 제공된 시간 수를 함께 기록하며 입력 결측을 비·눈 없음으로 채우지 않는다.

실현 SMP·`k`·참조 배정·당시 실측 기상·발전량 시차는 X에 넣지 않는다.
발전량/실측 수신시각이 없어 과거 발전량 시차와 실측 기반 예보 품질 특성 K는 보류한다.
외부 휴일 달력, 외부 기상·지도·사전학습 모델은 사용하지 않는다.
비식별 좌표의 복원·식별은 수행하지 않는다.

## 검증과 정산

[시간 분할](validation_splits.csv)은 2022·2023년 선택, 2024년 확인, 2025년 1~5월 최종 보류 구간이다.
각 fold의 학습 타깃은 첫 검증 운영일의 **전일 17시 이하**로 제한한다.
이는 타깃 시각 embargo이며 실제 수신시각까지 입증하지는 않는다.
이번 작업은 학습 파일 생성·구조 검증이다. 신규 모델 적합이나 특성 선택에 이 구간을 사용하지 않았다.
과거 저장소 실험에서는 2024~2025년 결과를 이미 확인했으므로, 연구 전체에서 한 번도 보지 않은 외부 테스트로 주장할 수 없다.

원본 배포 README의 벌금 계수를 적용한다.

`벌금(원) = 0.1 × Σ_(그룹,t) SMP(t) × k(t) × |그룹 예측합(t) − 그룹 실적합(t)|`

`정산금(원) = Σ_(그룹,t) SMP(t) × 그룹 실적합(t) − 벌금(원)`

`k = 3`은 **제출 ts의 시가 11~15**인 시간이고, 그 외는 1이다.
0.1이 없는 가중 절대오차와 원 단위 벌금을 구별한다.
`dataset.score_predictions`는 그룹 합산 뒤 절대오차를 계산하고 계수를 한 번만 적용한다.
253개소 실적이 모두 있는 시간은 3,802시간뿐이다(2024년 178, 2025년 1~5월 3,624).
누락 실적을 0으로 채우거나 부분 관측 점수를 연간·공식 성적으로 주장하지 않는다.

[검증 결과](DATASET_REPORT.md)와 [실험 상태](feature_ablation.csv)를 참조한다.
특성별 손실 개선·악화, SHAP/중요도, ablation은 아직 실행하지 않았다.
이전 지시문의 전체 요구사항과 후속 실험 범위는 [DATASET_SPEC.md](DATASET_SPEC.md)에 보관한다.
