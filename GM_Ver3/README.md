# GM_Ver3

2026 OIBC용 독립 프로젝트입니다. [AGENTS.md](AGENTS.md)에 목표와 개발 규칙을 기록했습니다.

## 실행

저장소 루트에서 실행합니다.

```powershell
python -m pip install -r GM_Ver3/requirements.txt
python -m unittest GM_Ver3.test_settlement -v
python -m GM_Ver3.settlement --help
```

`settlement`는 명시적으로 전달한 발전소 목록, 고정 배정, SMP, 발전소별 실적 및 예측을 읽습니다. SMP 파일의 KST 종료시각을 평가 범위로 사용합니다. 학습 구간 SMP처럼 `k`가 없으면 종료시각 11~15시에 3, 나머지에 1을 적용합니다. 그룹별 부호 있는 오차를 합한 뒤 절댓값을 취해 `0.1 × SMP × k`를 곱합니다. 실적이 없거나 음수인 발전소가 있는 시각은 두 그룹의 평가에서 함께 제외하고 제외 수를 기록합니다. 0 kWh 실적은 유효합니다. `issue_cutoff`는 종료시각별 예측 마감을 계산합니다.

테스트는 합성 자료로 정산식, KST 자정 경계, 결측 실적, 필수 예측 및 용량 제약, CLI를 검증합니다. 실제 모델 학습과 제출 파일 생성은 입력 데이터와 시간순 검증 구간을 정한 뒤 진행합니다.
