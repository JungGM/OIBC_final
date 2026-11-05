# OIBC_final

- [train](train/README.md): 이 채팅 이전부터 있던 학습 데이터 생성·검증 코드, 고정 데이터 설정과 문서
- [GM_Ver1](GM_Ver1/README.md): 이번 채팅에서 만든 XGBoost·GRU·TCN 예측 모델 6개, 학습 가중치와 예측·검증 결과
- [스태킹 결과](GM_Ver1/Stacking/README.md): 6개 예측을 결합한 Ridge 메타모델과 2025년 비교 결과
- [경사하강법 결과](GM_Ver1/GradientDescent/README.md): 그룹 벌금 기반 가중치 최적화와 2025년 4~5월 진단
- [GA VPP 배정과 정산](GM_Ver1/GeneticAlgorithm/README.md)에서 배정표, 그룹 예측과 정산 시나리오를 확인할 수 있다.
- [모델과 배정 감사](GM_Ver1/AUDIT_20261105.md)에 공식 제출 계열과의 비교 및 개선 우선순위를 기록했다.

실제 학습 데이터 Parquet는 기존 로컬 위치를 사용하며 새로 업로드하지 않는다. 모델 실행 명령과 결과는 각 폴더의 README를 참조한다.
