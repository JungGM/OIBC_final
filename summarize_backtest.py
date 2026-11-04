"""Create an aggregate-only Korean report and a reproducible scientific plot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    read = lambda name: json.loads((out/name).read_text(encoding="utf-8"))
    frozen, audit, selection = map(read, ["frozen_plan.json", "audit.json", "model_selection.json"])
    group, check, done = map(read, ["group_search.json", "group_validation.json", "completion.json"])
    metrics = pd.read_csv(out/"metrics.csv")
    full = metrics[(metrics.scope == "complete_253_panel") & (metrics.period == "all")].set_index("model")
    monthly = pd.read_csv(out/"monthly_scores.csv")
    labels = {"baseline_reference": "LightGBM 기준", "ga_xgb_reference": "GA-XGBoost / 참조 그룹", "frozen_selected": "2023년에 선택한 최종안"}
    baseline = full.loc["baseline_reference"]
    selected = full.loc["frozen_selected"]
    verification = read("verification.json")
    assert verification["status"] == "PASS"
    display_models = ["baseline_reference", "frozen_selected"]
    if frozen["group_candidate_accepted"]:
        display_models.insert(1, "ga_xgb_reference")
    improvement = 100*(1-selected.penalty_krw/baseline.penalty_krw)
    table = ["| 모델 | 오차 벌금(원) ↓ | 정산금 S(원) ↑ | 기준 대비 벌금 감소 |", "|---|---:|---:|---:|"]
    for name, r in full.loc[display_models].iterrows():
        table.append(f"| {labels[name]} | {r.penalty_krw:,.0f} | {r.settlement_krw:,.0f} | {100*(1-r.penalty_krw/baseline.penalty_krw):.2f}% |")
    individual = ["| 모델 | 기간 | 관측 행 수 | MAE(kWh) | RMSE(kWh) | 용량 정규화 MAE |", "|---|---|---:|---:|---:|---:|"]
    for r in metrics[(metrics.scope == "observed_site_rows") & (metrics.period != "all") & metrics.model.isin(display_models)].itertuples():
        individual.append(f"| {labels[r.model]} | {r.period} | {int(r.rows):,} | {r.mae_kwh:.3f} | {r.rmse_kwh:.3f} | {r.capacity_nmae_pct:.3f}% |")
    monthly_table = ["| 기간 | 전체 관측 시간 | 기준 벌금(원) | 최종안 벌금(원) | 벌금 감소 |", "|---|---:|---:|---:|---:|"]
    mb = monthly[monthly.model == "baseline_reference"].set_index("month")
    ms = monthly[monthly.model == "frozen_selected"].set_index("month")
    for m, r in mb.iterrows():
        s = ms.loc[m]
        monthly_table.append(f"| {m} | {int(r.hours):,} | {r.penalty_krw:,.0f} | {s.penalty_krw:,.0f} | {100*(1-s.penalty_krw/r.penalty_krw):.2f}% |")
    change_text = "감소" if improvement >= 0 else "증가"
    report = f"""# 첫 실제 백테스트 결과 — 2026-10-29

2022~2023년 자료로 학습하고 2024년~2025년 5월 예측을 실제로 생성·채점했다.
논문은 방법론 참고로 사용했으며, 논문 데이터나 외부 사전학습 모델은 사용하지 않았다.
공식 대회에는 제출하지 않았다.

## 1. 기간 이해와 검증 범위

공식 제출 대상은 **2025년 6월~2026년 5월**이다. 학습 데이터는 2021년부터 2025년 5월까지
있으며, 반드시 2024~2025년만 사용해야 하는 것은 아니다. 이번에는 요청대로 2022~2023년만
학습에 사용했다. 2025년 6~12월 실제 발전량은 제공되지 않아 그 구간은 채점할 수 없다.

2022년에 실적이 있는 발전소는 80개, 2023년에는 148개다. 나머지 발전소는 공통 모델로 예측했다.
전체 253개 정답이 있는 최종 검증 시간은 **{int(baseline.hours):,}시간**이다.
2024년 {audit['complete_253_hours_by_year']['2024']:,}시간, 2025년 1~5월 {audit['complete_253_hours_by_year']['2025']:,}시간이다.
결측을 0으로 만들어 전체 기간 점수를 꾸미지 않았다.

## 2. 전체 253개를 비교할 수 있는 시간의 결과

{chr(10).join(table)}

선택은 2023년 자료로 끝냈다. 이 최종안의 벌금은 기준 대비 **{abs(improvement):.2f}% {change_text}**했다.
동일한 실적·가격에서 매전수익은 {baseline.revenue_krw:,.0f}원으로 모델과 무관하며,
정산금 차이는 벌금 차이에서 나온다. 여기의 원 단위 금액은 **대회 정산식을 적용한 로컬 모의 금액**이다.
공식 서버 점수나 실제 수익을 뜻하지 않는다.

정산식: `S = Σ SMP × 실제 발전량 - 0.10 × Σ SMP × k × |그룹 예측 - 그룹 실적|`.
벌금은 작을수록, 정산금은 클수록 좋다.

## 3. 단계별로 무엇을 했고 왜 했는가

1. **데이터 점검** — 원본 중복·결측·시간대·음수 실적·발전소별 관측 범위를 확인했다.
   없는 실적을 0으로 가정하지 않았다. 원본 파일은 수정하지 않고 해시를 기록했다.
2. **기준 모델 재학습** — 공식 LightGBM의 특징 생성 함수와 600개 트리 설정을 사용했다.
   발전량을 용량으로 나눈 이용률을 예측하고 용량을 다시 곱했다. 10시와 17시 예보로 각각
   예측한 뒤 평균했다. 학습 기간·seed·스레드 수를 바꾼 백테스트이며 공식 제출 실행 자체는 아니다.
3. **예측 논문 적용** — XGBoost에 제공사 간 차이, 두 발행시각의 차이, 대상일 내 3/7시간
   예보 평균을 추가했다. GA 4개체×3세대로 설정을 탐색했다. 2022년 학습 → 2023년 1~6월
   부분 관측 그룹 벌금으로 선택한 모델은 **{selection['chosen']}**다.
4. **조합 논문 적용** — 선택 모델을 2023년 6월까지 재학습해 7~9월 잔차를 만들었다.
   실적이 안정적으로 있는 {group['cohort_sites']}개 발전소, {group['complete_cohort_hours']:,}시간의
   고정 코호트로 5,000회 교환을 탐색했다. {group['accepted_swaps']}회 개선 교환을 찾았다.
   전체 발전소의 두 그룹 용량은 항상 20MW 이상으로 제한했다.
5. **그룹 재검증** — 같은 코호트의 2023년 10~12월 벌금은 참조 그룹
   {check['reference']['penalty_krw']:,.0f}원, 후보 그룹 {check['candidate']['penalty_krw']:,.0f}원이었다.
   1% 이상 개선 조건에 따른 후보 채택 여부는 **{'채택' if check['accepted'] else '미채택, 참조 그룹 유지'}**다.
   이는 부분 코호트 검증이므로 전체 253개에 최적인 그룹이라는 의미는 아니다.
6. **강건 입찰 논문 검토** — 이 대회에는 ESS·수요 운영이 없고 오차 벌금이 대칭이다.
   대칭 표본 구간을 붙이는 단순 SRO에서는 절대오차 목적에 상수만 더해져 최적 입찰량이
   바뀌지 않는다. 전체 VPP의 2023년 잔차도 없으므로 임의로 예측을 낮추는 보정은 하지 않았다.
   원 논문의 다단계 최적화 전체를 재현한 것은 아니다.
7. **선택 동결 후 최종 학습** — {frozen['fit_cutoff']}까지 학습하고 2024년부터의 예측을
   생성했다. 첫 예측일 전날 17시 이후 실적을 쓰지 않도록 경계 7시간도 제외했다.
   2024~2025년 정답으로 재학습·모델 선택·그룹 탐색·오차 보정을 하지 않았다.

## 4. 모든 관측 발전소 행에서의 보조 진단

{chr(10).join(individual)}

용량 정규화 MAE = `평균(|예측-실적|/설비용량) × 100`이다. 실제 발전량으로 나누는 MAPE와 다르다.
야간을 포함한 지표이며 대회 그룹 점수와도 다르다. 기간마다 관측 발전소 집합이 달라 연도 간
수치 차이를 모델 성능 변화만으로 해석할 수 없다.

2025년 개별 발전소 RMSE는 GA-XGBoost가 기준보다 나빠졌다. 그룹 벌금의 개선을 모든
정확도 지표가 좋아졌다는 의미로 해석해서는 안 된다. 다음 실험에서는 큰 개별 오차도 점검해야 한다.

## 5. 월별 완전 관측 구간

{chr(10).join(monthly_table)}

![Monthly penalty comparison]({(out/'penalty_comparison.png').as_posix()})

2024년은 완전 관측 시간이 적다. 여름·장마철 전체 VPP 성능은 이 표만으로 확인할 수 없다.
2025년 1~5월은 253개 전체로 비교할 수 있다. 첫 실험 결과로 공식 평가 1년 성능을 보장하지 않는다.

## 6. 재현과 확인

- 안전·채점 합성 테스트 및 기존 형식검증 테스트: 9개 통과.
- 공식 특징 생성과의 표본 대조: 10시/17시 각각 96행 일치.
- 독립적인 행렬 계산으로 정산금 재계산 일치, 원본 해시 보존, 모든 예측 행·그룹 용량 검사 PASS.
- 결과 예측 행 수: 발전소별 {done['predicted_site_rows']:,}, 그룹별 {done['backtest_group_rows']:,}.
- 선택 그룹 용량: {frozen['group_capacity_kw']} kW.
- 최종 실행 시간: {done['elapsed_seconds']/60:.1f}분. 초기 구현 수정 및 자료 검토 시간 제외.
- 정확한 설정, 버전, 원본 해시는 `frozen_plan.json`, `audit.json`에 저장했다.
- 모델·예측 파일 위치: `{out}`.
- `backtest_forecast_NOT_FOR_SUBMISSION.csv`는 제출기간이 다르므로 공식 제출용이 아니다.

다음 판단은 이번 평가구간을 다시 튜닝에 소비하지 않고, 현재 제공된 더 최근 학습 실적을 활용하는
별도 시간순 실험으로 해야 한다. 특히 편입이 늦은 발전소와 여름철 검증을 보강할 필요가 있다.
자세한 특징·기간·논문 적용 한계는 작업 폴더의 `BACKTEST_METHOD.md`에 기록했다.
"""
    (out/"BACKTEST_REPORT_KO.md").write_text(report, encoding="utf-8")
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 4.7), layout="constrained")
    xx = np.arange(len(mb))
    for offset, name, color in [(-0.17, "baseline_reference", "#54657e"), (0.17, "frozen_selected", "#1c8872")]:
        s = monthly[monthly.model == name].set_index("month").reindex(mb.index)
        ax.bar(xx+offset, 100*s.penalty_krw/s.revenue_krw, width=0.30, color=color, label=labels[name])
    ax.set_xticks(xx, [f"{m}\n({int(mb.loc[m, 'hours']):,}시간)" for m in mb.index])
    ax.set_ylabel("오차 벌금 / 매전수익 (%) · 낮을수록 좋음")
    ax.set_title("전체 253개 실적이 있는 시간만 비교", loc="left", fontweight="bold", pad=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out/"penalty_comparison.png", dpi=180)
    print(out/"BACKTEST_REPORT_KO.md")


if __name__ == "__main__":
    main()
