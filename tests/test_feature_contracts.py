from dataclasses import asdict

from marvis.feature import (
    Bin,
    BinningResult,
    CorrelationReport,
    FeatureMetrics,
    WOEResult,
)


def _bin(index: int, lower: float, upper: float, bad_rate: float) -> Bin:
    return Bin(
        index=index,
        lower=lower,
        upper=upper,
        count=10,
        bad_count=int(bad_rate * 10),
        good_count=10 - int(bad_rate * 10),
        bad_rate=bad_rate,
        woe=0.1 * index,
        iv_contribution=0.01 * index,
    )


def test_binning_result_contract_round_trips_with_na_bin():
    bins = (_bin(0, float("-inf"), 0.0, 0.1), _bin(1, 0.0, float("inf"), 0.3))
    na_bin = _bin(-1, float("nan"), float("nan"), 0.2)
    result = BinningResult(
        feature="age",
        method="equal_freq",
        bins=bins,
        edges=(float("-inf"), 0.0, float("inf")),
        total_iv=0.03,
        monotonic=True,
        na_bin=na_bin,
    )

    payload = asdict(result)

    assert len(result.edges) == len(result.bins) + 1
    assert payload["feature"] == "age"
    assert payload["na_bin"]["index"] == -1
    assert payload["bins"][1]["bad_rate"] == 0.3


def test_feature_metrics_and_woe_result_contracts():
    metrics = FeatureMetrics(
        feature="score",
        iv=0.2,
        ks=0.34,
        auc=0.71,
        psi=None,
        missing_rate=0.05,
        unique_count=100,
        lift_top_bin=1.8,
    )
    woe = WOEResult(
        feature="score",
        edges=(float("-inf"), 0.0, float("inf")),
        woe_by_bin=(-0.2, 0.4),
        na_woe=None,
    )

    assert metrics.psi is None
    assert 0 <= metrics.ks <= 1
    assert len(woe.woe_by_bin) == len(woe.edges) - 1


def test_correlation_report_contract_shape():
    report = CorrelationReport(
        features=("x1", "x2"),
        matrix=((1.0, 0.92), (0.92, 1.0)),
        collinear_pairs=(("x1", "x2", 0.92),),
        vif={"x1": 6.2, "x2": 6.2},
    )

    assert len(report.matrix) == len(report.features)
    assert all(len(row) == len(report.features) for row in report.matrix)
    assert report.collinear_pairs[0] == ("x1", "x2", 0.92)
