"""Drop-off and resilience scores versus a clean baseline."""

from visioneval.robustness.degradation import drop_off, resilience_score, summarize_degradation


def test_drop_off_standard_cases() -> None:
    assert drop_off(1.0, 0.5) == 0.5
    assert drop_off(0.8, 0.8) == 0.0
    assert drop_off(0.5, 0.9) == 0.0  # improvement is not a negative drop
    assert drop_off(0.0, 0.0) == 0.0
    assert drop_off(0.0, 0.4) == 1.0


def test_resilience_is_one_minus_mean_drop() -> None:
    # clean=1, corrupted 0.8 and 0.6 -> drops 0.2 and 0.4 -> mean drop 0.3 -> res 0.7
    assert abs(resilience_score(1.0, [0.8, 0.6]) - 0.7) < 1e-9
    assert resilience_score(1.0, []) == 1.0


def test_summarize_degradation_orders_severities() -> None:
    report = summarize_degradation(
        metric="clip_score",
        corruption="gaussian_noise",
        clean_score=1.0,
        scores_by_severity={0.5: 0.8, 0.25: 0.9, 1.0: 0.4},
    )
    assert report.resilience == resilience_score(1.0, [0.9, 0.8, 0.4])
    assert [p.severity for p in report.points] == [0.25, 0.5, 1.0]
    assert report.points[0].drop_off == drop_off(1.0, 0.9)
    payload = report.as_dict()
    assert payload["metric"] == "clip_score"
    assert len(payload["points"]) == 3
