import numpy as np
import pytest
from nsga2_sats.metrics import hypervolume, igd, nondominated, reference_front, spread, paired_tests


def test_hv_matches_hand_computed_union_and_ignores_dominated_points():
    points = [[.2, .8], [.5, .5], [.8, .2], [.8, .8], [2, 0], [.2, .8]]
    assert hypervolume(points, (1, 1)) == pytest.approx(.37)
    assert hypervolume([], (1, 1)) == 0


def test_reference_is_sorted_unique_union_nondominated():
    ref = reference_front([[[.2, .8], [.5, .5]], [[.8, .2], [.5, .5], [.9, .9]]])
    np.testing.assert_array_equal(ref, [[.2, .8], [.5, .5], [.8, .2]])


def test_igd_and_spread_use_reference_points_and_extremes():
    ref = np.array([[0., 1.], [.5, .5], [1., 0.]])
    assert igd(ref, ref) == 0
    assert spread(ref, ref) == 0
    assert igd([[.5, .5]], ref) == pytest.approx(2 * np.sqrt(.5) / 3)
    assert spread([[.25, .75], [.75, .25]], ref) == pytest.approx(.5)
    assert spread([[.5, .5]], ref) == 1


def test_invalid_front_is_rejected():
    with pytest.raises(ValueError):
        nondominated([[np.nan, 0]])
    with pytest.raises(ValueError):
        igd([[0, 0]], [])


def test_one_algorithm_does_not_invent_statistical_comparisons():
    rows = [{"scenario": "AFIT1", "algorithm": "full", "HV_mean": .5, "IGD_mean": .1, "Spread_mean": .2}]
    report = paired_tests(rows)
    assert report["algorithms_present"] == ["full"]
    for entry in report["metrics"].values():
        assert entry["friedman"] is None
        assert entry["wilcoxon"] == []


def test_statistics_pair_shared_scenarios_and_handle_all_ties():
    rows = [{"scenario": s, "algorithm": a, "HV_mean": .5, "IGD_mean": .1, "Spread_mean": .2}
            for a in ("a", "b", "c") for s in ("AFIT1", "AFIT2", "AFIT3")]
    rows += [{"scenario": "AFIT4", "algorithm": "a", "HV_mean": .9, "IGD_mean": .1, "Spread_mean": .2}]
    report = paired_tests(rows)
    for entry in report["metrics"].values():
        assert entry["complete_scenarios"] == ["AFIT1", "AFIT2", "AFIT3"]
        assert entry["friedman"]["p_value"] == 1
        assert entry["average_ranks"] == {"a": 2, "b": 2, "c": 2}
        assert all(p["p_value_holm"] == 1 for p in entry["wilcoxon"])
