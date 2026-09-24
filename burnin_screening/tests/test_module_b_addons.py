"""Tests for Module B's 🔵 add-ons — Prompt B3 (MAPIE intervals), Prompt B4
(OOD gate), Prompt B5 (progressive refinement).

The required 🟢 B1/B2 path has its own regression suite (test_module_b.py);
here the default path is only re-checked for column invariance (no add-on
leakage when the new arguments stay at their defaults), then each add-on is
exercised on a small deterministic dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import PARAMS, module_b_temporal as mbt


@pytest.fixture(scope="module")
def df_small(df_data):
    return df_data.head(80).reset_index(drop=True)


@pytest.fixture(scope="module")
def y_spec_full(df_data):
    return {p: float(df_data[f"Y_spec_{p}"].iloc[0]) for p in PARAMS}


# ---------------------------------------------------------------------------
# Default-path invariance — add-ons must not leak into the 🟢 path
# ---------------------------------------------------------------------------
def test_default_path_columns_unchanged(df_small, y_spec_full):
    df_b, models, theta = mbt.run_module_b(df_small, PARAMS, train=True)
    for c in mbt.module_b_new_cols(PARAMS):
        assert c in df_b.columns
    for p in PARAMS:
        assert f"{p}_168h_lower" not in df_b.columns
        assert f"{p}_168h_upper" not in df_b.columns


def test_module_b_new_cols_signature_unchanged_by_default():
    cols = mbt.module_b_new_cols(PARAMS)
    assert "v_drift" in cols and "drift_flag" in cols
    assert not any(c.endswith("_lower") or c.endswith("_upper") for c in cols)
    # and the intervals variant appends exactly two columns per parameter
    cols_i = mbt.module_b_new_cols(PARAMS, with_intervals=True)
    assert len(cols_i) == len(cols) + 2 * len(PARAMS)
    for p in PARAMS:
        assert f"{p}_168h_lower" in cols_i and f"{p}_168h_upper" in cols_i


# ---------------------------------------------------------------------------
# 🔵 Prompt B3 — MAPIE conformal intervals (with_intervals=True)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def df_intervals(df_small, y_spec_full):
    df_a = __import__("module_a_static").run_module_a(df_small, y_spec_full, PARAMS)
    df_b, _models, _theta = mbt.run_module_b(
        df_small, PARAMS, train=True, static_outlier_flag=df_a["static_outlier_flag"],
        with_intervals=True,
    )
    return df_b


def test_b3_interval_columns_present(df_intervals):
    for p in PARAMS:
        for c in (f"{p}_168h_lower", f"{p}_168h_upper"):
            assert c in df_intervals.columns, f"missing B3 column {c}"


def test_b3_intervals_bracket_point_prediction(df_intervals):
    for p in PARAMS:
        lower = df_intervals[f"{p}_168h_lower"]
        upper = df_intervals[f"{p}_168h_upper"]
        pred = df_intervals[f"{p}_168h_pred"]
        assert (lower <= upper).all(), f"{p}: inverted interval"
        # "plus" method point prediction sits inside its own interval
        assert (pred >= lower - 1e-9).all() and (pred <= upper + 1e-9).all(), (
            f"{p}: point prediction outside its conformal interval"
        )


def test_b3_intervals_finite_and_positive_width(df_intervals):
    for p in PARAMS:
        width = df_intervals[f"{p}_168h_upper"] - df_intervals[f"{p}_168h_lower"]
        assert np.isfinite(df_intervals[f"{p}_168h_lower"]).all()
        assert np.isfinite(df_intervals[f"{p}_168h_upper"]).all()
        assert (width >= 0).all()


def test_b3_96h_pred_is_not_the_mapie_path(df_intervals, df_small, y_spec_full):
    """B3 covers the 168h target ONLY — 96h stays the plain MultiOutput path."""
    df_plain, _, _ = mbt.run_module_b(df_small, PARAMS, train=True)
    for p in PARAMS:
        pd.testing.assert_series_equal(
            df_intervals[f"{p}_96h_pred"].sort_index(),
            df_plain[f"{p}_96h_pred"].sort_index(),
            check_names=False,
        )


def test_b3_flag_and_v_drift_still_computed(df_intervals):
    assert df_intervals["drift_flag"].dtype == bool
    assert np.isfinite(df_intervals["v_drift"]).all()


# ---------------------------------------------------------------------------
# 🔵 Prompt B4 — OOD gate on raw regression inputs
# ---------------------------------------------------------------------------
def test_b4_fit_and_score_roundtrip(df_small):
    X = mbt.build_ood_features(df_small, PARAMS)
    detector = mbt.fit_ood_gate(X)
    ood_score, ood_flag = mbt.score_ood(detector, X)
    assert len(ood_score) == len(df_small) and len(ood_flag) == len(df_small)
    assert ood_flag.dtype == bool
    assert np.isfinite(ood_score).all()
    # training data itself: flagged fraction ~ contamination (0.05), not ~50%
    assert ood_flag.mean() < 0.25


def test_b4_score_higher_means_more_anomalous(df_small):
    """A far-out-of-distribution row must score higher than a normal one."""
    X = mbt.build_ood_features(df_small, PARAMS)
    detector = mbt.fit_ood_gate(X)
    normal_row = X.iloc[[5]].copy()
    ood_normal, _ = mbt.score_ood(detector, normal_row)
    wild = normal_row.copy()
    for p in PARAMS:
        wild[f"{p}_24h"] = wild[f"{p}_24h"] * 25.0
        wild[f"{p}_delta"] = wild[f"{p}_24h"] - wild[f"{p}_0h"]
    ood_wild, flag_wild = mbt.score_ood(detector, wild)
    assert ood_wild[0] > ood_normal[0]
    assert bool(flag_wild[0]) is True


def test_b4_features_are_raw_inputs_not_z_scores(df_small):
    """B4 watches raw inputs — features must be built from the §4b schema only."""
    X = mbt.build_ood_features(df_small, PARAMS)
    for p in PARAMS:
        for c in (f"{p}_0h", f"{p}_24h", f"{p}_delta"):
            assert c in X.columns
        assert f"{p}_z" not in X.columns
    assert "T_ambient" in X.columns and "V_stress" in X.columns
    # shared context appears once, not once per parameter
    assert list(X.columns).count("T_ambient") == 1
    # delta formula matches Prompt B1
    for p in PARAMS:
        pd.testing.assert_series_equal(
            X[f"{p}_delta"], df_small[f"{p}_24h"] - df_small[f"{p}_0h"], check_names=False
        )


def test_b4_per_parameter_feature_set(df_small):
    """Documented per-parameter use: one parameter -> its 5-column matrix."""
    X = mbt.build_ood_features(df_small, ["I_leak"])
    assert list(X.columns) == ["I_leak_0h", "I_leak_24h", "I_leak_delta", "T_ambient", "V_stress"]


# ---------------------------------------------------------------------------
# 🔵 Prompt B5 — progressive refinement (standalone)
# ---------------------------------------------------------------------------
def test_b5_prefers_measured_when_present():
    row = {"I_leak_96h_actual": 11.5, "I_leak_96h_pred": 10.0}
    val, src = mbt.resolve_checkpoint(row, "I_leak", 96, 10.0)
    assert val == 11.5 and src == "measured"


def test_b5_falls_back_to_prediction_when_absent():
    val, src = mbt.resolve_checkpoint({}, "I_leak", 168, 12.0)
    assert val == 12.0 and src == "predicted"


def test_b5_nan_actual_falls_back_to_prediction():
    row = {"t_pd_168h_actual": float("nan")}
    val, src = mbt.resolve_checkpoint(row, "t_pd", 168, 8.5)
    assert val == 8.5 and src == "predicted"


def test_b5_none_actual_falls_back_to_prediction():
    row = {"I_ddq_168h_actual": None}
    val, src = mbt.resolve_checkpoint(row, "I_ddq", 168, 2.2)
    assert val == 2.2 and src == "predicted"


def test_b5_horizon_is_parameterized():
    row = {"I_leak_96h_actual": 11.0}
    val, src = mbt.resolve_checkpoint(row, "I_leak", 96, 10.0)
    assert src == "measured"
    val, src = mbt.resolve_checkpoint(row, "I_leak", 168, 10.0)
    assert src == "predicted"
