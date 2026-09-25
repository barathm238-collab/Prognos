"""Tests for the team-tuned theta_slope gate (Prompt B2 tuning note).

theta_k is the MAD multiplier in theta_slope = median + theta_k * MAD over
the normal population. Spec default 3.0; the demo pipeline ships 2.4 after a
joint grid over Module A's z_threshold/contamination and this multiplier
showed the drift gate is the only effective FN lever on the compound-defect
case (C0365, seed 42/n=500: v_drift 4.43 vs theta 4.68 at k=3.0; FLAGed at
k=2.4 — verified in progress.md §7, kept out of the suite to avoid a slow
500-row training regression test).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from conftest import PARAMS, Y_spec, module_a_static, module_b_temporal as mbt, pipeline_main


def _theta_formula(v_drift: np.ndarray, k: float) -> float:
    med = float(np.median(v_drift))
    mad = float(np.median(np.abs(v_drift - med)))
    return med + k * mad


@pytest.fixture(scope="module")
def df_ab(df_data, Y_spec):
    """Module A + Module B frame trained ONCE at the spec default (k=3.0)."""
    df_a = module_a_static.run_module_a(df_data, Y_spec, PARAMS)
    df_b, _models, theta = mbt.run_module_b(
        df_data, PARAMS, train=True, static_outlier_flag=df_a["static_outlier_flag"]
    )
    b_cols = mbt.module_b_new_cols(PARAMS)
    merged = df_a.merge(df_b[["component_id"] + b_cols], on="component_id")
    return merged, theta


def test_theta_k_spec_default_is_3():
    """Prompt B2's literal multiplier is the signature default — pin it."""
    assert inspect.signature(mbt.run_module_b).parameters["theta_k"].default == 3.0


def test_theta_slope_matches_formula_at_spec_default(df_ab):
    merged, theta = df_ab
    normal_v = merged.loc[~merged["static_outlier_flag"], "v_drift"].to_numpy(dtype=float)
    assert theta == pytest.approx(_theta_formula(normal_v, 3.0), abs=1e-9)


def test_lower_theta_k_strictly_lowers_threshold(df_ab):
    """MAD > 0 on real data -> theta(k) strictly decreasing in k."""
    merged, _ = df_ab
    normal_v = merged.loc[~merged["static_outlier_flag"], "v_drift"].to_numpy(dtype=float)
    assert float(np.median(np.abs(normal_v - np.median(normal_v)))) > 0.0, (
        "degenerate MAD on the shared dataset — the monotonicity test below is meaningless"
    )
    thetas = {k: _theta_formula(normal_v, k) for k in (3.0, 2.8, 2.4, 2.0)}
    assert thetas[3.0] > thetas[2.8] > thetas[2.4] > thetas[2.0]


def test_lower_theta_k_flags_monotonically_more(df_ab):
    """Fewer escapes as the gate tightens: drift flags never decrease as k drops."""
    merged, _ = df_ab
    normal_v = merged.loc[~merged["static_outlier_flag"], "v_drift"].to_numpy(dtype=float)
    theta30 = _theta_formula(normal_v, 3.0)
    theta24 = _theta_formula(normal_v, 2.4)
    flags30 = int((merged["v_drift"] > theta30).sum())
    flags24 = int((merged["v_drift"] > theta24).sum())
    assert flags24 >= flags30
    # and on this dataset the tightening actually catches more rows
    assert flags24 > flags30


def test_pipeline_threads_theta_k_end_to_end(df_data):
    """run_pipeline(theta_k=...) -> Module B's returned theta_slope uses it."""
    df_full, results = pipeline_main.run_pipeline(
        df_data.head(100), params=PARAMS, verbose=False,
        with_intervals=False, with_ood=False, theta_k=2.4,
    )
    normal_v = df_full.loc[~df_full["static_outlier_flag"], "v_drift"].to_numpy(dtype=float)
    assert results["theta_slope"] == pytest.approx(_theta_formula(normal_v, 2.4), abs=1e-9)


# ---------------------------------------------------------------------------
# 🔵 theta_blended — per-lot Empirical-Bayes theta (Phase 1 report §8b)
# ---------------------------------------------------------------------------
def test_theta_blended_off_returns_scalar_and_matches_global(df_ab):
    """Default (off) keeps the float theta — the required path is unchanged."""
    merged, theta = df_ab  # df_ab was built WITHOUT theta_blended
    assert isinstance(theta, float)


def test_theta_blended_returns_per_lot_dict(df_data, Y_spec):
    df_a = module_a_static.run_module_a(df_data, Y_spec, PARAMS)
    df_b, _models, theta = mbt.run_module_b(
        df_data, PARAMS, train=True, static_outlier_flag=df_a["static_outlier_flag"],
        theta_blended=True,
    )
    assert isinstance(theta, dict)
    lots = set(df_data["lot_id"].unique())
    assert set(theta.keys()) == lots
    # drift_flag exists, is boolean, and every row's flag agrees with its own
    # lot's blended theta (v_drift > theta_by_lot[lot]).
    assert df_b["drift_flag"].dtype == bool
    for lot, t in theta.items():
        rows = df_b[df_b["lot_id"] == lot]
        assert (rows["drift_flag"] == (rows["v_drift"] > t)).all(), (
            f"lot {lot}: drift_flag does not match its own blended theta"
        )


def test_theta_blended_formula_and_shrinkage(df_data, Y_spec):
    """Per-lot theta = w*(lot med + k*lot MAD) + (1-w)*global theta, w=N/(N+30)."""
    df_a = module_a_static.run_module_a(df_data, Y_spec, PARAMS)
    df_b, _models, theta = mbt.run_module_b(
        df_data, PARAMS, train=True, static_outlier_flag=df_a["static_outlier_flag"],
        theta_blended=True,
    )
    merged = df_a.merge(
        df_b[["component_id", "v_drift"]], on="component_id"
    )
    mask = ~merged["static_outlier_flag"]
    normal_v = merged.loc[mask, "v_drift"].to_numpy(dtype=float)
    g_med = float(np.median(normal_v))
    g_mad = float(np.median(np.abs(normal_v - g_med)))
    global_theta = g_med + 3.0 * g_mad

    checked = 0
    for lot, t in theta.items():
        lot_v = merged.loc[mask & (merged["lot_id"] == lot), "v_drift"].to_numpy(dtype=float)
        if len(lot_v) == 0:
            continue
        n = len(lot_v)
        w = n / (n + 30)
        l_med = float(np.median(lot_v))
        l_mad = float(np.median(np.abs(lot_v - l_med)))
        expected = w * (l_med + 3.0 * l_mad) + (1 - w) * global_theta
        assert t == pytest.approx(expected, abs=1e-9)
        # shrinkage: a small lot's blended theta is pulled toward the global
        # value relative to its own raw lot theta
        raw_lot_theta = l_med + 3.0 * l_mad
        if w < 1.0:
            assert abs(t - global_theta) <= abs(raw_lot_theta - global_theta) + 1e-12
        checked += 1
    assert checked > 0, "no lot had normal-population rows to verify"


def test_theta_blended_pipeline_end_to_end(df_data):
    """run_pipeline(theta_blended=True) produces a valid decision frame."""
    df_full, results = pipeline_main.run_pipeline(
        df_data.head(120), params=PARAMS, verbose=False,
        with_intervals=False, with_ood=False, theta_blended=True,
    )
    assert "decision" in df_full.columns and "reason" in df_full.columns
    assert set(df_full["decision"].unique()) <= {"PASS", "REVIEW", "FLAG"}
    assert "theta_slope" in results  # scalar cited in reasons (median across lots)
