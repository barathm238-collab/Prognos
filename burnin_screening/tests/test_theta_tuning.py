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
