"""Tests for the 🔵 Phase 3 add-ons — PASS/REVIEW/FLAG split.

Covers:
  - decision_maker: OOD REVIEW branch, wide-interval REVIEW branch, priority
    (REVIEW never overrides FLAG), graceful degradation to binary PASS/FLAG.
  - main: theta_width derivation (median + 3*MAD, normal population),
    progressive refinement (no-op without actuals, endpoint override,
    96h-window slope, graded column untouched), add-on switches.
  - validate: strict vs lenient recall, deferred defects, compound REVIEW.

Hand-built rows follow the same pattern as test_decision_maker.py; pipeline
behavior is exercised on a small deterministic slice of the shared dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import (
    PARAMS,
    df_data,  # noqa: F401  (fixture import for reuse below)
    module_a_static,
    module_b_temporal as mbt,
    pipeline_main,
)
from decision_maker import decide, max_relative_width
from main import apply_progressive_refinement, derive_theta_width
from validate import validate


THETA = 1.23


def _base_row(component_id: str = "C0001") -> dict:
    """One clean row carrying every contract column decide() reads."""
    row: dict = {"component_id": component_id, "lot_id": 1}
    for p in PARAMS:
        row[f"{p}_24h"] = 1.0
        row[f"{p}_z"] = 0.1
        row[f"{p}_slope_norm"] = 0.2
    row["z_fused"] = 0.2
    row["joint_anomaly_score"] = 0.0
    row["joint_outlier_flag"] = False
    row["static_outlier_flag"] = False
    row["v_drift"] = 0.3
    row["drift_flag"] = False
    return row


def _interval_row(widths: dict[str, float], pred: float = 10.0) -> dict:
    """Clean row with 168h conformal intervals of the given relative width."""
    r = _base_row()
    for p in PARAMS:
        half = pred * widths.get(p, 0.0) / 2.0
        r[f"{p}_168h_pred"] = pred
        r[f"{p}_168h_lower"] = pred - half
        r[f"{p}_168h_upper"] = pred + half
    return r


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OOD REVIEW branch (decision_maker branch 5)
# ---------------------------------------------------------------------------
def test_ood_flag_defers_to_review():
    r = _base_row()
    r["ood_flag"] = True
    r["ood_score"] = 0.55
    out = decide(_frame([r]), PARAMS, THETA)
    assert out.loc[0, "decision"] == "REVIEW"
    reason = out.loc[0, "reason"]
    assert "OOD" in reason and "0.55" in reason
    assert any(p in reason for p in PARAMS)  # names a leading parameter


def test_ood_without_score_still_reviews():
    """ood_score column absent -> 'n/a' in the reason, REVIEW still fires."""
    r = _base_row()
    r["ood_flag"] = True
    out = decide(_frame([r]), PARAMS, THETA)
    assert out.loc[0, "decision"] == "REVIEW"
    assert "n/a" in out.loc[0, "reason"]


def test_ood_never_overrides_flag_branches():
    """Evidence-based detections stay actionable — REVIEW only for would-PASS."""
    r = _base_row()
    r["ood_flag"] = True
    r["I_leak_hard_flag"] = True
    out = decide(_frame([r]), PARAMS, THETA)
    assert out.loc[0, "decision"] == "FLAG"
    assert "Hard limit" in out.loc[0, "reason"]

    r2 = _base_row()
    r2["ood_flag"] = True
    r2["joint_outlier_flag"] = True
    r2["joint_anomaly_score"] = 0.7
    out2 = decide(_frame([r2]), PARAMS, THETA)
    assert out2.loc[0, "decision"] == "FLAG"
    assert "Joint" in out2.loc[0, "reason"]

    r3 = _base_row()
    r3["ood_flag"] = True
    r3["drift_flag"] = True
    r3["v_drift"] = 2.0
    out3 = decide(_frame([r3]), PARAMS, THETA)
    assert out3.loc[0, "decision"] == "FLAG"


def test_no_ood_column_means_no_review():
    """Contract-minimum frame (no ood_flag) -> unchanged binary behavior."""
    out = decide(_frame([_base_row()]), PARAMS, THETA)
    assert out.loc[0, "decision"] == "PASS"


# ---------------------------------------------------------------------------
# Wide-interval REVIEW branch (decision_maker branch 6)
# ---------------------------------------------------------------------------
def test_wide_interval_defers_to_review():
    r = _interval_row({p: 0.6 for p in PARAMS})
    out = decide(_frame([r]), PARAMS, THETA, theta_width=0.5)
    assert out.loc[0, "decision"] == "REVIEW"
    reason = out.loc[0, "reason"]
    assert "interval" in reason and "0.60" in reason and "0.50" in reason


def test_narrow_interval_stays_pass():
    r = _interval_row({p: 0.4 for p in PARAMS})
    out = decide(_frame([r]), PARAMS, THETA, theta_width=0.5)
    assert out.loc[0, "decision"] == "PASS"


def test_width_review_names_widest_parameter():
    r = _interval_row({"I_leak": 0.2, "I_ddq": 0.9, "t_pd": 0.3})
    out = decide(_frame([r]), PARAMS, THETA, theta_width=0.5)
    assert out.loc[0, "decision"] == "REVIEW"
    assert "I_ddq" in out.loc[0, "reason"]


def test_width_branch_inert_without_theta_width():
    """theta_width=None (intervals off) -> no width REVIEW, even when huge."""
    r = _interval_row({p: 9.9 for p in PARAMS})
    out = decide(_frame([r]), PARAMS, THETA, theta_width=None)
    assert out.loc[0, "decision"] == "PASS"


def test_width_branch_skips_nan_intervals():
    """NaN interval bounds must degrade to no-width-signal, not crash."""
    r = _base_row()
    for p in PARAMS:
        r[f"{p}_168h_pred"] = 10.0
        r[f"{p}_168h_lower"] = float("nan")
        r[f"{p}_168h_upper"] = float("nan")
    out = decide(_frame([r]), PARAMS, THETA, theta_width=0.1)
    assert out.loc[0, "decision"] == "PASS"


def test_degenerate_zero_prediction_skipped():
    """pred ~ 0 -> relative width undefined -> row passes, no crash."""
    r = _interval_row({p: 0.9 for p in PARAMS}, pred=0.0)
    out = decide(_frame([r]), PARAMS, THETA, theta_width=0.1)
    assert out.loc[0, "decision"] == "PASS"


def test_wide_interval_never_overrides_flag():
    r = _interval_row({p: 0.9 for p in PARAMS})
    r["static_outlier_flag"] = True
    r["z_fused"] = 4.0
    out = decide(_frame([r]), PARAMS, THETA, theta_width=0.5)
    assert out.loc[0, "decision"] == "FLAG"


# ---------------------------------------------------------------------------
# max_relative_width + derive_theta_width (main.py)
# ---------------------------------------------------------------------------
def test_max_relative_width_takes_row_max():
    df = _frame([
        _interval_row({"I_leak": 0.1, "I_ddq": 0.5, "t_pd": 0.3}),
        _interval_row({"I_leak": 0.2, "I_ddq": 0.2, "t_pd": 0.2}),
    ])
    w = max_relative_width(df, PARAMS)
    assert w.iloc[0] == pytest.approx(0.5)
    assert w.iloc[1] == pytest.approx(0.2)


def test_max_relative_width_missing_intervals_all_nan():
    df = _frame([_base_row()])
    w = max_relative_width(df, PARAMS)
    assert w.isna().all()


def test_derive_theta_width_median_plus_3mad():
    rows = [_interval_row({p: 0.1 for p in PARAMS}) for _ in range(9)]
    outlier = _interval_row({p: 0.1 for p in PARAMS})
    outlier["static_outlier_flag"] = True  # excluded from the normal population
    outlier_wide = _interval_row({p: 5.0 for p in PARAMS})
    outlier_wide["static_outlier_flag"] = True
    df = _frame(rows + [outlier, outlier_wide])
    theta = derive_theta_width(df, PARAMS)
    # all normal widths identical (0.1) -> median 0.1, MAD 0 -> theta == 0.1
    assert theta == pytest.approx(0.1)


def test_derive_theta_width_no_intervals_returns_zero():
    assert derive_theta_width(_frame([_base_row()]), PARAMS) == 0.0


# ---------------------------------------------------------------------------
# Progressive refinement (main.apply_progressive_refinement)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def df_refine_base(df_data, Y_spec) -> pd.DataFrame:
    """Merged Module A + Module B frame (no intervals) for refinement tests."""
    small = df_data.head(60).reset_index(drop=True)
    df_a = module_a_static.run_module_a(small, Y_spec, PARAMS)
    df_b, _models, _theta = mbt.run_module_b(
        small, PARAMS, train=True, static_outlier_flag=df_a["static_outlier_flag"]
    )
    b_cols = mbt.module_b_new_cols(PARAMS)
    df_full = df_a.merge(df_b[["component_id"] + b_cols], on="component_id")
    df_full["ood_score"] = np.nan
    df_full["ood_flag"] = False
    return df_full


def test_refinement_noop_without_actuals(df_refine_base):
    before = df_refine_base.copy(deep=True)
    out = apply_progressive_refinement(df_refine_base, PARAMS)
    pd.testing.assert_frame_equal(out, before)
    assert not any(c.endswith("_refined") for c in out.columns)


def test_refinement_nan_actual_column_is_noop(df_refine_base):
    df = df_refine_base.copy()
    df["I_leak_168h_actual"] = np.nan
    before = df.copy(deep=True)  # includes the added _actual column
    out = apply_progressive_refinement(df, PARAMS)
    pd.testing.assert_frame_equal(out, before)
    assert not any(c.endswith("_refined") for c in out.columns)


def test_refinement_168h_actual_overrides_endpoint(df_refine_base):
    df = df_refine_base.copy()
    i = 3
    cid = df.loc[i, "component_id"]
    graded_before = df.loc[i, "I_leak_168h_pred"]
    actual = float(df.loc[i, "I_leak_168h_pred"]) * 1.5
    df["I_leak_168h_actual"] = np.nan
    df.loc[i, "I_leak_168h_actual"] = actual

    out = apply_progressive_refinement(df, PARAMS)
    row = out[out["component_id"] == cid].iloc[0]

    assert row["I_leak_168h_pred_refined"] == pytest.approx(actual)
    # graded column untouched — the MAE record stays comparable run over run
    assert row["I_leak_168h_pred"] == pytest.approx(graded_before)
    # slope recomputed from the refined endpoint over the 24h->168h window
    expected_slope = (actual - float(row["I_leak_24h"])) / (168.0 - 24.0)
    assert row["I_leak_slope"] == pytest.approx(expected_slope, rel=1e-9)
    # drift features recomputed for every row (normalization over the frame)
    assert np.isfinite(out["v_drift"]).all()
    assert out["drift_flag"].dtype == bool
    assert any(c.endswith("_refined") for c in out.columns)


def test_refinement_96h_only_uses_shorter_window(df_refine_base):
    df = df_refine_base.copy()
    i = 5
    cid = df.loc[i, "component_id"]
    actual96 = float(df.loc[i, "I_ddq_96h_pred"]) * 1.2
    df["I_ddq_96h_actual"] = np.nan
    df.loc[i, "I_ddq_96h_actual"] = actual96

    out = apply_progressive_refinement(df, PARAMS)
    row = out[out["component_id"] == cid].iloc[0]

    assert row["I_ddq_96h_pred_refined"] == pytest.approx(actual96)
    expected_slope = (actual96 - float(row["I_ddq_24h"])) / (96.0 - 24.0)
    assert row["I_ddq_slope"] == pytest.approx(expected_slope, rel=1e-9)
    # unrefined rows keep the plain 24h->168h slope
    j = 7
    cid_j = df.loc[j, "component_id"]
    row_j = out[out["component_id"] == cid_j].iloc[0]
    plain_slope = (float(row_j["I_ddq_168h_pred"]) - float(row_j["I_ddq_24h"])) / (168.0 - 24.0)
    assert row_j["I_ddq_slope"] == pytest.approx(plain_slope, rel=1e-9)


def test_refinement_prefers_measured_over_predicted(df_refine_base):
    """Integration of Module B's resolve_checkpoint rule: measured wins."""
    df = df_refine_base.copy()
    i = 2
    actual = 42.0
    df["t_pd_168h_actual"] = actual  # finite for every row
    out = apply_progressive_refinement(df, PARAMS)
    assert np.allclose(out["t_pd_168h_pred_refined"].to_numpy(dtype=float), actual,
                       rtol=1e-9, atol=1e-12)
    row = out.iloc[i]
    val, src = mbt.resolve_checkpoint(row, "t_pd", 168, row["t_pd_168h_pred"])
    assert src == "measured" and val == actual


# ---------------------------------------------------------------------------
# Pipeline wiring (main.run_pipeline with the Phase 3 switches)
# ---------------------------------------------------------------------------
# MAPIE (cv=5 x 3 XGBoost fits) dominates runtime, so the interval-enabled
# pipeline is exercised through ONE session-scoped fixture on a small slice.
@pytest.fixture(scope="session")
def df_wired(df_data):
    df_out, results = pipeline_main.run_pipeline(
        df_data.head(80), params=PARAMS, verbose=False
    )
    return df_out, results


def test_run_pipeline_theta_width_and_intervals(df_wired):
    df_full, results = df_wired
    assert "theta_width" in results and results["theta_width"] > 0.0
    for p in PARAMS:
        assert f"{p}_168h_lower" in df_full.columns
        assert f"{p}_168h_upper" in df_full.columns
    assert "ood_score" in df_full.columns and "ood_flag" in df_full.columns
    assert set(df_full["decision"].unique()) <= {"PASS", "REVIEW", "FLAG"}


def test_run_pipeline_ood_flags_attach_and_are_finite(df_wired):
    df_full, _ = df_wired
    assert np.isfinite(df_full["ood_score"]).all()
    assert df_full["ood_flag"].dtype == bool
    # contamination=0.05 -> the OOD flag must not blanket the population
    assert df_full["ood_flag"].mean() < 0.25


def test_run_pipeline_addon_switches_off(df_data):
    """All add-ons off -> binary decisions, no interval/OOD columns."""
    df_full, results = pipeline_main.run_pipeline(
        df_data.head(80), params=PARAMS, verbose=False,
        with_intervals=False, with_ood=False, with_progressive_refinement=False,
    )
    assert results["theta_width"] is None
    assert df_full["ood_flag"].eq(False).all()
    for p in PARAMS:
        assert f"{p}_168h_lower" not in df_full.columns
        assert f"{p}_168h_upper" not in df_full.columns
    # degraded back to the binary required-path behavior
    assert set(df_full["decision"].unique()) <= {"PASS", "FLAG"}


# ---------------------------------------------------------------------------
# REVIEW-aware validation (validate.py)
# ---------------------------------------------------------------------------
def test_validate_lenient_recall_counts_review_as_caught(capsys):
    df = pd.DataFrame({
        "component_id": ["D0", "D1", "D2", "D3", "N0", "N1", "N2", "N3"],
        "is_defect": [True] * 4 + [False] * 4,
        "decision": ["FLAG", "FLAG", "REVIEW", "PASS", "PASS", "PASS", "PASS", "PASS"],
    })
    results = validate(df)
    out = capsys.readouterr().out
    assert results["fn"] == 2                      # D3 passed + D2 deferred (not FLAG)
    assert results["recall"] == pytest.approx(0.5)
    assert results["lenient_recall"] == pytest.approx(0.75)  # FLAG+REVIEW
    assert results["n_review"] == 1
    assert results["deferred_defects"] == 1
    assert "recall, lenient" in out and "REVIEW breakdown" in out
    assert "DEFERRED to REVIEW: 1" in out


def test_validate_binary_frame_unchanged(capsys):
    """No REVIEW rows -> no lenient line, results stay backward compatible."""
    df = pd.DataFrame({
        "component_id": ["D0", "N0", "N1"],
        "is_defect": [True, False, False],
        "decision": ["FLAG", "PASS", "PASS"],
    })
    results = validate(df)
    out = capsys.readouterr().out
    assert results["recall"] == 1.0
    assert "lenient" not in out
    assert "REVIEW breakdown" not in out


def test_validate_compound_deferred_counted(capsys):
    df = pd.DataFrame({
        "component_id": ["D0", "D1"],
        "is_defect": [True, True],
        "defect_case": ["compound", "compound"],
        "decision": ["REVIEW", "FLAG"],
    })
    results = validate(df)
    out = capsys.readouterr().out
    assert results["compound_missed"] == 0
    assert results["compound_deferred"] == 1
    assert "deferred to REVIEW" in out


def test_validate_review_reason_buckets(capsys):
    df = pd.DataFrame({
        "component_id": ["R0", "R1"],
        "is_defect": [False, False],
        "decision": ["REVIEW", "REVIEW"],
        "reason": ["Out-of-distribution inputs (OOD score 0.9) ...",
                   "Wide 168h prediction interval for I_leak ..."],
    })
    results = validate(df)
    out = capsys.readouterr().out
    assert results["n_review"] == 2
    assert "OOD gate: 1" in out and "wide interval: 1" in out
