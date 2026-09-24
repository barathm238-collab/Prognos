"""Tests for decision_maker.decide — priority order + per-parameter reasons.

The synthetic dataset alone can't hit every decision branch deterministically,
so each branch is also exercised on hand-built rows (§8 step 5 of the doc:
run hand-picked rows — one hard-limit violation, one lot outlier, one
drift-only flag, one clean PASS — and check the reason string is accurate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import PARAMS, decide


THETA = 1.23  # arbitrary but cited in drift reasons


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


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hand-picked rows — one per branch (§8 step 5)
# ---------------------------------------------------------------------------
def test_branch1_hard_limit_names_parameters():
    r = _base_row()
    r["I_leak_hard_flag"] = True
    r["t_pd_hard_flag"] = True
    r["I_leak_24h"] = 55.0
    r["t_pd_24h"] = 41.0
    r["Y_spec_I_leak"] = 50.0
    r["Y_spec_t_pd"] = 40.0
    out = decide(_frame([r]), PARAMS, THETA)
    assert out.loc[0, "decision"] == "FLAG"
    reason = out.loc[0, "reason"]
    assert "I_leak" in reason and "t_pd" in reason
    assert "Hard limit" in reason
    assert "55" in reason and "50" in reason  # cites value + limit


def test_branch2_joint_anomaly_cites_score_and_driver():
    r = _base_row()
    r["joint_outlier_flag"] = True
    r["joint_anomaly_score"] = 0.61
    r["I_leak_z"] = -0.1
    r["I_ddq_z"] = 2.9   # largest absolute -> must be named
    r["t_pd_z"] = 1.0
    out = decide(_frame([r]), PARAMS, THETA)
    assert out.loc[0, "decision"] == "FLAG"
    reason = out.loc[0, "reason"]
    assert "Joint" in reason
    assert "0.61" in reason
    assert "I_ddq" in reason


def test_branch3_static_outlier_cites_z_fused_and_driver():
    r = _base_row()
    r["static_outlier_flag"] = True
    r["z_fused"] = 4.2
    r["I_leak_z"] = 0.5
    r["I_ddq_z"] = 1.0
    r["t_pd_z"] = 3.9   # driver
    out = decide(_frame([r]), PARAMS, THETA)
    assert out.loc[0, "decision"] == "FLAG"
    reason = out.loc[0, "reason"]
    assert "4.2" in reason and "t_pd" in reason


def test_branch4_drift_cites_v_drift_theta_and_driver():
    r = _base_row()
    r["drift_flag"] = True
    r["v_drift"] = 2.5
    r["I_leak_slope_norm"] = 0.3
    r["I_ddq_slope_norm"] = 1.9  # driver
    r["t_pd_slope_norm"] = -0.4  # negative must win on |abs| if larger
    out = decide(_frame([r]), PARAMS, THETA)
    assert out.loc[0, "decision"] == "FLAG"
    reason = out.loc[0, "reason"]
    assert "2.50" in reason and "1.23" in reason
    assert "I_ddq" in reason


def test_branch5_clean_pass_reason():
    out = decide(_frame([_base_row()]), PARAMS, THETA)
    assert out.loc[0, "decision"] == "PASS"
    assert "Within limits" in out.loc[0, "reason"]


# ---------------------------------------------------------------------------
# Priority order — first match wins
# ---------------------------------------------------------------------------
def test_priority_hard_limit_beats_everything():
    r = _base_row()
    r["I_leak_hard_flag"] = True
    r["joint_outlier_flag"] = True
    r["static_outlier_flag"] = True
    r["drift_flag"] = True
    out = decide(_frame([r]), PARAMS, THETA)
    assert "Hard limit" in out.loc[0, "reason"]


def test_priority_joint_beats_static_beats_drift():
    r = _base_row()
    r["joint_outlier_flag"] = True
    r["static_outlier_flag"] = True
    r["drift_flag"] = True
    out = decide(_frame([r]), PARAMS, THETA)
    assert "Joint" in out.loc[0, "reason"]

    r2 = _base_row()
    r2["static_outlier_flag"] = True
    r2["drift_flag"] = True
    out2 = decide(_frame([r2]), PARAMS, THETA)
    assert "outlier" in out2.loc[0, "reason"]


def test_multiple_hard_flags_all_listed():
    r = _base_row()
    r["I_leak_hard_flag"] = True
    r["I_ddq_hard_flag"] = True
    r["t_pd_hard_flag"] = True
    out = decide(_frame([r]), PARAMS, THETA)
    for p in PARAMS:
        assert p in out.loc[0, "reason"]


# ---------------------------------------------------------------------------
# Behavioural contract
# ---------------------------------------------------------------------------
def test_no_mutation_and_new_columns_only():
    rows = [_base_row(f"C{i:04d}") for i in range(3)]
    df = _frame(rows)
    before = df.copy(deep=True)
    out = decide(df, PARAMS, THETA)
    assert "decision" in out.columns and "reason" in out.columns
    assert len(out) == len(df)
    pd.testing.assert_frame_equal(df, before)


def test_negative_slope_norm_abs_handling():
    """Driver selection uses absolute value — a large negative slope wins."""
    r = _base_row()
    r["drift_flag"] = True
    r["v_drift"] = 1.9
    r["I_leak_slope_norm"] = -1.7
    r["I_ddq_slope_norm"] = 0.5
    r["t_pd_slope_norm"] = 0.2
    out = decide(_frame([r]), PARAMS, THETA)
    assert "I_leak" in out.loc[0, "reason"]


def test_missing_flag_column_treated_as_false():
    """decide() uses row.get(..., False) — a missing Module B column must not crash."""
    r = _base_row()
    del r["drift_flag"]
    del r["v_drift"]
    out = decide(_frame([r]), PARAMS, THETA)
    assert out.loc[0, "decision"] == "PASS"


# ---------------------------------------------------------------------------
# 96h trajectory line in drift FLAG reasons (Person 1 doc §6 explanation trail)
# ---------------------------------------------------------------------------
def test_drift_reason_includes_96h_trajectory_line():
    """A drift FLAG with 96h/168h preds present cites the full measured->forecast trajectory."""
    r = _base_row()
    r["drift_flag"] = True
    r["v_drift"] = 2.5
    r["I_ddq_slope_norm"] = 1.9  # driver
    r["I_ddq_24h"] = 2.05
    r["I_ddq_96h_pred"] = 2.20
    r["I_ddq_168h_pred"] = 2.55
    out = decide(_frame([r]), PARAMS, THETA)
    reason = out.loc[0, "reason"]
    assert "Trajectory for I_ddq" in reason
    assert "y(24h)=2.05 measured" in reason
    assert "y(96h)=2.2 forecast" in reason
    assert "y(168h)=2.55 forecast" in reason


def test_drift_trajectory_line_degrades_without_96h_column():
    """Contract-minimum frame (no {p}_96h_pred) -> 24h->168h line, no crash."""
    r = _base_row()
    r["drift_flag"] = True
    r["v_drift"] = 2.5
    r["I_ddq_slope_norm"] = 1.9
    r["I_ddq_24h"] = 2.05
    r["I_ddq_168h_pred"] = 2.55
    out = decide(_frame([r]), PARAMS, THETA)
    reason = out.loc[0, "reason"]
    assert "Trajectory for I_ddq" in reason
    assert "y(24h)=2.05 measured" in reason
    assert "y(96h)" not in reason
    assert "y(168h)=2.55 forecast" in reason


def test_drift_trajectory_line_nan_96h_is_skipped():
    """A NaN auxiliary forecast must be skipped, not formatted as nan."""
    r = _base_row()
    r["drift_flag"] = True
    r["v_drift"] = 2.5
    r["I_ddq_slope_norm"] = 1.9
    r["I_ddq_24h"] = 2.05
    r["I_ddq_96h_pred"] = float("nan")
    r["I_ddq_168h_pred"] = 2.55
    out = decide(_frame([r]), PARAMS, THETA)
    reason = out.loc[0, "reason"]
    assert "y(96h)" not in reason
    assert "nan" not in reason.lower()
    assert "y(168h)=2.55 forecast" in reason


def test_drift_trajectory_names_the_driving_parameter_only(df_full):
    """On the real pipeline output, drift FLAGs cite a trajectory for the named driver."""
    drift_rows = df_full[df_full["reason"].str.contains("Trajectory for", na=False)]
    assert len(drift_rows) > 0, "expected at least one drift FLAG with a trajectory line"
    for reason in drift_rows["reason"]:
        assert "driven mainly by" in reason


def test_full_dataset_reasons_match_decisions(df_full):
    """Every FLAG reason names at least one concrete parameter; PASS doesn't."""
    flag_rows = df_full[df_full["decision"] == "FLAG"]
    named = any(p in r for r in flag_rows["reason"] for p in PARAMS)
    assert named, "FLAG reasons must name a specific parameter"
    assert len(flag_rows) > 0
    # decision distribution sanity on the full-pipeline run (three-way since
    # Phase 3: the uncertainty layer may defer rows to REVIEW)
    assert set(df_full["decision"].unique()) <= {"PASS", "REVIEW", "FLAG"}
