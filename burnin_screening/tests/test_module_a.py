"""Tests for module_a_static.run_module_a — Tier 1/2/3 + fusion (contract §4c).

Also covers the 🔵 Empirical Bayes small-lot blending add-on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import MODULE_A_NEW_COLS, PARAMS, module_a_static as mas


# ---------------------------------------------------------------------------
# Contract compliance (§4c)
# ---------------------------------------------------------------------------
def test_contract_columns_added_never_removed(df_a, df_data):
    """All §4c columns exist, and no input column was removed or renamed."""
    for c in MODULE_A_NEW_COLS:
        assert c in df_a.columns, f"missing contract column {c}"
    for c in df_data.columns:
        assert c in df_a.columns, f"input column {c} was dropped"


def test_same_rows_preserved(df_a, df_data):
    """Same row count and component_ids as the input (adds columns only)."""
    assert len(df_a) == len(df_data)
    assert sorted(df_a["component_id"]) == sorted(df_data["component_id"])


def test_no_mutation_of_input(df_data, Y_spec):
    """run_module_a must not modify the caller's dataframe in place."""
    before = df_data.copy(deep=True)
    mas.run_module_a(df_data, Y_spec, PARAMS)
    pd.testing.assert_frame_equal(df_data, before)


# ---------------------------------------------------------------------------
# Tier 1 — hard limits
# ---------------------------------------------------------------------------
def test_tier1_fires_only_on_over_limit_rows(df_data, Y_spec):
    df = mas.run_module_a(df_data, Y_spec, PARAMS)
    for p in PARAMS:
        expected = df[f"{p}_24h"] >= Y_spec[p]
        assert (df[f"{p}_hard_flag"] == expected).all()
    # sanity: the generator's sharp single defects push at least one parameter
    # over its limit in a 200-row sample
    assert df[[f"{p}_hard_flag" for p in PARAMS]].any(axis=1).any(), (
        "no hard-limit flag fires in the synthetic sample"
    )


# ---------------------------------------------------------------------------
# Tier 2 — robust modified Z, computed per lot (not globally)
# ---------------------------------------------------------------------------
def test_tier2_z_within_lot_not_global(df_data, Y_spec):
    """Z-score of a row must equal the modified Z over ITS LOT only."""
    df = mas.run_module_a(df_data, Y_spec, PARAMS)
    p = "I_leak"
    lot_ids = df["lot_id"].unique()
    for lot in lot_ids:
        sub = df.loc[df["lot_id"] == lot, f"{p}_24h"]
        med = np.median(sub)
        mad = np.median(np.abs(sub - med))
        expected = 0.6745 * (sub - med) / max(mad, 1e-6)
        np.testing.assert_allclose(df.loc[sub.index, f"{p}_z"], expected, rtol=1e-12)


def test_tier2_identical_value_in_different_lots(Y_spec):
    """The doc's example: 45uA is anomalous vs a ~10uA lot, unremarkable vs a
lot centred at 45uA — proving the Z is lot-relative, not absolute."""
    # build a 2-lot frame by hand: lot 1 ~10uA, lot 2 ~45uA
    n = 60
    rng = np.random.default_rng(0)
    lots = np.array([1] * 30 + [2] * 30)
    vals = np.concatenate([rng.normal(10, 0.5, 30), rng.normal(45, 0.5, 30)])
    vals[5] = 45.0   # far above lot 1's median -> clear anomaly
    vals[35] = 45.0  # exactly the centre of lot 2 -> z ~ 0
    df = pd.DataFrame({
        "component_id": [f"C{i:03d}" for i in range(n)],
        "lot_id": lots,
        "I_leak_24h": vals,
    })
    out = mas.run_module_a(df, {"I_leak": 50.0}, ["I_leak"])
    assert abs(out.loc[5, "I_leak_z"]) > 3.0
    assert abs(out.loc[35, "I_leak_z"]) < 1.0


def test_tier2_mad_floor_prevents_division_by_zero(Y_spec):
    """Constant-valued lot -> MAD=0 -> floored at 1e-6, no NaN/inf."""
    n = 10
    df = pd.DataFrame({
        "component_id": [f"C{i:03d}" for i in range(n)],
        "lot_id": [1] * n,
        "I_leak_24h": np.full(n, 10.0),
    })
    out = mas.run_module_a(df, {"I_leak": 50.0}, ["I_leak"])
    assert np.isfinite(out["I_leak_z"]).all()


# ---------------------------------------------------------------------------
# Fusion — weighted L2 norm
# ---------------------------------------------------------------------------
def test_fusion_weighted_l2_default_weights(df_a):
    expected = np.sqrt(sum(df_a[f"{p}_z"] ** 2 for p in PARAMS))
    np.testing.assert_allclose(df_a["z_fused"], expected, rtol=1e-12)


def test_fusion_respects_custom_weights(df_data, Y_spec):
    w = {"I_leak": 2.0, "I_ddq": 1.0, "t_pd": 0.5}
    df = mas.run_module_a(df_data, Y_spec, PARAMS, weights=w)
    expected = np.sqrt(sum(w[p] * df[f"{p}_z"] ** 2 for p in PARAMS))
    np.testing.assert_allclose(df["z_fused"], expected, rtol=1e-12)


def test_static_outlier_flag_threshold(df_a):
    assert (df_a["static_outlier_flag"] == (df_a["z_fused"] > 3.5)).all()


def test_fusion_catches_single_parameter_case(df_a):
    """Sharp single-parameter defects (huge per-param z) must fuse over 3.5."""
    big = df_a[(df_a["defect_case"] == "single") & (df_a["static_outlier_flag"])]
    assert len(big) > 0, "no single-parameter defect exceeded the fused threshold"


# ---------------------------------------------------------------------------
# Tier 3 — joint Isolation Forest
# ---------------------------------------------------------------------------
def test_tier3_contract_dtypes_and_ranges(df_a):
    assert df_a["joint_outlier_flag"].dtype == bool
    assert (df_a["joint_anomaly_score"] >= -1.0).all()
    assert df_a.loc[df_a["joint_outlier_flag"], "joint_anomaly_score"].ge(
        df_a["joint_anomaly_score"].quantile(1 - 0.05 - 0.02)
    ).all(), "flagged rows should sit in the top anomaly-score band"


def test_tier3_flags_some_single_defects(df_a):
    """Single defects are extreme in one coordinate — Tier 3 should see many."""
    flagged = df_a[(df_a["defect_case"] == "single") & df_a["joint_outlier_flag"]]
    assert len(flagged) > 0


# ---------------------------------------------------------------------------
# Missing-column guards
# ---------------------------------------------------------------------------
def test_missing_column_raises(df_data, Y_spec):
    bad = df_data.drop(columns=["I_leak_24h"])
    try:
        mas.run_module_a(bad, Y_spec, PARAMS)
    except KeyError as e:
        assert "I_leak_24h" in str(e)
    else:
        raise AssertionError("expected KeyError for missing I_leak_24h")

    try:
        mas.run_module_a(df_data.drop(columns=["lot_id"]), Y_spec, PARAMS)
    except KeyError as e:
        assert "lot_id" in str(e)
    else:
        raise AssertionError("expected KeyError for missing lot_id")


# ---------------------------------------------------------------------------
# 🔵 Add-on — Empirical Bayes small-lot blending
# ---------------------------------------------------------------------------
def test_small_lot_blending_off_matches_required_path(df_data, Y_spec):
    """Default (off) must be byte-identical to the plain 🟢 call."""
    plain = mas.run_module_a(df_data, Y_spec, PARAMS)
    off = mas.run_module_a(df_data, Y_spec, PARAMS, small_lot_blending=False)
    pd.testing.assert_frame_equal(plain, off)


def test_small_lot_blending_on_differs_and_stays_finite(df_data, Y_spec):
    """Blending must change small-lot rows (shrinkage) and produce finite Zs."""
    plain = mas.run_module_a(df_data, Y_spec, PARAMS)
    blend = mas.run_module_a(df_data, Y_spec, PARAMS, small_lot_blending=True)
    assert np.isfinite(blend["I_leak_z"]).all()
    small_lots = blend.groupby("lot_id")["N"].first()
    small_ids = small_lots[small_lots < mas.SMALL_LOT_N].index
    assert len(small_ids) > 0, "generator should include small lots"
    for lot in small_ids:
        rows = blend["lot_id"] == lot
        # shrinkage pulls the small lot toward the global prior: values differ
        assert not np.allclose(
            blend.loc[rows, "I_leak_z"], plain.loc[rows, "I_leak_z"]
        ), f"small lot {lot} Zs unchanged — blending did not engage"
    # large lots must keep pure lot statistics (identical to the plain path)
    big_lots = small_lots[small_lots >= mas.SMALL_LOT_N].index
    for lot in big_lots:
        rows = blend["lot_id"] == lot
        np.testing.assert_allclose(
            blend.loc[rows, "I_leak_z"], plain.loc[rows, "I_leak_z"], rtol=1e-12
        )


def test_small_lot_blending_decision_columns_unchanged_names(df_data, Y_spec):
    blend = mas.run_module_a(df_data, Y_spec, PARAMS, small_lot_blending=True)
    for c in MODULE_A_NEW_COLS:
        assert c in blend.columns
