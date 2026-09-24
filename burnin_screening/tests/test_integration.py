"""Integration tests — main.run_pipeline end to end (merge §4e + contracts)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import (
    MODULE_A_NEW_COLS,
    MODULE_B_NEW_COLS,
    PARAMS,
    generate_data,
    module_a_static,
    module_b_temporal,
    pipeline_main,
)


def test_merge_on_component_id_not_position(df_data):
    """§4e: merge must join on component_id — even when Module B shuffles rows."""
    Y_spec = {p: float(df_data[f"Y_spec_{p}"].iloc[0]) for p in PARAMS}

    df_a = module_a_static.run_module_a(df_data, Y_spec, PARAMS)
    result_b = module_b_temporal.run_module_b(
        df_a, PARAMS, train=True, static_outlier_flag=df_a["static_outlier_flag"]
    )
    df_b, _models, _theta = result_b
    b_cols = module_b_temporal.module_b_new_cols(PARAMS)

    # shuffle Module B's rows before the merge — position-joins would corrupt
    df_b_shuffled = df_b.sample(frac=1.0, random_state=7)
    df_full = df_a.merge(df_b_shuffled[["component_id"] + b_cols], on="component_id")

    check = df_b_shuffled.set_index("component_id")["v_drift"]
    merged_vals = df_full.set_index("component_id")["v_drift"]
    pd.testing.assert_series_equal(merged_vals.sort_index(), check.sort_index(),
                                   check_names=False)


def test_run_pipeline_end_to_end(df_data):
    df_full, results = pipeline_main.run_pipeline(df_data, params=PARAMS, verbose=False)
    # merged frame carries both modules' contract columns + decision columns
    for c in MODULE_A_NEW_COLS + MODULE_B_NEW_COLS + ["decision", "reason"]:
        assert c in df_full.columns, f"missing {c} in merged output"
    assert len(df_full) == len(df_data)
    assert "theta_slope" in results
    assert results["theta_slope"] >= 0.0


def test_run_pipeline_does_not_drop_input_rows_or_cols(df_data):
    df_full, _ = pipeline_main.run_pipeline(df_data, params=PARAMS, verbose=False)
    assert sorted(df_full["component_id"]) == sorted(df_data["component_id"])
    for c in df_data.columns:
        assert c in df_full.columns, f"input column {c} lost in the merge"


def test_theta_slope_derivation_matches_formula(df_data):
    """derive_theta_slope = median + 3*MAD of v_drift over non-static-outliers."""
    Y_spec = {p: float(df_data[f"Y_spec_{p}"].iloc[0]) for p in PARAMS}
    df_a = module_a_static.run_module_a(df_data, Y_spec, PARAMS)
    df_b, _models, theta_b = module_b_temporal.run_module_b(
        df_a, PARAMS, train=True, static_outlier_flag=df_a["static_outlier_flag"]
    )
    b_cols = module_b_temporal.module_b_new_cols(PARAMS)
    df_full = df_a.merge(df_b[["component_id"] + b_cols], on="component_id")

    normal = df_full.loc[~df_full["static_outlier_flag"], "v_drift"]
    med = float(np.median(normal))
    mad = float(np.median(np.abs(normal - med)))
    expected = med + 3.0 * mad
    assert abs(pipeline_main.derive_theta_slope(df_full) - expected) < 1e-12
    # Module B's own theta_slope (returned per Prompt B2 step 4) agrees —
    # both derive median + 3*MAD over the same normal population.
    assert abs(theta_b - expected) < 1e-9


def test_data_generator_schema_matches_contract():
    """Generator output satisfies the frozen §4b schema (Prompt A1)."""
    df = generate_data.generate_dataset(n=80, seed=123)
    required = ["component_id", "lot_id", "N", "T_ambient", "V_stress"]
    for p in PARAMS:
        required += [f"{p}_0h", f"{p}_24h", f"{p}_96h", f"{p}_168h", f"Y_spec_{p}"]
    required += ["is_defect", "defect_case", "defect_param"]
    for c in required:
        assert c in df.columns, f"generator missing schema column {c}"
    # ~5% defects, both failure signatures present
    assert 0.02 <= df["is_defect"].mean() <= 0.10
    assert (df["defect_case"] == "compound").any()
    assert (df["defect_case"] == "single").any()
    # Y_spec constant per parameter
    for p in PARAMS:
        assert df[f"Y_spec_{p}"].nunique() == 1


def test_full_dataset_pipeline_runs():
    """The real 500-row dataset flows through the whole pipeline cleanly."""
    df = generate_data.generate_dataset(n=500, seed=42)
    df_full, results = pipeline_main.run_pipeline(df, params=PARAMS, verbose=False)
    assert len(df_full) == 500
    assert {"PASS", "FLAG"} <= set(df_full["decision"].unique())
    assert 0.0 <= results["recall"] <= 1.0
