"""main.py — integration entrypoint. Person 1 owns this file.

Pipeline per the architecture doc:
  1. Ensure data/burnin_data.csv exists (generate if missing).
  2. Call run_module_a() and run_module_b() on the SAME input dataframe
     (with_intervals=True — Prompt B3 — so the Decision Maker has conformal
     intervals to reason over; the required path is unaffected when callers
     omit them).
  3. Merge on component_id (never row position — §4e merge rule).
  4. Derive theta_slope from Module B's v_drift over the normal population.
  5. 🔵 OOD gate (Prompt B4): fit on the raw regression features and attach
     ood_score/ood_flag to the merged frame — one OOD verdict per component
     (Module B's documented combined-feature choice).
  6. 🔵 Progressive refinement (Prompt B5): {p}_96h_actual / {p}_168h_actual
     columns, when present, override the corresponding predictions and the
     drift columns are recomputed — a no-op unless actuals exist.
  7. Run the Decision Maker -> decision + reason columns.
  8. Validate: confusion matrix + REVIEW breakdown + compound-defect report.
  9. Save the merged output to data/pipeline_output.csv.

Module ownership: this file calls both modules' single public functions and
reads only their documented output columns (contracts §4c/§4d; §4f for the
decision columns).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as `python main.py` from anywhere.
sys.path.insert(0, str(Path(__file__).parent))

import module_a_static
import module_b_temporal
from decision_maker import decide
from validate import validate

PARAMS = ["I_leak", "I_ddq", "t_pd"]
DATA_PATH = Path(__file__).parent / "data" / "burnin_data.csv"
OUTPUT_PATH = Path(__file__).parent / "data" / "pipeline_output.csv"


def load_data(path: Path, n: int, seed: int) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows from {path}")
        return df
    print(f"{path} not found — generating synthetic data (n={n}, seed={seed})...")
    from data.generate_data import generate_dataset
    df = generate_dataset(n=n, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")
    return df


def derive_theta_slope(df: pd.DataFrame, theta_k: float = 3.0) -> float:
    """Safety-slope threshold from the normal population (per Prompt B2 step 4).

    Computed here from Module B's v_drift over rows that are NOT static
    outliers, using median + theta_k * MAD (Prompt B2's literal multiplier is
    3.0 — the default; run_pipeline passes the team-tuned value through so
    the fallback path always agrees with Module B's own theta_slope). If
    Module B returns theta_slope alongside (df, models), main() prefers that
    value (see run_pipeline) — this is only the contract-shape fallback.
    """
    normal = df.loc[~df["static_outlier_flag"], "v_drift"]
    med = float(np.median(normal))
    mad = float(np.median(np.abs(normal - med)))
    return med + theta_k * mad


def derive_theta_width(df: pd.DataFrame, params: list = PARAMS) -> float:
    """🔵 REVIEW threshold for conformal-interval width (Phase 3).

    median + 3*MAD of the per-row maximum relative interval width over the
    NORMAL population (static_outlier_flag == False) — deliberately the same
    derivation philosophy as theta_slope (Prompt B2 step 4): a threshold
    derived from the given data, not a hardcoded constant.

    Returns 0.0 (inert: the width branch only fires on width > 0.0, and
    valid widths are >= 0) when interval columns are missing or the normal
    population yields no finite widths — i.e. the Decision Maker's width
    REVIEW branch can only refine, never create, uncertainty.
    """
    from decision_maker import max_relative_width

    widths = max_relative_width(df, params)
    if "static_outlier_flag" in df.columns:
        widths = widths[~df["static_outlier_flag"]]
    finite = widths.dropna().to_numpy(dtype=float)
    if finite.size == 0:
        return 0.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    return med + 3.0 * mad


def apply_progressive_refinement(df: pd.DataFrame, params: list = PARAMS,
                                 theta_k: float = 2.4,
                                 theta_by_lot: dict | None = None) -> pd.DataFrame:
    """🔵 Progressive refinement layer (Prompt B5 / architecture guide add-on).

    For each parameter and each refinement horizon, prefer a real measured
    checkpoint over Module B's prediction — via Module B's own
    resolve_checkpoint() helper, so the preference rule lives in exactly one
    place — then recompute the drift features from whichever mix of
    measured+predicted values now exists (the architecture guide's
    "downstream decisions recomputed from whichever mix of real+predicted
    data currently exists").

    Horizon logic (per row): 168h and 96h actuals each replace their
    prediction as a slope endpoint where measured; rows with only a 96h
    actual use the measured 24h->96h window (/72), rows with a 168h actual
    the 24h->168h window (/144), and untouched rows keep their original
    slope. Column names follow Module B's Prompt B5 convention:
    {p}_{h}h_actual.

    Refined values land in {p}_96h_pred_refined / {p}_168h_pred_refined — the
    graded {p}_168h_pred column is deliberately NOT overwritten, so the MAE
    scoring record stays comparable run over run; refinement changes
    decisions, never the graded metric.

    A no-op (returns df with identical values) unless at least one
    {p}_{h}h_actual column exists with a finite value — the graded pipeline
    frame carries no actuals, so this can never perturb the required path.
    """
    touched = False
    for p in params:
        y96 = df[f"{p}_96h_pred"].to_numpy(dtype=float)
        y168 = df[f"{p}_168h_pred"].to_numpy(dtype=float)
        y24 = df[f"{p}_24h"].to_numpy(dtype=float)

        # NaN/None actuals resolve to "predicted" per finite check — the
        # vectorized form of Module B's resolve_checkpoint preference rule.
        y96a = (pd.to_numeric(df[f"{p}_96h_actual"], errors="coerce").to_numpy(dtype=float)
                if f"{p}_96h_actual" in df.columns else None)
        y168a = (pd.to_numeric(df[f"{p}_168h_actual"], errors="coerce").to_numpy(dtype=float)
                 if f"{p}_168h_actual" in df.columns else None)

        p_touched = ((y96a is not None and np.isfinite(y96a).any())
                     or (y168a is not None and np.isfinite(y168a).any()))
        if not p_touched:
            continue
        touched = True

        if y96a is not None:
            y96 = np.where(np.isfinite(y96a), y96a, y96)
        if y168a is not None:
            y168 = np.where(np.isfinite(y168a), y168a, y168)

        # Slope endpoints follow the architecture guide: recompute from
        # whichever mix of real+predicted data now exists — PER ROW, so
        # unrefined rows keep their original 24h->168h prediction slope.
        #   - 168h actual present   -> (y168_refined - y24) / 144
        #   - 96h actual only       -> the measured 24h->96h window / 72
        #   - neither (untouched)   -> original slope, unchanged
        slope = df[f"{p}_slope"].to_numpy(dtype=float)
        finite168 = np.isfinite(y168a) if y168a is not None else np.zeros(len(df), dtype=bool)
        finite96 = np.isfinite(y96a) if y96a is not None else np.zeros(len(df), dtype=bool)
        slope = np.where(finite96 & ~finite168, (y96 - y24) / (96.0 - 24.0), slope)
        slope = np.where(finite168, (y168 - y24) / (168.0 - 24.0), slope)
        df[f"{p}_slope"] = slope
        df[f"{p}_96h_pred_refined"] = y96
        df[f"{p}_168h_pred_refined"] = y168

    if not touched:
        return df

    # Re-derive the drift features exactly as Module B does (same normal
    # population, same std/MAD floors), so refined decisions are computed
    # under the same rules as the required path.
    mask = ((~df["static_outlier_flag"]).to_numpy(dtype=bool)
            if "static_outlier_flag" in df.columns else np.ones(len(df), dtype=bool))
    for p in params:
        sd = float(df.loc[mask, f"{p}_slope"].std())
        df[f"{p}_slope_norm"] = df[f"{p}_slope"] / (sd + 1e-6)
    weights = {p: 1.0 for p in params}
    df["v_drift"] = np.sqrt(sum(weights[p] * df[f"{p}_slope_norm"] ** 2 for p in params))
    if theta_by_lot:
        # 🔵 §8b: recompute refined drift flags under the SAME per-lot blended
        # theta Module B used (rows in lots missing from the dict fall back to
        # the median across the dict — defensive; every row has a lot_id).
        theta_map = pd.Series(dict(theta_by_lot))
        row_theta = df["lot_id"].map(theta_map)
        fallback = float(np.median(list(theta_by_lot.values())))
        row_theta = row_theta.fillna(fallback).to_numpy(dtype=float)
        df["drift_flag"] = df["v_drift"].to_numpy(dtype=float) > row_theta
        return df
    normal_v = df.loc[mask, "v_drift"].to_numpy(dtype=float)
    med = float(np.median(normal_v))
    mad = float(np.median(np.abs(normal_v - med)))
    df["drift_flag"] = df["v_drift"] > (med + theta_k * mad)
    return df


def run_pipeline(df_input: pd.DataFrame, params: list = PARAMS, verbose: bool = True,
                 with_intervals: bool = True, with_ood: bool = True,
                 with_progressive_refinement: bool = True, theta_k: float = 2.4,
                 theta_blended: bool = False):
    """Run the full pipeline; returns (df_full, results_dict).

    The 🔵 add-on switches (Phase 3) all default True for the demo run and
    degrade gracefully: with_intervals=False removes the width-REVIEW signal,
    with_ood=False removes the OOD REVIEW signal, and progressive refinement
    is inert unless the frame carries {p}_{h}h_actual columns. Every
    combination still produces a valid decision frame.

        theta_k: TEAM-TUNED MAD multiplier for Module B's theta_slope gate
    (default 2.4 for this demo, see run_module_b's docstring for the joint
    tuning story). Pass 3.0 for Prompt B2's literal spec value.
        theta_blended: 🔵 Phase 1 review §8b — per-lot Empirical-Bayes-blended
    theta_slope (Module A's small-lot pattern applied to Module B's gate).
    Default False: the required path's single global theta_slope is unchanged.
    When True, Module B returns dict[lot_id -> theta] and drift_flag is
    computed per lot inside Module B; this function just carries the dict
    through to progressive refinement so refined flags recompute under the
    SAME per-lot rules.
    """
    # Y_spec from the data itself (columns Y_spec_{p} are part of the frozen schema)
    Y_spec = {p: float(df_input[f"Y_spec_{p}"].iloc[0]) for p in params}

    # --- Module A (Person 1) and Module B (Person 2) on the SAME input frame ---
    df_a = module_a_static.run_module_a(df_input, Y_spec, params)
    # Prompt B2: Module B normalizes slopes and derives theta_slope over the
    # normal population (static_outlier_flag == False) — pass Module A's flag
    # via its optional argument rather than changing the §4e input frame.
    # with_intervals=True (Prompt B3): the MAPIE conformal wrapper replaces
    # the plain regressor for the 168h target; {p}_168h_lower/upper ride
    # along in df_b and are consumed by the Decision Maker's width-REVIEW.
    result_b = module_b_temporal.run_module_b(
        df_input, params, train=True, static_outlier_flag=df_a["static_outlier_flag"],
        with_intervals=with_intervals, theta_k=theta_k, theta_blended=theta_blended,
    )

    # Contract §4d says run_module_b returns (df, models); Prompt B2 allows
    # theta_slope as a third element — accept both shapes. With theta_blended
    # (§8b) the third element is dict[lot_id -> theta] instead of a float.
    if len(result_b) == 3:
        df_b, models_b, theta_b = result_b
    else:
        df_b, models_b = result_b
        theta_b = None

    # --- Merge on component_id, never row position (§4e) ---
    b_cols = module_b_temporal.module_b_new_cols(params, with_intervals=with_intervals)
    df_full = df_a.merge(df_b[["component_id"] + b_cols], on="component_id")

    # --- Safety-slope threshold (Module B's own value preferred; the
    #     fallback derives median + theta_k*MAD over the same population).
    #     §8b: dict => per-lot blended theta — use the GLOBAL-lot value (or
    #     the median across lots) as the scalar cited in drift reasons. ---
    if isinstance(theta_b, dict):
        theta_slope = theta_b.get("GLOBAL")
        if theta_slope is None:
            theta_slope = float(np.median(list(theta_b.values())))
    else:
        theta_slope = theta_b if theta_b is not None else derive_theta_slope(df_full, theta_k)

    # --- 🔵 OOD gate (Prompt B4) — fit on raw regression inputs, one verdict
    #     per component via Module B's documented combined feature set ---
    if with_ood:
        from module_b_temporal import build_ood_features, fit_ood_gate, score_ood

        ood_X = build_ood_features(df_input, params)
        detector = fit_ood_gate(ood_X)
        df_full["ood_score"], df_full["ood_flag"] = score_ood(detector, ood_X)
    else:
        df_full["ood_score"] = np.nan
        df_full["ood_flag"] = False

    # --- 🔵 REVIEW threshold for interval width — same derivation philosophy
    #     as theta_slope (median + 3*MAD over the normal population) ---
    theta_width = derive_theta_width(df_full, params) if with_intervals else None

    # --- 🔵 Progressive refinement (Prompt B5): no-op without actual columns ---
    if with_progressive_refinement:
        df_full = apply_progressive_refinement(
            df_full, params, theta_k=theta_k,
            theta_by_lot=theta_b if isinstance(theta_b, dict) else None,
        )

    # --- Decision Maker + validation ---
    # Y_spec is threaded so the predicted-168h-breach FLAG branch (Phase 1
    # review §8a) is active in the full pipeline; decide() degrades to the
    # previous 7-branch behavior when called without it.
    df_full = decide(df_full, params, theta_slope, theta_width=theta_width, Y_spec=Y_spec)
    results = validate(df_full)
    results["theta_slope"] = theta_slope
    results["theta_width"] = theta_width

    if verbose:
        print(f"\ntheta_slope (safety slope): {theta_slope:.4f}")
        if theta_width is not None:
            print(f"theta_width (interval-width REVIEW threshold): {theta_width:.4f}")
        print(f"\nDecision distribution:\n{df_full['decision'].value_counts().to_string()}")
        print("\nSample FLAG rows:")
        flag_rows = df_full[df_full["decision"] == "FLAG"].head(5)
        for _, r in flag_rows.iterrows():
            print(f"  {r['component_id']}: {r['reason']}")
        review_rows = df_full[df_full["decision"] == "REVIEW"].head(5)
        if len(review_rows):
            print("\nSample REVIEW rows:")
            for _, r in review_rows.iterrows():
                print(f"  {r['component_id']}: {r['reason']}")
        print("\nSample PASS row:")
        pass_row = df_full[df_full["decision"] == "PASS"].head(1)
        for _, r in pass_row.iterrows():
            print(f"  {r['component_id']}: {r['reason']}")

    return df_full, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Burn-in predictive screening pipeline")
    parser.add_argument("--n", type=int, default=500, help="components (used only when generating data)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (used only when generating data)")
    parser.add_argument("--regen", action="store_true", help="regenerate burnin_data.csv even if it exists")
    parser.add_argument(
        "--theta-k", type=float, default=2.4,
        help="MAD multiplier for the safety-slope gate (team-tuned 2.4; "
             "Prompt B2's literal spec value is 3.0)",
    )
    args = parser.parse_args()

    if args.regen and DATA_PATH.exists():
        DATA_PATH.unlink()

    df_input = load_data(DATA_PATH, n=args.n, seed=args.seed)
    df_full, results = run_pipeline(df_input, theta_k=args.theta_k)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved merged output ({len(df_full)} rows) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
