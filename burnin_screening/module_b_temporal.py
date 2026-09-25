"""Module B — Temporal Drift Prediction. Person 2 owns this file.

Implementation note: originally written by Person 1 as a stand-in (RESUME.md
resume step 1 sanctions this: "implement Prompts B1/B2 if standing in"),
strictly per Prompts B1 + B2 of the implementation docs. Prompts B3/B4/B5
(🔵 MAPIE conformal intervals, OOD gate, progressive refinement) are now
implemented below as opt-in add-on paths — the default call (plain point
predictions) is byte-for-byte the required 🟢 path. Person 2: review, tune,
and take ownership — the function signature and output column names below
are the frozen §4d contract and must not change.

Implemented behavior (Prompt B1 + B2):
  For each parameter p:
    1. {p}_delta = {p}_24h - {p}_0h
    2. X = [{p}_0h, {p}_24h, {p}_delta, T_ambient, V_stress]
    3. train=True: fit XGBRegressor (n_estimators=200, max_depth=4,
       learning_rate=0.1) as MultiOutputRegressor -> [{p}_96h, {p}_168h],
       print the 168h MAE separately (the graded metric)
    4. {p}_96h_pred, {p}_168h_pred from predictions on full X
    5. {p}_slope = ({p}_168h_pred - {p}_24h) / (168 - 24)
    6. {p}_slope_norm = {p}_slope / std(slope over rows where
       static_outlier_flag is False) — the mask is taken from the optional
       argument if given, else from a same-named column in df, else all rows
    7. v_drift = sqrt(sum weight_p * {p}_slope_norm^2)  (weights default 1.0,
       same convention as Module A's z_fused)
    8. theta_slope = median(v_drift over the same normal subset)
       + 3 * mad(v_drift over that subset)
    9. drift_flag = v_drift > theta_slope
   10. with_intervals=True (🔵 Prompt B3): additionally fits a
       CrossConformalRegressor (method="plus", cv=5, confidence_level=0.95 —
       the MAPIE>=1.0 successor of MapieRegressor) per parameter on the SAME
       train split, 168h target specifically; {p}_168h_pred is then taken
       from the MAPIE-wrapped regressor and {p}_168h_lower/{p}_168h_upper
       carry the 95% conformal interval. Default off — the required 🟢 path
       is unchanged. (96h stays the plain MultiOutput prediction; B3 covers
       the 168h target only.)

Output contract (§4d — frozen, do not rename):
  run_module_b(df, params, train=True) returns (df, models, theta_slope)
  where df has these columns ADDED (theta_slope may be a float or, with
  theta_blended=True, a dict[lot_id -> float] — see the theta_blended note):
    {p}_96h_pred     float — auxiliary forecast (🔵, never validated against)
    {p}_168h_pred    float — required forecast (🟢, graded via MAE)
    {p}_168h_lower   float — MAPIE 95% lower bound (🔵, with_intervals=True only)
    {p}_168h_upper   float — MAPIE 95% upper bound (🔵, with_intervals=True only)
    {p}_slope        float — per parameter
    {p}_slope_norm   float — normalized by normal-population slope std
    v_drift           float — fused intra-component drift vector (weighted L2)
    drift_flag        bool  — v_drift > theta_slope
  models: dict[param -> fitted MultiOutputRegressor], reusable via train=False.
  theta_slope: the safety-slope threshold, logged/reused by the Decision Maker.

🔵 Add-ons built here (opt-in — the required 🟢 path is unaffected by default):
  Prompt B3 — MAPIE conformal intervals via with_intervals=True (step 10 above).
  Prompt B4 — fit_ood_gate(X_train) fits an IsolationForest on the RAW
              regression input features (NOT the Z-scores — those belong to
              Module A's Tier 3); score_ood(detector, X) returns
              (ood_score, ood_flag) with score = -score_samples (higher =
              more anomalous, same convention as Module A Tier 3).
              Documented choice: the pipeline wires these on the COMBINED
              feature set (build_ood_features) so each component gets ONE
              OOD verdict matching its component-level decision;
              per-parameter use also works (pass one parameter's features).
  Prompt B5 — resolve_checkpoint(row, param, horizon, predicted_value):
              standalone progressive-refinement helper, deliberately NOT
              wired into run_module_b (per the prompt) so it cannot
              destabilize the required path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mapie.regression import CrossConformalRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# MAD floor, mirroring Module A's Tier 2 — keeps theta_slope finite when the
# normal population's v_drift is degenerate (e.g. all-identical slopes).
MAD_FLOOR = 1e-6
# Std floor for slope normalization (architecture guide Phase 3 snippet).
STD_FLOOR = 1e-6
# MAPIE conformal level (Prompt B3: predict(X, alpha=0.05) == a 95% interval).
MAPIE_CONFIDENCE_LEVEL = 0.95
# Blending constant for the 🔵 per-lot theta (§8b) — same N/(N+K) shape as
# Module A's small-lot blending (K=30 per the doc).
SMALL_LOT_K = 30


def _normal_mask(df: pd.DataFrame, static_outlier_flag) -> np.ndarray:
    """Boolean mask of the 'normal' population for normalization/threshold.

    Precedence (Prompt B2 step 2): explicit argument > same-named column in
    df > all rows. The mask marks rows where static_outlier_flag is FALSE.
    """
    if static_outlier_flag is not None:
        return ~np.asarray(static_outlier_flag, dtype=bool)
    if "static_outlier_flag" in df.columns:
        return ~df["static_outlier_flag"].to_numpy(dtype=bool)
    return np.ones(len(df), dtype=bool)


def run_module_b(
    df: pd.DataFrame,
    params: list,
    train: bool = True,
    weights: dict | None = None,
    static_outlier_flag=None,
    models: dict | None = None,
    with_intervals: bool = False,
    theta_k: float = 3.0,
    theta_blended: bool = False,
):
    """Run the temporal prediction stack. See module docstring for the exact
    output columns added (contract §4d) and the return signature.

    Args:
        df: input dataframe with the shared schema (§4b). For train=True it
            must also carry the training-only columns {p}_96h/{p}_168h.
            Never modified in place — a copy with new columns is returned.
        params: list of parameter names, e.g. ["I_leak", "I_ddq", "t_pd"].
        train: fit fresh regressors (True) or score with pre-fitted `models`.
        weights: optional dict[param -> float] L2 fusion weights; defaults to
            1.0 (same convention as Module A's z_fused).
        static_outlier_flag: optional boolean array/Series marking Module A's
            static outliers; rows marked False (or absent -> all rows) form
            the "normal" population used for slope normalization and
            theta_slope. Accepted as the optional argument Prompt B2 asks for.
        models: pre-fitted dict[param -> MultiOutputRegressor] for train=False.
        with_intervals: 🔵 Prompt B3 — also produce MAPIE conformal intervals
            for the 168h target ({p}_168h_lower/{p}_168h_upper) and take
            {p}_168h_pred from the MAPIE-wrapped regressor. Only supported
            with train=True (the conformal model is fitted here). Default
            False so the required 🟢 path is byte-identical to B1+B2.
        theta_blended: 🔵 Phase 1 review §8b — derive theta_slope PER LOT
            using the same Empirical Bayes blending pattern as Module A's
            small-lot Z (w = N/(N+30)): lot_theta = w * (lot median +
            theta_k*lot MAD) + (1-w) * (global median + theta_k*global MAD).
            Returns dict[lot_id -> float] instead of a float; drift_flag
            uses each row's own lot's blended theta. Small lots shrink
            toward the global prior, so their gate stops being noise-driven.
            Default False — the required 🟢 path (single global theta_slope)
            is unchanged. Intended for real burn-in data where small lots
            are common; on data without small lots it degenerates to ~the
            global value (w -> 1 for large lots).
        theta_k: MAD multiplier in theta_slope = median + theta_k * MAD of the
            normal population's v_drift. Prompt B2's literal spec is 3.0 (the
            default — required path unchanged). Tunable by TEAM DECISION for
            false-negative minimization: a joint grid over Module A's
            z_threshold / contamination and this multiplier showed the drift
            gate is the only effective FN lever on the compound-defect case
            (a margin-of-0.25 defect at v_drift's 93.5th normal percentile);
            main.py's demo entrypoint ships theta_k=2.4 (documented in the
            person docs, Prompt B2 note). Lower values trade FP for FN.

    Returns:
        (df, models, theta_slope)
    """
    df = df.copy()
    if weights is None:
        weights = {p: 1.0 for p in params}
    if models is None:
        models = {}
    mask = _normal_mask(df, static_outlier_flag)

    for p in params:
        need = [f"{p}_0h", f"{p}_24h", "T_ambient", "V_stress"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise KeyError(f"run_module_b: input df is missing required columns: {missing}")

        # ---- features (Prompt B1 step 1-2) ----
        df[f"{p}_delta"] = df[f"{p}_24h"] - df[f"{p}_0h"]
        X = df[[f"{p}_0h", f"{p}_24h", f"{p}_delta", "T_ambient", "V_stress"]]

        if train:
            # ---- fit multi-output regressor (Prompt B1 step 3) ----
            train_only = [c for c in (f"{p}_96h", f"{p}_168h") if c not in df.columns]
            if train_only:
                raise KeyError(
                    f"run_module_b(train=True) needs training-only columns {train_only} "
                    f"(schema §4b) — present in data/burnin_data.csv, absent at inference"
                )
            y_multi = df[[f"{p}_96h", f"{p}_168h"]]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_multi, test_size=0.2, random_state=0
            )
            m = MultiOutputRegressor(
                XGBRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.1, random_state=0
                )
            )
            m.fit(X_tr, y_tr)
            models[p] = m

            # ---- predictions on the full X (Prompt B1 step 4) ----
            preds = m.predict(X)
            df[f"{p}_96h_pred"] = preds[:, 0]
            df[f"{p}_168h_pred"] = preds[:, 1]

            # 168h MAE reported separately — it is the graded metric
            mae_168 = mean_absolute_error(y_te.iloc[:, 1], m.predict(X_te)[:, 1])
            mae_96 = mean_absolute_error(y_te.iloc[:, 0], m.predict(X_te)[:, 0])
            print(f"  [{p}] 168h MAE (graded): {mae_168:.4f}   (96h auxiliary MAE: {mae_96:.4f})")

            if with_intervals:
                # ---- 🔵 Prompt B3: MAPIE conformal intervals, 168h target ----
                # MapieRegressor(method="plus", cv=5) was removed in MAPIE >=
                # 1.0; CrossConformalRegressor(method="plus", cv=5) is its
                # direct successor (same cross-fitted "plus" logic).
                # Prompt B3's alpha=0.05 == confidence_level=0.95. Fitted on
                # the SAME train split as the B1 regressor, on the 168h target
                # specifically, so the held-out test rows never leak into the
                # conformal calibration.
                ccr = CrossConformalRegressor(
                    estimator=XGBRegressor(
                        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=0
                    ),
                    confidence_level=MAPIE_CONFIDENCE_LEVEL,
                    method="plus",
                    cv=5,
                    random_state=0,
                )
                ccr.fit_conformalize(X_tr, y_tr[f"{p}_168h"])
                ccr_preds, pis = ccr.predict_interval(X)
                # B3: the MAPIE-wrapped regressor replaces the plain one for
                # the 168h target specifically — its point prediction and the
                # 95% interval bounds land in the frame. 96h stays the plain
                # MultiOutput prediction (B3 covers 168h only).
                df[f"{p}_168h_pred"] = np.asarray(ccr_preds, dtype=float).reshape(-1)
                pis_arr = np.asarray(pis)
                if pis_arr.ndim == 3:  # scalar confidence -> (n, 2, 1)
                    df[f"{p}_168h_lower"] = pis_arr[:, 0, 0]
                    df[f"{p}_168h_upper"] = pis_arr[:, 1, 0]
                else:
                    df[f"{p}_168h_lower"] = pis_arr[:, 0]
                    df[f"{p}_168h_upper"] = pis_arr[:, 1]
                mae_168_mapie = mean_absolute_error(
                    y_te[f"{p}_168h"],
                    np.asarray(ccr.predict(X_te), dtype=float).reshape(-1),
                )
                print(f"  [{p}] 168h MAE (MAPIE point, test): {mae_168_mapie:.4f}")
        else:
            # ---- score with pre-fitted models ----
            if p not in models:
                raise ValueError(
                    f"run_module_b(train=False) requires a fitted model for '{p}' "
                    f"in the `models` argument"
                )
            preds = models[p].predict(X)
            df[f"{p}_96h_pred"] = preds[:, 0]
            df[f"{p}_168h_pred"] = preds[:, 1]

    # ---- intra-parameter slope (Prompt B2 step 1) ----
    for p in params:
        df[f"{p}_slope"] = (df[f"{p}_168h_pred"] - df[f"{p}_24h"]) / (168.0 - 24.0)

    # ---- slope normalization over the normal population (Prompt B2 step 2) ----
    for p in params:
        sd = float(df.loc[mask, f"{p}_slope"].std())
        df[f"{p}_slope_norm"] = df[f"{p}_slope"] / (sd + STD_FLOOR)

    # ---- fuse intra-component: weighted L2 drift vector (Prompt B2 step 3) ----
    df["v_drift"] = np.sqrt(sum(weights[p] * df[f"{p}_slope_norm"] ** 2 for p in params))

    # ---- safety-slope gate, derived from the normal population (steps 4-5) ----
    # theta_k=3.0 is Prompt B2's spec default; see the theta_k docstring for
    # the team-approved FN-tuning story.
    normal_v = df.loc[mask, "v_drift"].to_numpy(dtype=float)
    med = float(np.median(normal_v))
    mad = float(np.median(np.abs(normal_v - med)))
    theta_slope = med + theta_k * mad
    if theta_blended:
        # 🔵 §8b: per-lot Empirical-Bayes-blended theta (default OFF — see
        # the docstring). Returns a dict keyed by lot_id; drift_flag uses
        # each row's own lot's blended theta.
        theta_by_lot = _theta_slope_blended(df, mask, theta_k=theta_k)
        row_theta = df["lot_id"].map(theta_by_lot).to_numpy(dtype=float)
        df["drift_flag"] = df["v_drift"].to_numpy(dtype=float) > row_theta
        return df, models, theta_by_lot
    df["drift_flag"] = df["v_drift"] > theta_slope

    return df, models, theta_slope


def _theta_slope_blended(
    df: pd.DataFrame, mask: np.ndarray, lot_col: str = "lot_id", theta_k: float = 3.0
) -> dict:
    """🔵 Per-lot theta_slope with Empirical Bayes blending (§8b).

    Reuses Module A's small-lot pattern (w = N/(N+30)) instead of inventing
    new logic: for each lot, the blended theta is w * (lot median + theta_k *
    lot MAD) + (1-w) * (global median + theta_k * global MAD), all computed
    over the SAME normal population (mask) the global gate uses. Large lots
    (w -> 1) keep their own statistics; small lots shrink toward the global
    prior so their gate is not dominated by noise. Global fallback for any
    lot missing from the frame (defensive; every row has a lot_id per §4b).
    """
    normal_v = df.loc[mask, "v_drift"].to_numpy(dtype=float)
    global_med = float(np.median(normal_v))
    global_mad = float(np.median(np.abs(normal_v - global_med)))
    global_theta = global_med + theta_k * global_mad

    theta_by_lot: dict = {}
    lot_ids = df.loc[mask, lot_col]
    v_by_lot = df.loc[mask, "v_drift"].groupby(lot_ids)
    for lot, vals in v_by_lot:
        n_lot = int(df.loc[mask & (df[lot_col] == lot), lot_col].count())
        w = n_lot / (n_lot + SMALL_LOT_K)
        vals = vals.to_numpy(dtype=float)
        lot_med = float(np.median(vals))
        lot_mad = float(np.median(np.abs(vals - lot_med)))
        theta_by_lot[lot] = w * (lot_med + theta_k * lot_mad) + (1 - w) * global_theta
    return theta_by_lot


def module_b_new_cols(params: list, with_intervals: bool = False) -> list[str]:
    """Columns this module adds, in order — used by main.py for the §4e merge.

    with_intervals=True appends the 🔵 MAPIE interval columns (Prompt B3);
    the default list is unchanged so existing callers are unaffected.
    """
    cols: list[str] = []
    for p in params:
        cols += [f"{p}_96h_pred", f"{p}_168h_pred", f"{p}_slope", f"{p}_slope_norm"]
        if with_intervals:
            cols += [f"{p}_168h_lower", f"{p}_168h_upper"]
    cols += ["v_drift", "drift_flag"]
    return cols


# ---------------------------------------------------------------------------
# 🔵 Prompt B4 — OOD gate on RAW regression inputs (not Module A's Z-scores)
# ---------------------------------------------------------------------------

def build_ood_features(df: pd.DataFrame, params: list) -> pd.DataFrame:
    """Raw regression feature matrix for the OOD gate (Prompt B4).

    Documented choice: the pipeline uses the COMBINED feature set — every
    parameter's raw features [{p}_0h, {p}_24h, {p}_delta] side by side, then
    the shared stress context (T_ambient, V_stress) ONCE, so each component
    gets a single OOD verdict matching its component-level decision. The
    shared context is not repeated per parameter (it is identical across
    parameters and would only triple its weight in the Isolation Forest).
    Per-parameter use also works: build_ood_features(df, [p]) returns exactly
    that parameter's 5-column regression feature matrix.

    {p}_delta is computed here (24h - 0h, the same formula as Prompt B1) so
    this helper needs only the frozen §4b schema — not Module B internals.
    """
    out = pd.DataFrame(index=df.index)
    for p in params:
        out[f"{p}_0h"] = df[f"{p}_0h"]
        out[f"{p}_24h"] = df[f"{p}_24h"]
        out[f"{p}_delta"] = df[f"{p}_24h"] - df[f"{p}_0h"]
    out["T_ambient"] = df["T_ambient"]
    out["V_stress"] = df["V_stress"]
    return out


def fit_ood_gate(
    X_train: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 0,
) -> IsolationForest:
    """Fit the OOD detector on the RAW regression input features (Prompt B4).

    Plain sklearn IsolationForest — deliberately NOT on the Z-scores, which
    belong to Module A's Tier 3; this gate watches the raw feature space for
    failure physics never seen in training. Returns the fitted detector.
    """
    detector = IsolationForest(contamination=contamination, random_state=random_state)
    detector.fit(X_train)
    return detector


def score_ood(detector: IsolationForest, X: pd.DataFrame):
    """Score raw inputs with the fitted OOD gate (Prompt B4).

    Returns (ood_score, ood_flag): ood_score = -score_samples (higher = more
    anomalous — same convention as Module A's joint_anomaly_score) and
    ood_flag = predict(X) == -1. Column-shaped numpy arrays; the caller
    decides how to attach them to a frame (kept separate so the Decision
    Maker wiring stays a Person 1/main.py concern, per the prompt).
    """
    ood_score = -detector.score_samples(X)
    ood_flag = detector.predict(X) == -1
    return ood_score, ood_flag


# ---------------------------------------------------------------------------
# 🔵 Prompt B5 — progressive refinement (standalone, NOT wired into the module)
# ---------------------------------------------------------------------------

def resolve_checkpoint(row, param: str, horizon: int, predicted_value: float):
    """Prefer a real measured checkpoint value over a prediction (Prompt B5).

    Looks for the column f"{param}_{horizon}h_actual" in `row` (a Series or
    dict); if present and not NaN, returns (actual_value, "measured"),
    otherwise returns (predicted_value, "predicted"). Deliberately standalone
    — callers apply it as an optional pre-processing step before the main
    pipeline, so the required prediction path can never be destabilized.
    """
    real_col = f"{param}_{horizon}h_actual"
    if real_col in row and pd.notna(row[real_col]):
        return row[real_col], "measured"
    return predicted_value, "predicted"
