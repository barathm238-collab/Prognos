"""Module A — Static (0h/24h) Anomaly Detection. Person 1 owns this file.

Implements the three-tier detection stack from the architecture doc:

  Tier 1 (🟢): hard datasheet limit check, per parameter — {p}_24h >= Y_spec[p].
  Tier 2 (🟢): intra-parameter robust modified Z-score (median/MAD), computed
               per parameter, per lot, on the {p}_24h column. This is the
               "dynamic outlier" ask: flags a 45uA part in a 10uA-average lot
               even though 45 < the 50uA absolute limit.
  Fusion (🟢): weighted L2 norm over the 3 per-parameter Z-scores into one
               component-level score — multi-parameter fusion, required
               because "latent defect" is defined as drift invisible to any
               single parameter's limit.
  Tier 3 (🟢): joint anomaly score — Isolation Forest on the raw 3-parameter
               Z-vector. Catches compound drift where each parameter
               individually stays under its Z-threshold but all three move
               together (the direct fix for Tier 2's false-negative blind spot).

Output contract (§4c — frozen, do not rename):
  run_module_a(df, Y_spec, params) returns df with these columns ADDED
  (never removes/renames input columns):
    {p}_hard_flag        bool   — per parameter, Tier 1
    {p}_z                float  — per parameter, Tier 2
    z_fused               float  — fused intra-component Z (weighted L2)
    static_outlier_flag   bool   — z_fused > threshold
    joint_anomaly_score   float  — Tier 3, higher = more anomalous
    joint_outlier_flag    bool   — Tier 3 Isolation Forest verdict

🔵 Add-on (build after the 🟢 path validates): Empirical Bayes small-lot
blending — see `small_lot_blending` argument, off by default so the required
path is unaffected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Small-lot threshold for the Empirical Bayes blending add-on (§6, Person 1):
# raw median/MAD statistics become unreliable for lots under ~30 units.
SMALL_LOT_N = 30
# Blending weight w = N / (N + K): a lot of size N gets weight w on its own
# statistics and (1 - w) on the global (pooled) statistics. K=30 per the doc.
SMALL_LOT_K = 30


def _modified_z(vals: np.ndarray) -> np.ndarray:
    """Robust modified Z-score: 0.6745 * (x - median) / MAD, MAD floored at 1e-6."""
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    return 0.6745 * (vals - med) / max(mad, 1e-6)


def _z_with_small_lot_blending(
    df: pd.DataFrame, col: str, small_mask: pd.Series, global_med: float, global_mad: float
) -> np.ndarray:
    """🔵 Empirical Bayes blend of lot stats with global priors for small lots.

    For each row: w = N / (N + K); blended median/mad = w * lot_stat +
    (1 - w) * global_stat. Large lots (w -> 1) keep pure lot statistics; small
    lots shrink toward the global prior, keeping the Z-score meaningful when
    the lot's own median/MAD is noise-dominated. Normal-sized lots are
    computed with the plain lot statistics (identical to the 🟢 path).
    """
    out = np.empty(len(df), dtype=float)
    med_col = df.loc[:, col]
    for lot, idx in df.groupby("lot_id").indices.items():
        vals = med_col.iloc[idx].to_numpy(dtype=float)
        n = len(vals)
        if small_mask.iloc[idx[0]]:  # whole lot shares one N
            w = n / (n + SMALL_LOT_K)
            med = w * np.median(vals) + (1 - w) * global_med
            mad = w * np.median(np.abs(vals - med)) + (1 - w) * global_mad
        else:
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
        out[idx] = 0.6745 * (vals - med) / max(mad, 1e-6)
    return out


def run_module_a(
    df: pd.DataFrame,
    Y_spec: dict,
    params: list,
    weights: dict | None = None,
    z_threshold: float = 3.5,
    contamination: float = 0.05,
    small_lot_blending: bool = False,
) -> pd.DataFrame:
    """Run the full static detection stack. See module docstring for the
    exact output columns added (contract §4c).

    Args:
        df: input dataframe with the shared schema (§4b). Never modified
            in place — a copy with new columns is returned.
        Y_spec: dict[param -> hard datasheet limit].
        params: list of parameter names, e.g. ["I_leak", "I_ddq", "t_pd"].
        weights: optional dict[param -> float] L2 fusion weights; defaults to
            1.0 for every parameter (tune later, equal to start).
        z_threshold: static_outlier_flag fires when z_fused > this (3.5 per doc).
        contamination: Isolation Forest contamination fraction (0.05 per doc).
        small_lot_blending: 🔵 add-on — enable Empirical Bayes blending of
            lot statistics with global priors for lots with N < 30. Off by
            default so the required 🟢 path is unaffected.
    """
    df = df.copy()
    if weights is None:
        weights = {p: 1.0 for p in params}
    missing = [c for p in params for c in (f"{p}_24h",) if c not in df.columns]
    if missing:
        raise KeyError(f"run_module_a: input df is missing required columns: {missing}")
    if "lot_id" not in df.columns:
        raise KeyError("run_module_a: input df is missing required column: lot_id")

    # ---- Tier 1: hard datasheet limits, per parameter (🟢) ----
    for p in params:
        df[f"{p}_hard_flag"] = df[f"{p}_24h"] >= Y_spec[p]

    # ---- Tier 2: intra-parameter robust modified Z-score, per lot (🟢) ----
    if small_lot_blending:
        for p in params:
            col = f"{p}_24h"
            global_med = float(np.median(df[col]))
            global_mad = float(np.median(np.abs(df[col] - global_med)))
            small_mask = df["N"] < SMALL_LOT_N if "N" in df.columns else pd.Series(False, index=df.index)
            df[f"{p}_z"] = _z_with_small_lot_blending(df, col, small_mask, global_med, global_mad)
    else:
        for p in params:
            df[f"{p}_z"] = df.groupby("lot_id")[f"{p}_24h"].transform(_modified_z)

    # ---- Fuse intra-component: weighted L2 norm over the Z-scores (🟢) ----
    df["z_fused"] = np.sqrt(sum(weights[p] * df[f"{p}_z"] ** 2 for p in params))
    df["static_outlier_flag"] = df["z_fused"] > z_threshold

    # ---- Tier 3: joint Isolation Forest on the raw Z-vector (🟢) ----
    # Run on the per-parameter Z-vector (not a single scalar) so it can catch
    # correlation the L2 fusion's fixed weights might under-weight.
    z_matrix = df[[f"{p}_z" for p in params]].to_numpy()
    iso = IsolationForest(contamination=contamination, random_state=0)
    iso.fit(z_matrix)
    # score_samples: higher = more normal -> negate so higher = more anomalous
    df["joint_anomaly_score"] = -iso.score_samples(z_matrix)
    df["joint_outlier_flag"] = iso.predict(z_matrix) == -1

    return df


# Columns this module adds, in order — used by main.py to select Module B's
# columns for the merge (contract §4e) and by tests.
def module_a_new_cols(params: list) -> list[str]:
    cols: list[str] = []
    for p in params:
        cols += [f"{p}_hard_flag", f"{p}_z"]
    cols += ["z_fused", "static_outlier_flag", "joint_anomaly_score", "joint_outlier_flag"]
    return cols
