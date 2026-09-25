"""Decision Maker — combines Module A's and Module B's flags into a final
PASS/FLAG verdict with a plain-language, per-parameter reason. Person 1 owns
this file.

It reads ONLY the documented output columns of both modules (contracts §4c
and §4d) — never either module's internals.

Decision priority order (first match wins — 1 per Prompt A4, 2 added per the
Phase 1 review §8a, 3-5 per Prompt A4, 6-7 are the 🔵 Phase 3 REVIEW branches):
  1. Any {p}_hard_flag True                         -> FLAG (Tier 1 violation, measured 24h)
  2. {p}_168h_pred >= Y_spec[p] with a narrow interval -> FLAG (predicted 168h breach)
  3. joint_outlier_flag True                        -> FLAG (Tier 3 joint anomaly)
  4. static_outlier_flag True                       -> FLAG (fused lot outlier)
  5. drift_flag True (Module B)                     -> FLAG (safety-slope breach)
  6. ood_flag True (Module B OOD gate)              -> REVIEW (OOD inputs)
  7. widest 168h interval > theta_width             -> REVIEW (low confidence)
  8. Otherwise                                      -> PASS

🔵 Predicted-breach FLAG (branch 2, Phase 1 review §8a): the design's own
Module 4 decision table specifies FLAG on "predicted 168h value >= Y_spec with
a narrow interval" as a trigger INDEPENDENT of the drift-slope gate. The drift
gate answers "is this trending toward failure"; this branch answers the
literal question "would the part's predicted end-of-test value itself violate
the datasheet" — a component can stay under theta_slope yet have its
predicted 168h value cross Y_spec (mild slope from a value already close to
the limit). Requires Y_spec (dict param -> datasheet limit) threaded into
decide(); Y_spec=None/{} leaves the branch inert so the Phase 3 7-branch
behavior is unchanged for existing callers. "Narrow" means the parameter's
relative interval width is <= theta_width — when intervals are off
(theta_width=None) or missing, every predicted breach is flaggable (there is
no width to check); a WIDE interval defers to the width-REVIEW branch instead
of double-flagging an already low-confidence prediction.

Every FLAG reason names a specific parameter — never just a bare score. This
is what satisfies the Explainability evaluation metric: a QA inspector must
be able to audit exactly why any part was rejected.

Drift FLAG reasons additionally carry a 96h trajectory line (Person 1 doc §6:
the 🔵 96h auxiliary forecast "improves the drift-fit and the explanation
trail"): for the driving parameter it shows the measured 24h value followed
by Module B's forecast 96h and 168h values, so the inspector sees the
predicted trajectory that breached the safety slope, not just a rate number.
It reads only contract §4d columns ({p}_96h_pred / {p}_168h_pred) and degrades
to a 24h->168h line when the 96h column is absent or non-finite — the required
path never depends on the auxiliary forecast.

🔵 Three-way PASS/REVIEW/FLAG split (architecture guide add-on #6: split
FLAG "once interval width and OOD score exist" — both exist since Phase 2).
REVIEW defers UNCERTAIN cases to a human instead of forcing a guess; it never
overrides evidence-based detections:

  6. OOD gate fired (ood_flag True) but no FLAG branch did
     -> REVIEW ("failure physics outside the training distribution — the
     point predictions and intervals are not trustworthy here").
  7. A would-PASS row whose widest 168h conformal interval is unusually wide
     relative to its point prediction
     -> REVIEW ("the model is guessing — per architecture guide #3, treat
     any prediction with a very wide interval as unreliable").
  8. Otherwise -> PASS.

Both REVIEW conditions degrade gracefully: on a contract-minimum frame
(no interval columns, no ood_flag) every missing/non-finite value counts as
"not uncertain", so decide() returns exactly the binary PASS/FLAG verdict of
the required path — an absent OOD gate or absent intervals can never create
a REVIEW, only fail to refine one.

Threshold policy (decided with the team, mirroring theta_slope's
derived-from-the-normal-population philosophy rather than hardcoded
cutoffs: theta_width must be the median + 3*MAD of the per-row maximum
relative interval width over the NORMAL population (Module A's
static_outlier_flag == False) — the same normal-population derivation as
Module B's safety-slope gate. main.py computes it; the required-path
semantics don't depend on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _driving_param(row: pd.Series, params: list, score_template: str) -> str:
    """Parameter with the largest absolute value of the given score column.

    .get with a 0.0 fallback keeps this crash-free on hand-built frames that
    carry the trigger flag but not every score column — it then names the
    first parameter rather than raising, still satisfying "reasons name a
    parameter".
    """
    return max(params, key=lambda p: abs(row.get(score_template.format(p=p), 0.0)))


def _finite(row: pd.Series, col: str) -> float | None:
    """Row value at `col` as a finite float, else None (missing/NaN/non-numeric).

    The trajectory line is an explanation nicety — a missing auxiliary column
    (the contract-minimum frame) or a NaN must degrade the line, not crash.
    """
    v = row.get(col)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _trajectory_line(row: pd.Series, top: str) -> str:
    """96h trajectory line for the drift FLAG reason (Person 1 doc §6).

    Format: " Trajectory for {top}: y(24h)=<24h value> measured ->
    y(96h)=<{top}_96h_pred> forecast -> y(168h)=<{top}_168h_pred> forecast."
    The 96h hop is skipped when the auxiliary column is absent or non-finite;
    an empty string is returned if no finite point exists at all.
    """
    points: list[str] = []
    y24 = _finite(row, f"{top}_24h")
    y96 = _finite(row, f"{top}_96h_pred")
    y168 = _finite(row, f"{top}_168h_pred")
    if y24 is not None:
        points.append(f"y(24h)={y24:.3g} measured")
    if y96 is not None:
        points.append(f"y(96h)={y96:.3g} forecast")
    if y168 is not None:
        points.append(f"y(168h)={y168:.3g} forecast")
    if not points:
        return ""
    return f" Trajectory for {top}: " + " -> ".join(points) + "."


def _rel_widths(row: pd.Series, params: list) -> dict[str, float]:
    """Relative width of each parameter's 168h conformal interval.

    width_p = ({p}_168h_upper - {p}_168h_lower) / |{p}_168h_pred| — a
    scale-free uncertainty measure comparable ACROSS parameters (I_leak runs
    ~10uA, I_ddq ~2mA, t_pd ~8ns, so raw widths aren't comparable). Only
    finite entries are returned; missing/non-finite interval columns (the
    contract-minimum frame, or intervals disabled) yield {} — no width
    signal, no width REVIEW.
    """
    widths: dict[str, float] = {}
    for p in params:
        pred = _finite(row, f"{p}_168h_pred")
        lower = _finite(row, f"{p}_168h_lower")
        upper = _finite(row, f"{p}_168h_upper")
        if pred is None or lower is None or upper is None:
            continue
        denom = abs(pred)
        if denom < 1e-12:
            continue  # degenerate denominator — skip rather than divide by ~0
        w = (upper - lower) / denom
        if np.isfinite(w) and w >= 0:
            widths[p] = w
    return widths


def max_relative_width(df: pd.DataFrame, params: list) -> pd.Series:
    """Per-row MAXIMUM relative 168h interval width (vectorized form).

    Same formula as `_rel_widths` (kept row-wise above for per-parameter
    attribution in reason strings) — this is the vectorized version main.py
    uses to derive theta_width over the normal population. Rows missing any
    interval column (or with a degenerate prediction) get NaN, which the
    theta_width derivation drops — an absent interval layer can never
    manufacture uncertainty.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    needed = [f"{p}_{c}" for p in params for c in ("168h_pred", "168h_lower", "168h_upper")]
    if not all(c in df.columns for c in needed):
        return out
    for p in params:
        pred = df[f"{p}_168h_pred"].astype(float)
        lower = df[f"{p}_168h_lower"].astype(float)
        upper = df[f"{p}_168h_upper"].astype(float)
        denom = pred.abs().where(pred.abs() >= 1e-12)  # degenerate denom -> NaN
        w = (upper - lower) / denom
        # max over parameters; NaN until some parameter yields a valid width
        out = pd.concat([out, w], axis=1).max(axis=1, skipna=True)
    return out


def _fmt_limit(row: pd.Series, param: str) -> str:
    """Format Y_spec_{param} for a reason string, tolerating its absence.

    The merged pipeline frame always carries Y_spec_{p} columns (schema §4b),
    but decide() also runs on hand-built frames — a missing/NaN limit must
    degrade to a placeholder instead of crashing the :.3g format.
    """
    v = row.get(f"Y_spec_{param}")
    try:
        return f"{float(v):.3g}"
    except (TypeError, ValueError):
        return "Y_spec"


def decide(
    df: pd.DataFrame,
    params: list,
    theta_slope: float,
    theta_width: float | None = None,
    Y_spec: dict | None = None,
) -> pd.DataFrame:
    """Returns df with two new columns: decision ("PASS"/"REVIEW"/"FLAG") and
    reason.

    Args:
        df: merged dataframe containing both Module A's and Module B's output
            columns per the frozen contracts (§4c + §4d). Not modified in place.
        params: list of parameter names, e.g. ["I_leak", "I_ddq", "t_pd"].
        theta_slope: the safety-slope threshold computed by Module B (logged
            alongside its outputs) — cited in drift FLAG reasons.
        Y_spec: 🔵 optional dict[param -> datasheet limit] enabling the
            predicted-168h-breach FLAG branch (Phase 1 review §8a — the
            design decision table's "predicted value >= Y_spec with a narrow
            interval" rule, ranked below the measured 24h violation and above
            all statistical layers). None/{} -> branch inert, degrading to the
            previous 7-branch behavior. Parameters absent from the dict are
            skipped, never crash.
        theta_width: 🔵 REVIEW threshold for the conformal interval width —
            the median + 3*MAD of the per-row maximum relative interval width
            over the normal population (same derivation philosophy as
            theta_slope; main.py computes it). None/absent interval columns
            -> the width REVIEW branch is inert and the split degrades to the
            binary PASS/FLAG of the required path.

    REVIEW is consulted only when no FLAG branch fired: evidence-based
    detections (hard limit, predicted 168h breach, joint anomaly, lot
    outlier, drift breach) stay actionable — the OOD/uncertainty signals
    modulate CONFIDENCE, not guilt (decided with the team for Phase 3;
    conservative for false negatives).
    """
    df = df.copy()

    decisions: list[str] = []
    reasons: list[str] = []

    for _, row in df.iterrows():
        # 1. Tier 1 — hard datasheet limits (highest priority)
        hard_hits = [p for p in params if row.get(f"{p}_hard_flag", False)]
        if hard_hits:
            details = ", ".join(
                f"{p} at {row[f'{p}_24h']:.3g} (limit {_fmt_limit(row, p)})"
                for p in hard_hits
                if f"{p}_24h" in row.index
            )
            decisions.append("FLAG")
            reasons.append(f"Hard limit violated on: {', '.join(hard_hits)}. {details}.")
            continue

        # 2. 🔵 Predicted 168h breach (Phase 1 review §8a): the design
        #    decision table's "predicted 168h value >= Y_spec with a narrow
        #    interval" rule — independent of the drift-slope gate. Inert
        #    without Y_spec; "narrow" is checked against the parameter's
        #    relative interval width only when a theta_width exists, so a
        #    wide-interval prediction defers to the width REVIEW instead of
        #    double-flagging low-confidence numbers.
        if Y_spec:
            pred_hits: list[tuple[str, float, float]] = []
            for p in params:
                limit = Y_spec.get(p)
                if limit is None:
                    continue
                pred = _finite(row, f"{p}_168h_pred")
                if pred is None or pred < limit:
                    continue
                if theta_width is not None and _rel_widths(row, params).get(p, 0.0) > theta_width:
                    continue  # wide interval -> width REVIEW may catch it instead
                pred_hits.append((p, pred, float(limit)))
            if pred_hits:
                details = ", ".join(
                    f"{p} predicted {v:.3g} (limit {lim:.3g})" for p, v, lim in pred_hits
                )
                decisions.append("FLAG")
                reasons.append(
                    "Predicted 168h value would cross the datasheet limit, "
                    f"independent of drift rate: {details}."
                )
                continue

        # 3. Tier 3 — joint multi-parameter anomaly (Isolation Forest)
        if row.get("joint_outlier_flag", False):
            top = _driving_param(row, params, "{p}_z")
            decisions.append("FLAG")
            reasons.append(
                f"Joint multi-parameter anomaly (score {row['joint_anomaly_score']:.2f}); "
                f"led by {top} (z={row[f'{top}_z']:.2f})."
            )
            continue

        # 4. Fused lot-relative outlier (weighted L2 over per-parameter Z)
        if row.get("static_outlier_flag", False):
            top = _driving_param(row, params, "{p}_z")
            decisions.append("FLAG")
            reasons.append(
                f"Fused lot-relative outlier (z_fused={row['z_fused']:.1f}); "
                f"driven mainly by {top} (z={row[f'{top}_z']:.2f})."
            )
            continue

        # 5. Predicted drift breach (Module B safety-slope gate) — with the
        #    96h trajectory line for the driving parameter (explanation trail)
        if row.get("drift_flag", False):
            top = _driving_param(row, params, "{p}_slope_norm")
            decisions.append("FLAG")
            reasons.append(
                f"Predicted drift vector {row['v_drift']:.2f} exceeds safety "
                f"threshold {theta_slope:.2f}; driven mainly by {top} "
                f"(slope_norm={row[f'{top}_slope_norm']:.2f})."
                + _trajectory_line(row, top)
            )
            continue

        # 6. 🔵 OOD gate (Module B helpers, wired by main.py): failure physics
        #    outside the training distribution — predictions carry no trust
        #    here. Deferred to a human instead of forcing a guess.
        if row.get("ood_flag", False):
            top = _driving_param(row, params, "{p}_slope_norm")
            ood_score = row.get("ood_score")
            ood_txt = f"{float(ood_score):.2f}" if _finite(row, "ood_score") is not None else "n/a"
            decisions.append("REVIEW")
            reasons.append(
                f"Out-of-distribution inputs (OOD score {ood_txt}) — failure "
                f"physics outside the training data; predictions untrustworthy "
                f"(leading parameter {top}). Manual review required."
            )
            continue

        # 7. 🔵 Unusually wide conformal interval on an otherwise-clean row:
        #    the model is guessing (architecture guide #3 — "treat any
        #    prediction with a very wide interval as unreliable").
        if theta_width is not None:
            widths = _rel_widths(row, params)
            if widths:
                top = max(widths, key=widths.get)
                if widths[top] > theta_width:
                    decisions.append("REVIEW")
                    reasons.append(
                        f"Wide 168h prediction interval for {top} "
                        f"(relative width {widths[top]:.2f} > uncertainty "
                        f"threshold {theta_width:.2f}) — low-confidence "
                        f"prediction. Manual review required."
                    )
                    continue

        # 8. Clean pass
        decisions.append("PASS")
        reasons.append("Within limits, lot-normal, no joint anomaly, drift within safety bounds.")

    df["decision"] = decisions
    df["reason"] = reasons
    return df
