# Phase 1 Report — Burn-In Predictive Screening

**Date:** 2026-09-25
**Status:** ✅ All 🟢 required features + all 🔵 add-on features implemented and tested
**Test suite:** 93/93 passing (`cd burnin_screening && python -m pytest`)
**Docs audited:** `Implementation/burnin-architecture-guide.md`, `Implementation/person1_static_decision.md`, `Implementation/person2_temporal_prediction.md`

---

## 1. Verification Evidence

- **Full pytest run:** 93 passed, 0 failed — 16 decision_maker + 6 integration +
  17 module_a + 16 module_b_addons + 27 phase3_review_split + 5 theta_tuning + 6 validate.
- **500-row demo artifact** (`data/pipeline_output.csv`, theta_k=2.4):
  PASS 402 / FLAG 58 / REVIEW 40 · TN 442, FP 33, **FN 0**, TP 25 → **strict recall 1.000**.
  All **12/12 compound defects FLAGged** (the architecture guide's "non-negotiable"
  compound check). 33 drift reasons carry the 96h trajectory line.

## 2. Implemented Features

### 🟢 Required — Module A: Static Anomaly Detection (`module_a_static.py`)

| Feature | Contract col(s) |
|---|---|
| Tier 1 — hard datasheet limit check, per parameter | `{p}_hard_flag` |
| Tier 2 — robust modified Z-score (median/MAD), per parameter, per lot | `{p}_z` |
| Weighted L2 fusion of the 3 per-parameter Z-scores | `z_fused`, `static_outlier_flag` |
| Tier 3 — joint Isolation Forest on the Z-vector | `joint_anomaly_score`, `joint_outlier_flag` |

### 🟢 Required — Module B: Temporal Drift Prediction (`module_b_temporal.py`)

| Feature | Contract col(s) |
|---|---|
| XGBoost per-parameter regressor `[y0h, y24h, Δy, T_ambient, V_stress] → [96h, 168h]`, 168h MAE printed | `{p}_96h_pred`, `{p}_168h_pred` |
| Intra-parameter slope `(ŷ_168h − y_24h)/144` | `{p}_slope` |
| Slope normalization over the normal population (`static_outlier_flag == False`) | `{p}_slope_norm` |
| Weighted L2 drift-vector fusion (same convention as Module A) | `v_drift` |
| Safety-slope gate `theta_slope = median + theta_k·MAD` (normal population), returned to caller | `drift_flag` |

### 🟢 Required — Decision Maker & Validation (`decision_maker.py`, `validate.py`)

| Feature | Detail |
|---|---|
| PASS/FLAG decision, priority order 1–4 (Prompt A4) | hard limit → joint anomaly → static outlier → drift breach |
| Per-parameter reason strings | every FLAG names the driving parameter + score (Explainability metric) |
| Confusion matrix + prominent FN report | strict recall/precision, FN rows named |
| Compound-defect breakdown | shows whether the fusion/joint layers catch the case they were built for |

### 🟢 Required — Data & Integration (`data/generate_data.py`, `main.py`)

| Feature | Detail |
|---|---|
| Wide-schema synthetic generator (~500 rows, 8 large + 2 small lots) | `single` (sharp one-param) + `compound` (mild all-3-params) defect signatures |
| Merge on `component_id` (never row position, §4e) | both modules run on the SAME input frame |
| CLI demo entrypoint (`--n`, `--seed`, `--regen`, `--theta-k`) | saves `data/pipeline_output.csv` |

### 🔵 Add-ons (all implemented, all tested)

| Add-on | Where | Tests |
|---|---|---|
| 96h auxiliary forecast + trajectory line in drift FLAG reasons | `module_b_temporal.py`, `decision_maker.py` | 4 trajectory tests + `test_b3_96h_pred_is_not_the_mapie_path` |
| MAPIE conformal intervals (B3): `CrossConformalRegressor(method="plus", cv=5, 0.95)` on the 168h target via `with_intervals=True` | `module_b_temporal.py` | 5 `test_b3_*` + wiring test |
| OOD gate on RAW regression inputs (B4): `build_ood_features` / `fit_ood_gate` / `score_ood`, combined feature set | `module_b_temporal.py` + `main.py` wiring | 4 `test_b4_*` + wiring test |
| Progressive refinement (B5): `resolve_checkpoint` standalone + pipeline layer — measured overrides predicted, per-row slope window (/144 or /72), `_refined` columns, graded `{p}_168h_pred` never overwritten | `module_b_temporal.py`, `main.py::apply_progressive_refinement` | 5 `test_b5_*` + 5 refinement tests |
| PASS/REVIEW/FLAG split — OOD REVIEW (branch 5), wide-interval REVIEW (branch 6); REVIEW never overrides FLAG; graceful degradation to binary PASS/FLAG | `decision_maker.py` | 27 tests in `test_phase3_review_split.py` |
| `theta_width` derivation — median + 3·MAD of per-row max relative interval width over the normal population (`max_relative_width`) | `main.py`, `decision_maker.py` | 4 tests |
| REVIEW-aware validation — lenient recall (FLAG+REVIEW), REVIEW breakdown by cause, deferred defects, `compound_deferred` | `validate.py` | 4 tests |
| Empirical Bayes small-lot blending (w = N/(N+30)), off by default | `module_a_static.py` | 3 `test_small_lot_blending_*` |
| `theta_k` tuning lever — spec default 3.0, demo ships 2.4 (joint-grid tuning story) | `module_b_temporal.py`, `main.py` | 5 tests in `test_theta_tuning.py` |
| Add-on switches (`with_intervals`, `with_ood`, `with_progressive_refinement`) — all-off = byte-identical required path | `main.py::run_pipeline` | `test_run_pipeline_addon_switches_off`, `test_default_path_columns_unchanged` |

### ⚪ Stretch — intentionally not implemented

| Item | Why |
|---|---|
| Arrhenius/exponential time-to-failure estimate | "Good demo material, not graded" per both person docs |
| Physics-Informed Neural Network alternative | "High effort, low marginal grading benefit; only if the 🟢 path is solid with time to spare" |

## 3. Pending Tasks

**None blocking.** Required + add-on scope is complete and green (93/93); the shipped
demo shows FN = 0 with every compound defect caught. Optional residuals:

1. ⚪ Arrhenius time-to-failure estimate (Module B) — demo material only.
2. ⚪ PINN alternative — high effort, explicitly deferred.
3. Hygiene: refresh `data/pipeline_output.csv` after any threshold change
   (`python main.py`); keep the §4c/§4d/§4f contract freeze when editing either
   person's module.
