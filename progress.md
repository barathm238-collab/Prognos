# progress.md — Phase 3 implementation state

**Date:** 2026-09-24 · **Status:** ✅ COMPLETE — resumed after pause, all steps finished.
See §7 for the final report; §1-6 below document the paused state for reference.

---

## 1. What Phase 3 is (scope recap)

User approved **full Phase 3** with two locked design decisions:

1. **REVIEW never overrides FLAG.** Evidence-based detections (hard limit / joint anomaly /
   static outlier / drift breach) stay FLAG. REVIEW is only for rows that would otherwise PASS.
   Rationale: conservative for false negatives (the metric penalized most).
2. **Width cutoff is data-derived**, mirroring theta_slope's philosophy:
   `theta_width = median + 3·MAD` of the per-row **max relative interval width** over the
   NORMAL population (`static_outlier_flag == False`), where
   `relative_width_p = ({p}_168h_upper − {p}_168h_lower) / |{p}_168h_pred|`
   (scale-free so I_leak ~10uA, I_ddq ~2mA, t_pd ~8ns are comparable).

---

## 2. Implemented so far (all code changes)

### `burnin_screening/decision_maker.py`
- `decide(df, params, theta_slope, theta_width=None)` — 7-branch priority:
  1-4 unchanged FLAG branches (Prompt A4), then
  5. `ood_flag` True → **REVIEW**, reason cites OOD score + leading parameter
     (`_driving_param` on `{p}_slope_norm`), tolerates missing `ood_score` ("n/a").
  6. widest relative interval > `theta_width` → **REVIEW**, reason cites parameter,
     width, threshold. Inert when `theta_width is None` or interval columns absent/NaN;
     degenerate `|pred| < 1e-12` rows skipped.
  7. PASS (reason string unchanged).
- `_driving_param` now uses `row.get(..., 0.0)` so hand-built frames can't crash it.
- New helpers: `_rel_widths(row, params)` (row-wise, for reason attribution) and
  **`max_relative_width(df, params)`** (public, vectorized per-row max; NaN when any
  interval column missing — used by main.py for theta_width derivation).
- Docstring updated with the 7-branch table + threshold policy.

### `burnin_screening/main.py`
- `run_pipeline(df_input, params, verbose, with_intervals=True, with_ood=True,
  with_progressive_refinement=True)` — add-on switches; every combination produces a
  valid frame; all-off degrades to the binary required path.
- Module B called with `with_intervals=with_intervals`; merge uses
  `module_b_new_cols(params, with_intervals=...)`.
- **OOD wiring**: `build_ood_features(df_input)` → `fit_ood_gate` → `score_ood`;
  attaches `ood_score` (float) + `ood_flag` (bool) columns post-merge. When
  `with_ood=False`: `ood_score=NaN`, `ood_flag=False`.
- **`derive_theta_width(df, params)`** — median + 3·MAD of `max_relative_width` over
  non-static-outlier rows; returns 0.0 (inert) when no finite widths. Stored in
  `results["theta_width"]` (None when intervals off).
- **`apply_progressive_refinement(df, params)`** — vectorized form of Module B's
  `resolve_checkpoint` preference rule. Reads `{p}_96h_actual` / `{p}_168h_actual`
  (Prompt B5 naming); per row: actuals override the prediction endpoints; slope window
  is per-row — 168h actual → `(y168 − y24)/144`; 96h actual only → `(y96 − y24)/72`;
  untouched rows keep the original slope. Writes `{p}_96h_pred_refined` /
  `{p}_168h_pred_refined`; **never touches graded `{p}_168h_pred`** (MAE record stays
  comparable). Recomputes `{p}_slope`, `{p}_slope_norm`, `v_drift`, `drift_flag` under
  Module B's rules (same normal mask, same 1e-6 floors). No-op unless ≥1 finite actual.
- Verbose output now prints theta_width + sample REVIEW rows.

### `burnin_screening/validate.py`
- Strict confusion matrix unchanged (FLAG vs not-FLAG).
- Added: `lenient_recall` (FLAG+REVIEW = "not passed"; REVIEW is not a miss), printed
  side by side with strict recall; **DEFERRED to REVIEW** line listing deferred defect
  counts; **REVIEW breakdown** (counts by reason containing "OOD" / "interval"; guarded
  with `if "reason" in review.columns` so hand-built frames don't crash); compound
  breakdown now also reports `compound_deferred`.
- Returns new keys: `lenient_recall`, `n_review`, `deferred_defects`, `compound_deferred`.

### Tests
- `tests/conftest.py`: `df_full` fixture now runs the **fast binary path**
  (`with_intervals=False, with_ood=False`) — MAPIE dominates runtime (~69 s per 80-row
  interval run); required-path semantics are byte-identical with add-ons off. Phase 3
  wiring has its own fixture in the new file.
- `tests/test_decision_maker.py`: binary-decision assertion widened to
  `{PASS, REVIEW, FLAG}` (last test only).
- `tests/test_phase3_review_split.py` (NEW, ~25 tests): OOD REVIEW branch (+n/a score,
  +never-overrides-FLAG ×3, +no-column PASS), width REVIEW branch (+narrow stays PASS,
  +widest-param named, +inert without theta_width, +NaN intervals, +zero pred, +never
  overrides FLAG), `max_relative_width` row-max/missing-cols, `derive_theta_width`
  (median+3MAD exact, 0.0 without intervals), refinement (noop, NaN-actual noop,
  168h override + graded col untouched, 96h-only window, resolve_checkpoint
  integration), pipeline wiring (`df_wired` session fixture: theta_width>0, interval
  cols, ood cols, decision vocab; switches-off test; ood finite/flag-rate), validate
  (lenient vs strict, binary frame unchanged, compound deferred, reason buckets).

### Fixes already applied during verification
- `validate.py`: guarded REVIEW breakdown behind `if "reason" in review.columns`
  (KeyError on hand-built frames).
- `main.py` refinement: slope window is now **per-row** via np.where masks
  (bug: untouched rows got the 72h window when 96h actuals existed for other rows).
- `decision_maker.py`: `_driving_param` .get fallback.

---

## 3. Verification state

- `python -m pytest tests/test_phase3_review_split.py -q` → **22 pass / 3 fail**
  (all other test files untouched and green as of the last full run attempt; full
  suite rerun pending after fixes).
- Manual pipeline check (80 rows, add-ons on): `{PASS: 67, FLAG: 9, REVIEW: 4}`,
  all 4 REVIEWs from the width branch, FN = 0, theta_width ≈ 1.15, OOD flags ~5%.
- Standalone refinement debug: actual=42.0 override reproduces exactly 42.0 for all
  60 rows (exact equality) — see §4 for why one test still fails.

**Environment:** Python 3.14, pytest 9.1.1, MAPIE 1.5 (CrossConformalRegressor),
Windows/bash. Runtime: interval-enabled run_pipeline ≈ 70 s per 80-row call; full
suite ≈ 5–8 min — use `cd burnin_screening && python -m pytest tests/ -q >
/tmp/pytest_out.txt 2>&1; tail -45 /tmp/pytest_out.txt` (piping straight to `tail`
swallowed output once; redirect to a file instead). Use `grep -a` on captured output
(binary-ish bytes from the unicode dash).

---

## 4. IMMEDIATE next steps — 3 known test failures with prepared fixes

All in `burnin_screening/tests/test_phase3_review_split.py`:

### 4.1 `test_refinement_nan_actual_column_is_noop` — TEST bug
Compares `out` (50 cols) against the 49-col fixture; the added `_actual` column is
still present in `out`, so shapes differ by design. Fix — compare against the modified
input, not the fixture:
```python
def test_refinement_nan_actual_column_is_noop(df_refine_base):
    df = df_refine_base.copy()
    df["I_leak_168h_actual"] = np.nan
    before = df.copy(deep=True)
    out = apply_progressive_refinement(df, PARAMS)
    pd.testing.assert_frame_equal(out, before)
    assert not any(c.endswith("_refined") for c in out.columns)
```

### 4.2 `test_validate_lenient_recall_counts_review_as_caught` — wrong expectation
D2's REVIEW is correctly NOT a FLAG in the strict matrix, so `fn == 2`
(D3 PASSed + D2 REVIEWed), recall 0.5, lenient 0.75. Fix:
```python
assert results["fn"] == 2                      # D3 passed + D2 deferred (not FLAG)
assert results["recall"] == pytest.approx(0.5)
assert results["lenient_recall"] == pytest.approx(0.75)  # FLAG+REVIEW
# (keep: deferred_defects == 1, n_review == 1, output-string asserts)
```

### 4.3 `test_refinement_prefers_measured_over_predicted` — comparison mechanism suspect
Assertion `(out["t_pd_168h_pred_refined"] == pytest.approx(actual)).all()` fails, yet
standalone debug shows exact equality on all 60 rows. Visible values print as exactly
42.0; failing rows (52–59) hidden in pytest truncation. Working hypothesis: pytest
9.1.1 `pd.Series == pytest.approx(scalar)` does not do reliable elementwise comparison
(the other passing approx uses in this file are scalar==scalar). Fix — use allclose:
```python
assert np.allclose(out["t_pd_168h_pred_refined"].to_numpy(dtype=float), actual,
                   rtol=1e-9, atol=1e-12)
```
If that still fails, print the deviating rows first:
`print((out["t_pd_168h_pred_refined"] - 42.0).abs().describe())` + `idxmax()` row,
to rule out a real data path bug before changing anything else.

### 4.4 After the 3 fixes
Run the full suite: `cd burnin_screening && python -m pytest tests/ -q` (expect all
green; earlier full-suite failure list was exactly these 3 + the two already-fixed
items in §2).

---

## 5. Remaining work (after tests green)

1. **Docs — add §4f decision contract, IDENTICAL text in BOTH
   `Implementation/person1_static_decision.md` and `Implementation/person2_temporal_prediction.md`**
   (frozen-contract rule: change both together). Draft to insert after §4e:
   - `decide(df, params, theta_slope, theta_width=None) -> df + [decision, reason]`;
     vocabulary `PASS/REVIEW/FLAG`; priority: Prompt A4 FLAG branches 1-4 unchanged →
     5 `ood_flag` → REVIEW → 6 width > theta_width → REVIEW → 7 PASS. REVIEW never
     overrides FLAG (team decision, conservative for FN).
   - Consumes: §4c + §4d columns + `ood_flag`/`ood_score` (attached by main.py),
     `{p}_168h_lower/upper` (only with_intervals=True), `theta_width`.
   - `theta_width = median + 3·MAD` of per-row max relative interval width over the
     normal population (same derivation philosophy as theta_slope); computed in
     main.py via `decision_maker.max_relative_width`.
   - Degradation guarantees: missing ood/interval/theta_width → binary PASS/FLAG
     identical to the required path (absence can never CREATE a REVIEW).
   - main.py wiring: `run_pipeline(..., with_intervals=True, with_ood=True,
     with_progressive_refinement=True)`; OOD on the COMBINED feature set (Module B's
     documented choice); progressive refinement reads `{p}_{h}h_actual` (Prompt B5
     naming), writes `{p}_{h}h_pred_refined`, recomputes slope/v_drift/drift_flag
     per-row, never touches graded `{p}_168h_pred`.
   - validate.py: strict + lenient recall, REVIEW breakdown, deferred defects,
     compound REVIEW count; new result keys.
2. **Full-suite rerun** (§4.4).
3. **Real 500-row demo run** (`cd burnin_screening && python main.py`, ~7–8 min with
   MAPIE) — verify REVIEW distribution, report summary to user; refreshes
   `data/pipeline_output.csv`.
4. Optionally reconcile `Implementation/burnin-architecture-guide.md` add-on #6
   wording with the implemented FLAG-preserving REVIEW (guide says "split FLAG into
   FLAG/REVIEW"; implementation defers only would-PASS rows — worth a one-line note).

## 6. Key context for resume

- Owner mapping: I have access to BOTH persons' files (user said so); Person 1 owns
  decision_maker/main/validate, Person 2 owns module_b_temporal — respect contract
  freeze: §4d names unchanged, only additive changes documented.
- `decision_maker.decide` signature changed (added optional `theta_width`) —
  backward compatible; `conftest.py` imports unaffected.
- `run_pipeline` result dict now also carries `theta_width` — existing consumers
  (`results["theta_slope"]`) unchanged.
- The Phase 2 summary (user message) is the baseline: 61 tests green, MAEs
  0.070/0.064/0.044 on 500 rows, intervals median widths 0.46/0.08/0.40, OOD 5.0%.

---

## 7. FINAL REPORT (completion, 2026-09-24)

### Tests
- The 3 known failures fixed per §4 (2 test bugs + `np.allclose` for the Series
  comparison — pytest's `Series == pytest.approx(scalar)` returned a scalar bool).
- **Full suite: 88 passed** (61 pre-existing + 27 new/updated), zero regressions.

### Docs
- §4f decision contract added **identically** to both person docs (verified same
  insertion point after the §4e freeze sentence). Covers decide() signature/vocabulary,
  7-branch priority, REVIEW-never-overrides-FLAG team decision, theta_width derivation,
  degradation guarantees, run_pipeline wiring incl. progressive refinement, and the
  validate.py REVIEW reporting keys.

### 500-row demo run (data/pipeline_output.csv refreshed)
- Decisions: PASS 412 / FLAG 47 / REVIEW 41 (4 OOD + 37 wide-interval).
- Confusion: TN 452, FP 23, FN 1, TP 24 — strict recall 0.960, lenient 0.960,
  precision 0.511.
- theta_slope 4.8953, theta_width 0.0537.
- **The single FN (C0365, compound case) is PRE-EXISTING**: verified identical PASS
  verdict on the binary required path (with_intervals=False, with_ood=False). Phase 3
  introduced no new false negatives and deferred no defects (deferred_defects=0) — all
  REVIEW rows are normal parts triaged for prediction confidence, exactly the layer's
  designed purpose.
- C0365 remains a coordinated-tuning item for the team (Module A z_threshold/
  contamination vs Module B theta_slope) — deliberately NOT touched in Phase 3, since
  threshold changes alter the frozen required path and must be a joint decision.

---

## 8. POST-PHASE-3 ADDENDUM — C0365 tuning (complete, 2026-09-24)

**Diagnosis (grid over z_threshold x contamination x theta multiplier, n=500/seed=42):**
C0365's 24h Z-scores are near zero (z_fused 0.79, joint flag False) — Module A's
value-domain layers cannot see it; it misses on the drift gate by 0.25
(v_drift 4.43 vs theta 4.68, the 93.5th percentile of normal v_drift). Lowering
z_threshold to 3.0/2.5 or raising contamination to 0.10 adds 10-30 FPs while C0365
STAYS missed (its Z-values are normal — it never crosses Module A). The drift gate is
the only effective FN lever: k=2.4 catches it at +6 FP total, keeping z=3.5/cont=0.05.
Seed sweep (4 seeds x 2 sizes): k=2.4 never worse on FN, FN=0 in 7/8 configs
(k=3.0: 3/8), cost +4-8 FPs (~2-4% of population).

**Shipped changes:**
- `module_b_temporal.run_module_b(..., theta_k=3.0)` — additive MAD-multiplier arg,
  spec default 3.0 (required path byte-identical by default); tuning story in docstring.
- `main.run_pipeline(..., theta_k=2.4)` + CLI `--theta-k`; threaded into
  `derive_theta_slope`, `apply_progressive_refinement` (per-row recompute uses the
  same k). `derive_theta_width` keeps its own 3.0 (width gate was NOT retuned).
- Docs: theta_k documented in §4d of BOTH person docs + Prompt A5 step-6 tuning-order
  note in person1 §8 + Prompt B2 tuning note in person2 §7.
- Tests: `tests/test_theta_tuning.py` (5 tests: spec-default pin, formula agreement,
  monotonic theta/flag behavior, end-to-end threading). **Suite: 93 passed.**
- Demo rerun with k=2.4: PASS 402 / FLAG 58 / REVIEW 40, **FN=0**, all 12 compound
  defects caught, theta_slope 4.5255, precision 0.417 (binary: 35 FP).
- Todos tool state mirrors §4-5 (3 fixes pending, docs, verify).
