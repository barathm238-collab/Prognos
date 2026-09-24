"""Shared synthetic data generator for the burn-in predictive screening project.

Co-owned by Person 1 and Person 2 — write once together, freeze after first commit.

Produces a wide-format dataset matching the frozen schema in §4b of the
implementation documents:

  Per component row:
    component_id, lot_id, N, T_ambient, V_stress
    per parameter p in {I_leak, I_ddq, t_pd}:
      {p}_0h, {p}_24h, {p}_96h, {p}_168h, Y_spec_{p}
  Training-only columns (never present at inference):
    is_defect, defect_case, defect_param

Failure signatures injected (~5% of components total):
  case "single": one parameter drifts sharply — visible at 24h, crosses its
                 Z-threshold and, for a subset, its hard datasheet limit.
  case "compound": ALL THREE parameters drift mildly and simultaneously in the
                   same direction — none individually extreme. This is the
                   latent-defect case that Module A's fusion + Tier 3
                   Isolation Forest and Module B's v_drift exist to catch.

Usage:
    python generate_data.py [--n 500] [--seed 42] [--out burnin_data.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = ["I_leak", "I_ddq", "t_pd"]

# Nominal scales per parameter — used to size lot spread and defect injections
# relative to a physically sensible magnitude.
PARAM_SCALE = {
    "I_leak": 10.0,   # leakage current, ~10 uA nominal
    "I_ddq": 2.0,     # standby current, ~2 mA nominal
    "t_pd": 8.0,      # propagation delay, ~8 ns nominal
}

# Datasheet hard limits (Y_spec) — normal parts sit comfortably below them,
# but a sharply-drifting single-parameter defect can cross.
Y_SPEC = {
    "I_leak": 50.0,
    "I_ddq": 10.0,
    "t_pd": 40.0,
}

DEFECT_FRACTION = 0.05
COMPOUND_SHARE = 0.5          # fraction of defects that are the compound case
# Compound defects share one elevated degradation rate across ALL THREE
# parameters. The multiplier must be wide enough that the elevated rate is
# visible in the 24h reading (upper tail of the lot distribution, z~2-4 per
# parameter) — with the original (1.4, 1.8) range the 24h shift was ~1-3%,
# inside normal noise, making the compound case undetectable by ANY 0h/24h
# model (changed jointly per RESUME.md resume step 2; milder multipliers are
# physically invisible at decision time, not merely hard).
COMPOUND_LAMBDA_MULT = (2.0, 3.0)
SINGLE_STEP_MULT = (3.0, 6.0)       # sharp — visible at 24h
SINGLE_LAMBDA_MULT = (1.05, 1.5)    # bounded post-step continuation to 168h

T_AMBIENT = 125.0
V_STRESS_CHOICES = [3.3, 5.0]
N_LOTS_LARGE = 8   # ~45-55 units each
N_LOTS_SMALL = 2   # ~12-20 units each — small-lot design challenge (N < 30)


def _assign_lots(rng: np.random.Generator, n: int) -> np.ndarray:
    """Assign each component to one of 8 large (~45-55) + 2 small (~12-20) lots."""
    sizes = [int(rng.integers(45, 56)) for _ in range(N_LOTS_LARGE)]
    sizes += [int(rng.integers(12, 20)) for _ in range(N_LOTS_SMALL)]
    lot_ids = np.concatenate([np.full(s, i + 1, dtype=int) for i, s in enumerate(sizes)])
    if len(lot_ids) < n:  # pad in case random sizes fell short
        lot_ids = np.concatenate([lot_ids, np.full(n - len(lot_ids), 1, dtype=int)])
    lot_ids = lot_ids[:n]
    rng.shuffle(lot_ids)
    return lot_ids


def generate_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    lot_ids = _assign_lots(rng, n)
    df = pd.DataFrame({
        "component_id": [f"C{i:04d}" for i in range(1, n + 1)],
        "lot_id": lot_ids,
        "T_ambient": np.full(n, T_AMBIENT),
        "V_stress": rng.choice(V_STRESS_CHOICES, n).astype(float),
    })
    # N = actual lot size (the small lots genuinely have N < 30)
    df["N"] = df["lot_id"].map(pd.Series(lot_ids).value_counts()).astype(int)

    # --- decide who is defective, and which failure signature they get ---
    defect_idx = rng.choice(n, size=int(DEFECT_FRACTION * n), replace=False)
    n_compound = int(COMPOUND_SHARE * len(defect_idx))
    compound_idx = set(int(i) for i in defect_idx[:n_compound])
    single_idx = set(int(i) for i in defect_idx[n_compound:])

    df["is_defect"] = np.isin(np.arange(n), sorted(defect_idx))
    df["defect_case"] = ""
    df.loc[sorted(compound_idx), "defect_case"] = "compound"
    df.loc[sorted(single_idx), "defect_case"] = "single"
    # Guilty parameter for single-parameter defects: deterministic by index,
    # so each sharp defect is sharp in exactly one parameter.
    df["defect_param"] = ""
    for i in sorted(single_idx):
        df.loc[i, "defect_param"] = PARAMS[i % len(PARAMS)]

    # --- generate each parameter's trajectory ---
    for p in PARAMS:
        scale = PARAM_SCALE[p]
        y0h = rng.normal(scale, 0.05 * scale, n)

        # Normal physics: mild exponential degradation. The 0h->24h drift
        # fixes lambda = ln(y24/y0)/24; the same lambda is extrapolated.
        y24h = y0h * (1.02 + rng.normal(0, 0.01, n))
        lam = np.log(y24h / y0h) / 24.0

        y96h = y0h * np.exp(lam * 96.0) * (1 + rng.normal(0, 0.005, n))
        y168h = y0h * np.exp(lam * 168.0)

        # Compound case: ALL THREE parameters drift mildly in the same
        # direction. The same multiplier applies to every parameter of the
        # component — a joint phenomenon by construction, mild per parameter.
        for i in compound_idx:
            mult = rng.uniform(*COMPOUND_LAMBDA_MULT)
            y24h[i] = y0h[i] * np.exp(lam[i] * mult * 24.0)
            y96h[i] = y0h[i] * np.exp(lam[i] * mult * 96.0)
            y168h[i] = y0h[i] * np.exp(lam[i] * mult * 168.0)

        # Single-parameter case: only the guilty parameter steps up sharply
        # at 24h (strong lot-relative Z, sometimes over the hard limit), then
        # CONTINUES drifting at a moderately elevated rate through 96h/168h.
        # The continuation is bounded (y168 = y24 * 1.05-1.5): extrapolating
        # the sharp step exponentially (exp(lam*168) with the stepped lam)
        # produced physically impossible training targets (~1e16 x nominal)
        # that poisoned the XGBoost training targets and destabilised every
        # downstream slope/theta calculation (changed jointly per RESUME.md
        # resume step 2, same coordinated change as COMPOUND_LAMBDA_MULT).
        guilty_rows = [i for i in single_idx if df.loc[i, "defect_param"] == p]
        for i in guilty_rows:
            step = rng.uniform(*SINGLE_STEP_MULT)
            y24h[i] = y24h[i] * step
            y168h[i] = y24h[i] * rng.uniform(*SINGLE_LAMBDA_MULT)
            y96h[i] = y24h[i] * (1 + (y168h[i] / y24h[i] - 1) * (96.0 - 24.0) / (168.0 - 24.0))
            y96h[i] *= 1 + rng.normal(0, 0.005)

        df[f"{p}_0h"] = y0h
        df[f"{p}_24h"] = y24h
        df[f"{p}_96h"] = y96h
        df[f"{p}_168h"] = y168h
        df[f"Y_spec_{p}"] = Y_SPEC[p]

    # Reorder columns: ids/context first, then per-parameter blocks, then labels
    front = ["component_id", "lot_id", "N", "T_ambient", "V_stress"]
    per_param = [c for p in PARAMS for c in
                 (f"{p}_0h", f"{p}_24h", f"{p}_96h", f"{p}_168h", f"Y_spec_{p}")]
    labels = ["is_defect", "defect_case", "defect_param"]
    df = df[front + per_param + labels]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic burn-in data")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "burnin_data.csv"))
    args = parser.parse_args()

    df = generate_dataset(n=args.n, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {len(df)} rows x {len(df.columns)} cols to {out}")
    print(f"  lots: {df['lot_id'].nunique()}, small lots (N<30): "
          f"{(df.groupby('lot_id')['N'].first() < 30).sum()}")
    print(f"  defects: {int(df['is_defect'].sum())} "
          f"(compound: {(df['defect_case'] == 'compound').sum()}, "
          f"single: {(df['defect_case'] == 'single').sum()})")


if __name__ == "__main__":
    main()
