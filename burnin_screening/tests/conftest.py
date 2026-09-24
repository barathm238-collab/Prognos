"""Shared fixtures and import bootstrap for the burn-in screening tests.

Makes `python -m pytest` work both from `burnin_screening/` and from the repo
root by putting this directory on sys.path (needed because main.py, the
modules, and `data/` are flat here, not a package-relative layout).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from data import generate_data  # noqa: E402  (after sys.path bootstrap)
import main as pipeline_main  # noqa: E402
import module_a_static
import module_b_temporal
from decision_maker import decide  # noqa: E402
from validate import validate  # noqa: E402

PARAMS = ["I_leak", "I_ddq", "t_pd"]

MODULE_A_NEW_COLS = module_a_static.module_a_new_cols(PARAMS)
MODULE_B_NEW_COLS = module_b_temporal.module_b_new_cols(PARAMS)


@pytest.fixture(scope="session")
def df_data() -> pd.DataFrame:
    """Small deterministic synthetic dataset (fast: no models trained here)."""
    return generate_data.generate_dataset(n=200, seed=42)


@pytest.fixture(scope="session")
def Y_spec(df_data) -> dict:
    """Y_spec read from the data itself, exactly as main.py does it."""
    return {p: float(df_data[f"Y_spec_{p}"].iloc[0]) for p in PARAMS}


@pytest.fixture(scope="session")
def df_a(df_data, Y_spec) -> pd.DataFrame:
    """Module A output on the synthetic data (the required-path call)."""
    return module_a_static.run_module_a(df_data, Y_spec, PARAMS)


@pytest.fixture(scope="session")
def df_full(df_data) -> pd.DataFrame:
    """Full pipeline output (Module A + Module B + decision) on the small
dataset — exercises the §4e merge and the Decision Maker end to end.

    Runs the fast binary path (with_intervals=False, with_ood=False): the
    MAPIE conformal fit dominates runtime (cv=5 x 3 XGBoost models) and most
    tests here assert the required-path semantics, which are byte-identical
    when the 🔵 add-ons are off. Phase 3's REVIEW wiring gets its own
    dedicated fixture in test_phase3_review_split.py.
    """
    df_out, _ = pipeline_main.run_pipeline(
        df_data, params=PARAMS, verbose=False,
        with_intervals=False, with_ood=False,
    )
    return df_out


__all__ = [
    "PARAMS",
    "MODULE_A_NEW_COLS",
    "MODULE_B_NEW_COLS",
    "pipeline_main",
    "generate_data",
    "module_a_static",
    "module_b_temporal",
    "decide",
    "validate",
    "np",
    "pd",
    "pytest",
]
