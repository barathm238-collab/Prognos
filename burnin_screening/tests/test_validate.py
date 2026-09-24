"""Tests for validate.validate — the acceptance-test reporting (Prompt A5)."""

from __future__ import annotations

import pandas as pd

from conftest import validate


def _report(capsys):
    return capsys.readouterr().out


def test_perfect_screening_metrics(capsys):
    df = pd.DataFrame({
        "component_id": [f"C{i}" for i in range(6)],
        "is_defect": [True, True, False, False, False, False],
        "decision": ["FLAG", "FLAG", "PASS", "PASS", "PASS", "PASS"],
    })
    results = validate(df)
    out = _report(capsys)
    assert results["fn"] == 0 and results["tp"] == 2
    assert results["fp"] == 0 and results["tn"] == 4
    assert results["recall"] == 1.0 and results["precision"] == 1.0
    assert "FALSE NEGATIVES: 0" in out


def test_false_negatives_reported_prominently(capsys):
    df = pd.DataFrame({
        "component_id": ["C0", "C1", "C2", "C3"],
        "lot_id": [1, 1, 2, 2],
        "is_defect": [True, True, False, False],
        "decision": ["FLAG", "PASS", "PASS", "PASS"],  # C1 is a FN
    })
    results = validate(df)
    out = _report(capsys)
    assert results["fn"] == 1
    assert "FALSE NEGATIVES: 1" in out
    assert "C1" in out  # the missed component is named


def test_compound_breakdown_reported(capsys):
    df = pd.DataFrame({
        "component_id": ["C0", "C1", "C2"],
        "is_defect": [True, True, False],
        "defect_case": ["compound", "single", ""],
        "decision": ["FLAG", "PASS", "PASS"],
    })
    results = validate(df)
    out = _report(capsys)
    assert "compound" in out
    assert results["compound_missed"] == 0  # the compound row was flagged


def test_compound_missed_counted(capsys):
    df = pd.DataFrame({
        "component_id": ["C0", "C1"],
        "is_defect": [True, True],
        "defect_case": ["compound", "compound"],
        "decision": ["PASS", "FLAG"],
    })
    results = validate(df)
    assert results["compound_missed"] == 1


def test_missing_columns_raise():
    try:
        validate(pd.DataFrame({"decision": ["PASS"]}))
    except KeyError as e:
        assert "is_defect" in str(e)
    else:
        raise AssertionError("expected KeyError for missing is_defect")


def test_full_pipeline_report_runs(capsys, df_full):
    """The real merged frame validates without error and reports headline numbers."""
    results = validate(df_full)
    out = _report(capsys)
    assert set(results) >= {"tn", "fp", "fn", "tp", "recall", "precision"}
    assert "VALIDATION REPORT" in out
    assert results["tp"] + results["fn"] == int(df_full["is_defect"].sum())
