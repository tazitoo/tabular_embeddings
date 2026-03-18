"""Tests for inference validation results (01_validate_inference.py)."""

import json
from pathlib import Path

import pytest

REPORT_PATH = Path("output/sae_training_round9/validation_report.json")


@pytest.fixture(scope="module")
def report():
    if not REPORT_PATH.exists():
        pytest.skip("Run 01_validate_inference.py first")
    return json.loads(REPORT_PATH.read_text())


def test_alignment_all_51_ok(report):
    summary = report["summary"]
    assert summary["alignment_ok"] == 51
    assert summary["alignment_mismatch"] == 0
    assert summary["alignment_error"] == 0


def test_alignment_no_size_mismatch(report):
    bad = [r for r in report["alignment"] if r["status"] != "ok"]
    assert not bad, f"Alignment failures: {bad}"


def test_inference_all_ok(report):
    inf = report.get("inference", [])
    if not inf:
        pytest.skip("No inference results (run without --skip-inference)")
    bad = [r for r in inf if r["status"] != "ok"]
    assert not bad, f"Inference failures: {bad}"


def test_inference_accuracy_reasonable(report):
    """Each dataset should have accuracy above 55% (better than chance)."""
    inf = report.get("inference", [])
    if not inf:
        pytest.skip("No inference results")
    for r in inf:
        if "our_accuracy" in r:
            assert r["our_accuracy"] >= 0.55, (
                f"{r['dataset']}: acc={r['our_accuracy']:.3f} below 0.55"
            )
