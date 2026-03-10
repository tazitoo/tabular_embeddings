"""Tests for concept description generation."""
import json
import pytest
from unittest.mock import MagicMock


def test_generate_group_description_calls_sonnet():
    """Group description calls Sonnet with correct prompt structure."""
    from scripts.generate_concept_descriptions import describe_group

    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Rows with many zero-valued features and low entropy.")]
    )

    group = {
        "members": [["TabPFN", 305], ["Mitra", 42]],
        "n_models": 2,
        "top_probes": [["frac_zeros", 2, -1.2]],
    }
    high_rows = [{"a": 0.0, "b": 1.0}]
    low_rows = [{"a": 0.5, "b": 0.3}]

    result = describe_group(
        group_id=0, group=group, high_rows=high_rows, low_rows=low_rows,
        client=mock_client, model="claude-sonnet-4-20250514",
    )

    assert "zero-valued" in result
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"


def test_describe_group_falls_back_on_api_error():
    """Returns None on API error without crashing."""
    from scripts.generate_concept_descriptions import describe_group

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("rate limit")

    group = {"members": [], "n_models": 0, "top_probes": []}
    result = describe_group(
        group_id=0, group=group, high_rows=[], low_rows=[],
        client=mock_client, model="claude-sonnet-4-20250514",
    )

    assert result is None


def test_output_schema_has_required_keys():
    """Output JSON has metadata, groups, and unmatched sections."""
    from scripts.generate_concept_descriptions import build_output_skeleton

    skeleton = build_output_skeleton()
    assert "metadata" in skeleton
    assert "groups" in skeleton
    assert "unmatched" in skeleton
    assert "haiku_model" in skeleton["metadata"]
    assert "sonnet_model" in skeleton["metadata"]


def test_checkpoint_resume_skips_completed(tmp_path):
    """Resuming from checkpoint skips already-described groups."""
    from scripts.generate_concept_descriptions import (
        build_output_skeleton,
        load_checkpoint,
        save_checkpoint,
    )

    output = build_output_skeleton()
    output["groups"]["0"] = {
        "brief_label": "sparse rows",
        "summary": "Rows with many zeros.",
    }

    ckpt_path = tmp_path / "checkpoint.json"
    save_checkpoint(output, ckpt_path)

    loaded = load_checkpoint(ckpt_path)
    assert "0" in loaded["groups"]
    assert loaded["groups"]["0"]["brief_label"] == "sparse rows"


def test_call_llm_returns_none_on_error():
    """call_llm returns None on API error."""
    from scripts.generate_concept_descriptions import call_llm

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("timeout")

    result = call_llm(mock_client, "model", "system", "prompt")
    assert result is None


def test_call_llm_returns_text_on_success():
    """call_llm returns stripped text on success."""
    from scripts.generate_concept_descriptions import call_llm

    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="  high correlation  ")]
    )

    result = call_llm(mock_client, "model", "system", "prompt")
    assert result == "high correlation"


def test_build_per_member_detail():
    """Per-member detail extracts R2 and top probes from probes data."""
    from scripts.generate_concept_descriptions import _build_per_member_detail

    probes_data = {
        "models": {
            "TabPFN": {
                "per_feature": {
                    "305": {
                        "r2": 0.45,
                        "top_probes": [
                            ["frac_zeros", 1.2, 1],
                            ["skewness", -0.5, 2],
                        ],
                    }
                }
            }
        }
    }

    members = [["TabPFN", 305], ["Mitra", 42]]
    detail = _build_per_member_detail(members, probes_data)

    assert len(detail) == 1  # Mitra:42 not in probes_data
    assert detail[0]["model"] == "TabPFN"
    assert detail[0]["feature_idx"] == 305
    assert detail[0]["r2"] == 0.45


def test_find_best_member():
    """Finds member with highest R2."""
    from scripts.generate_concept_descriptions import _find_best_member

    probes_data = {
        "models": {
            "TabPFN": {
                "per_feature": {
                    "305": {"r2": 0.45},
                    "61": {"r2": 0.72},
                }
            },
            "Mitra": {
                "per_feature": {
                    "42": {"r2": 0.30},
                }
            },
        }
    }

    members = [["TabPFN", 305], ["TabPFN", 61], ["Mitra", 42]]
    best = _find_best_member(members, probes_data)
    assert best == ("TabPFN", 61)


def test_run_pass1_skips_completed():
    """Pass 1 skips groups that already have brief_label."""
    from scripts.generate_concept_descriptions import run_pass1

    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="new label")]
    )

    labels = {
        "concept_groups": {
            "0": {
                "members": [["TabPFN", 1]],
                "n_models": 1,
                "top_probes": [["frac_zeros", 1, -0.5]],
            },
            "1": {
                "members": [["Mitra", 2]],
                "n_models": 1,
                "top_probes": [],
            },
        }
    }
    probes_data = {"models": {}}

    output = {
        "metadata": {"n_haiku_calls": 0, "n_sonnet_calls": 0},
        "groups": {
            "0": {"brief_label": "already done"},
        },
        "unmatched": {},
    }

    n = run_pass1(mock_client, labels, probes_data, output)

    # Should only generate for group "1", not "0"
    assert n == 1
    assert output["groups"]["0"]["brief_label"] == "already done"
    assert output["groups"]["1"]["brief_label"] == "new label"


def test_checkpoint_roundtrip_preserves_data(tmp_path):
    """Checkpoint save/load preserves all data faithfully."""
    from scripts.generate_concept_descriptions import (
        build_output_skeleton,
        load_checkpoint,
        save_checkpoint,
    )

    output = build_output_skeleton()
    output["groups"]["42"] = {
        "brief_label": "high kurtosis",
        "summary": "Rows with heavy-tailed distributions.",
        "n_models": 3,
        "n_members": 12,
    }
    output["unmatched"]["TabPFN:99"] = {
        "model": "TabPFN",
        "feature_idx": 99,
        "r2": 0.02,
        "summary": "Unknown pattern.",
    }
    output["metadata"]["n_haiku_calls"] = 5
    output["metadata"]["n_sonnet_calls"] = 3

    ckpt_path = tmp_path / "test_ckpt.json"
    save_checkpoint(output, ckpt_path)
    loaded = load_checkpoint(ckpt_path)

    assert loaded["groups"]["42"]["brief_label"] == "high kurtosis"
    assert loaded["groups"]["42"]["summary"] == "Rows with heavy-tailed distributions."
    assert loaded["unmatched"]["TabPFN:99"]["r2"] == 0.02
    assert loaded["metadata"]["n_haiku_calls"] == 5
    assert loaded["metadata"]["n_sonnet_calls"] == 3


def test_label_key_to_model_key_complete():
    """All 7 models have entries in the mapping."""
    from scripts.generate_concept_descriptions import LABEL_KEY_TO_MODEL_KEY

    expected = {"TabPFN", "Mitra", "TabICL", "TabDPT", "HyperFast", "CARTE", "Tabula-8B"}
    assert set(LABEL_KEY_TO_MODEL_KEY.keys()) == expected
