#!/usr/bin/env python3
"""Ring-consensus mesh labeling of SAE features from contrastive example CSVs.

Topology
    n agents (one per dataset where feature f_i fires) arranged in a ring.
    Round 1: each agent sees its own CSV + preprocessing context, writes a
             proto-label (10-25 words).
    Round k (k >= 2): each agent sees its own CSV AND peer (i - peer_offset)
             at offset (k-1) mod n: previous label + peer's dataset summary.
    Synthesizer (final step): a single agent receives all n final labels and
             all n CSVs, writes a consensus label that abstracts across
             datasets. Used once the ring plateaus or after n-1 rounds.

Convergence / stopping
    After each round, embed all n current labels with nomic-embed-text-v1.5
    and compute mean pairwise cosine. Stop when cosine non-increasing for
    two rounds, or at MAX_ROUNDS. Threshold for "converged": 0.80.

Final label
    If synthesis recorded -> synthesized label.
    Else -> most-central label from the highest-cosine round (not the last).

Agent dispatch happens in Claude Code conversation; this module prepares
prompts, persists state, and scores similarity.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts._project_root import PROJECT_ROOT

CONTRASTIVE_DIR = PROJECT_ROOT / "output" / "contrastive_examples"
CONVERGENCE_THRESHOLD = 0.80  # diagnostic only; judge decides when to stop
MAX_ROUNDS = 5
LABEL_WORDS_MIN, LABEL_WORDS_MAX = 10, 25
VALID_LABEL_FORMATS = {"sentence", "rules"}
DEFAULT_LABEL_FORMAT = "sentence"
NOMIC_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
JUDGE_SAMPLE_SEED = 13
PEER_SAMPLE_N_ACT = 3
PEER_SAMPLE_N_CON = 3
JUDGE_SAMPLE_N_ACT = 2
JUDGE_SAMPLE_N_CON = 2
RINGLITE_SELF_SAMPLE_N_ACT = 3
RINGLITE_SELF_SAMPLE_N_CON = 3

# Section ordering for round1_prompt and ring_prompt. Set via `init --prompt-order`.
#   A — evidence before instructions (current baseline).
#       header | preprocessing | data | task | shape-only | contrast-discipline | output
#   B — task and constraints first, then evidence.
#       header | task | output | shape-only | contrast-discipline | preprocessing | data
#   C — evidence, contrast discipline, shape-only, then task and output last.
#       header | preprocessing | data | contrast-discipline | shape-only | task | output
# Ring rounds also include peer-evidence + judge-feedback blocks; those ride with
# "data" in all three orders.
PROMPT_ORDER_SEQUENCES = {
    "A": ["header", "preprocessing", "data", "peer", "judge_feedback", "task", "shape", "contrast", "output"],
    "B": ["header", "task", "output", "shape", "contrast", "preprocessing", "data", "peer", "judge_feedback"],
    "C": ["header", "preprocessing", "data", "peer", "judge_feedback", "contrast", "shape", "task", "output"],
}
VALID_PROMPT_ORDERS = set(PROMPT_ORDER_SEQUENCES)
DEFAULT_PROMPT_ORDER = "A"

VALID_ARCHES = {"baseline", "ringlite", "ringlite_freeze"}
DEFAULT_ARCH = "baseline"

VALID_PAIRINGS = {"on", "off"}
DEFAULT_PAIRINGS = "on"

_PREDICTION_COLUMNS = {
    "pred_class",
    "pred_conf",
    "pred_correct",
    "pred_value",
    "pred_abs_err",
}

_NOMIC_MODEL = None


def _nomic_model():
    global _NOMIC_MODEL
    if _NOMIC_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _NOMIC_MODEL = SentenceTransformer(NOMIC_MODEL_NAME, trust_remote_code=True)
    return _NOMIC_MODEL


def nomic_embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings for a list of short texts."""
    model = _nomic_model()
    prefixed = [f"clustering: {t}" for t in texts]
    return model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True)


def mean_pairwise_cosine(labels: list[str]) -> float:
    if len(labels) < 2:
        return 1.0
    embs = nomic_embed(labels)
    sims = embs @ embs.T
    n = len(labels)
    off = sims[~np.eye(n, dtype=bool)]
    return float(off.mean())


@dataclass
class ContrastiveMeshPipeline:
    model: str
    feat_idx: int
    contrastive_dir: Path = CONTRASTIVE_DIR
    arch: str = DEFAULT_ARCH
    prompt_order: str = DEFAULT_PROMPT_ORDER
    label_format: str = DEFAULT_LABEL_FORMAT
    pairings_mode: str = DEFAULT_PAIRINGS

    def __post_init__(self):
        # Validate any explicitly-passed knob values; state overrides on resume.
        if self.arch not in VALID_ARCHES:
            raise ValueError(f"arch={self.arch!r} not one of {sorted(VALID_ARCHES)}")
        if self.prompt_order not in VALID_PROMPT_ORDERS:
            raise ValueError(f"prompt_order={self.prompt_order!r} not one of {sorted(VALID_PROMPT_ORDERS)}")
        if self.label_format not in VALID_LABEL_FORMATS:
            raise ValueError(f"label_format={self.label_format!r} not one of {sorted(VALID_LABEL_FORMATS)}")
        if self.pairings_mode not in VALID_PAIRINGS:
            raise ValueError(f"pairings_mode={self.pairings_mode!r} not one of {sorted(VALID_PAIRINGS)}")
        self._model_dir = self.contrastive_dir / self.model
        ctx_path = self._model_dir / f"f{self.feat_idx}_context.json"
        self.feat_context = json.loads(ctx_path.read_text())
        self.datasets: list[str] = self.feat_context["datasets_used"]
        self.preprocessing = self.feat_context["preprocessing"]
        self.csv_paths = {ds: self._model_dir / f"f{self.feat_idx}_{ds}.csv"
                          for ds in self.datasets}
        self._state_path = self._model_dir / f"f{self.feat_idx}_mesh_state.json"
        self.rounds: list[list[str]] = []
        self.similarities: list[float] = []
        self.judge_verdicts: list[dict] = []  # [{"verdict": "done"|"continue", "note": str}, ...]
        self.synthesis: Optional[str] = None
        self.validator_results: Optional[dict] = None
        self._load_state_if_exists()

    # ── State persistence ──────────────────────────────────────────────
    def _load_state_if_exists(self):
        """When a state file exists, its knob values override the constructor
        defaults so subcommands after `init` always reflect the init config."""
        if not self._state_path.exists():
            return
        data = json.loads(self._state_path.read_text())
        if data.get("feat_idx") != self.feat_idx or data.get("model") != self.model:
            return
        self.arch = data.get("arch", DEFAULT_ARCH)
        self.prompt_order = data.get("prompt_order", DEFAULT_PROMPT_ORDER)
        self.label_format = data.get("label_format", DEFAULT_LABEL_FORMAT)
        self.pairings_mode = data.get("pairings_mode", DEFAULT_PAIRINGS)
        self.rounds = data.get("rounds", [])
        self.similarities = data.get("similarities", [])
        self.judge_verdicts = data.get("judge_verdicts", [])
        self.synthesis = data.get("synthesis")
        self.validator_results = data.get("validator_results")

    def _save_state(self):
        self._state_path.write_text(json.dumps({
            "model": self.model,
            "feat_idx": self.feat_idx,
            "arch": self.arch,
            "prompt_order": self.prompt_order,
            "label_format": self.label_format,
            "pairings_mode": self.pairings_mode,
            "datasets": self.datasets,
            "rounds": self.rounds,
            "similarities": self.similarities,
            "judge_verdicts": self.judge_verdicts,
            "synthesis": self.synthesis,
            "validator_results": self.validator_results,
        }, indent=2))

    def reset(self):
        self.rounds = []
        self.similarities = []
        self.judge_verdicts = []
        self.synthesis = None
        self.validator_results = None
        if self._state_path.exists():
            self._state_path.unlink()

    # ── Prompt construction ────────────────────────────────────────────
    def _preprocessing_block(self) -> str:
        pp = self.preprocessing
        steps = "\n  - ".join(pp.get("preprocessing", []))
        return (
            f"Model: {pp.get('model', self.model)}\n"
            f"Architecture: {pp.get('architecture', '?')}\n"
            f"Preprocessing:\n  - {steps}\n"
            f"Implication: {pp.get('implication', '?')}"
        )

    @staticmethod
    def _drop_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in df.columns if c not in _PREDICTION_COLUMNS]
        return df.loc[:, cols]

    def _prompt_csv_text(self, df: pd.DataFrame) -> str:
        return self._drop_prediction_columns(df).to_csv(index=False)

    def _dataset_block(self, ds: str, *, full_csv: bool) -> str:
        meta = self.feat_context["dataset_stats"].get(ds, {})
        target = meta.get("target_summary") or {}
        header = (
            f"Dataset: {ds} ({meta.get('task_type', '?')}), "
            f"n_num={meta.get('nr_num', '?')}, n_cat={meta.get('nr_cat', '?')}, "
            f"n_bin={meta.get('nr_bin', '?')}, n_classes={meta.get('nr_class', '?')}, "
            f"n_attrs={meta.get('nr_attr', '?')}"
        )
        if target.get("task") == "classification":
            cf = target.get("class_freq", {})
            cf_str = ", ".join(f"y={k}:{v:.3f}" for k, v in cf.items())
            header += f"\nTarget base rates (train, n={target.get('n')}): {cf_str}"
        elif target.get("task") == "regression":
            header += (
                f"\nTarget distribution (train, n={target.get('n')}): "
                f"mean={target.get('mean')}, std={target.get('std')}, "
                f"p25={target.get('p25')}, p50={target.get('p50')}, p75={target.get('p75')}"
            )
        if full_csv:
            csv_text = self._prompt_csv_text(pd.read_csv(self.csv_paths[ds]))
            note = (
                "Each numeric cell is annotated as 'value (pXX)' = pXX percentile in the SAE "
                "training split. Each categorical/binary cell is 'value (freq Y.YY)' = prevalence "
                "of that value in training.\n"
            )
            return f"{header}\n\n{note}\nContrastive CSV ({self.csv_paths[ds].name}):\n{csv_text}"
        df = pd.read_csv(self.csv_paths[ds])
        act = df[df.label == "activating"]
        con = df[df.label == "contrast"]
        band_counts = act["band"].value_counts().to_dict() if "band" in df.columns else {}
        band_str = ", ".join(f"{b}={c}" for b, c in sorted(band_counts.items())) or "unstratified"
        return (
            f"{header}\n"
            f"  activating rows: {len(act)} ({band_str}; "
            f"activation range [{act['activation'].min():.2f}, {act['activation'].max():.2f}])\n"
            f"  contrast rows: {len(con)}"
        )

    def _sampled_dataset_block(
        self,
        ds: str,
        *,
        round_num: int,
        n_act: int,
        n_con: int,
        purpose: str,
        label: str,
    ) -> str:
        meta = self.feat_context["dataset_stats"].get(ds, {})
        target = meta.get("target_summary") or {}
        header = (
            f"Dataset: {ds} ({meta.get('task_type', '?')}), "
            f"n_num={meta.get('nr_num', '?')}, n_cat={meta.get('nr_cat', '?')}, "
            f"n_bin={meta.get('nr_bin', '?')}, n_classes={meta.get('nr_class', '?')}, "
            f"n_attrs={meta.get('nr_attr', '?')}"
        )
        if target.get("task") == "classification":
            cf = target.get("class_freq", {})
            cf_str = ", ".join(f"y={k}:{v:.3f}" for k, v in cf.items())
            header += f"\nTarget base rates (train, n={target.get('n')}): {cf_str}"
        elif target.get("task") == "regression":
            header += (
                f"\nTarget distribution (train, n={target.get('n')}): "
                f"mean={target.get('mean')}, std={target.get('std')}, "
                f"p25={target.get('p25')}, p50={target.get('p50')}, p75={target.get('p75')}"
            )
        samples = self._sample_rows_for(round_num, ds, n_act, n_con, purpose=purpose)
        note = (
            "Numeric cells: 'value (pXX)' = pXX percentile in training.\n"
            "Categorical cells: 'value (freq Y.YY)' = training prevalence.\n"
            "These rows are a normalized evidence packet for this round, not the full CSV.\n"
        )
        return (
            f"{header}\n\n{note}\n"
            f"{label} {n_act} sampled activating rows:\n{samples['act']}\n"
            f"{label} {n_con} sampled contrast rows:\n{samples['con']}"
        )

    # ── Shared section builders (for round1 and ring prompts) ──────────

    @staticmethod
    def _shape_only_block() -> str:
        return (
            "CRITICAL INSTRUCTION — describe SHAPE and STATISTICAL FINGERPRINT only:\n"
            "  * IGNORE column names and the domain meaning they suggest.\n"
            "  * IGNORE column ordering. The pattern should be invariant under permutation of columns.\n"
            "  * Use the marginal annotations (pXX / freq) as your primary source — these are\n"
            "    dataset-agnostic: 'p85' means the same thing whether the column is a magnitude,\n"
            "    a permission bit, or an age.\n"
            "  * Describe distributional properties: do activating rows concentrate at high/low/mid\n"
            "    percentiles? At rare/common categorical levels? Across many columns or a narrow\n"
            "    subset? With tight or dispersed within-row spread of percentiles?\n"
            "  * Do NOT mention specific column names or domain terms."
        )

    @staticmethod
    def _rules_block() -> str:
        return (
            "RULE-SET TARGET\n"
            "  * Output a necessary-and-sufficient RULE SET for when this concept is present.\n"
            "  * The rule set should express the concept's MEANING overall, not activation strength.\n"
            "  * Rules must be semantic/structural invariants that could hold across datasets even if\n"
            "    the surface manifestation differs.\n"
            "  * Avoid directional activation phrasing like 'higher than contrast', 'stronger', 'more\n"
            "    active', or any claim whose meaning depends on activation magnitude rather than the\n"
            "    row structure itself.\n"
            "  * Do not target a specific rule count. Include exactly the rules needed to be necessary\n"
            "    and sufficient; no more, no fewer.\n"
            "  * Final rules must be direct row-level structural conditions, not commentary about the\n"
            "    evidence or the pipeline.\n"
        )

    @staticmethod
    def _contrast_discipline_block() -> str:
        return (
            "CONTRAST DISCIPLINE — include only distinguishing properties:\n"
            "  * Before adding ANY property to the label, verify activating rows DIFFER from\n"
            "    contrast rows on it — in any combination of inputs (percentile positions,\n"
            "    categorical frequencies, within-row spread, column-subset co-firing), or labels\n"
            "    (target values, base-rate enrichment).\n"
            "  * Properties shared between activating and contrast are NOT feature-level signals\n"
            "    and must not appear in the label.\n"
            "  * Keep looking — properties can combine (an interaction between two inputs; a\n"
            "    target × percentile pair; a within-row spread × rarity regime).\n"
            "    Exhaust the combinations before concluding anything is shared."
        )

    def _output_block(self) -> str:
        if self.label_format == "rules":
            return (
                "Output ONLY the rule set. No preamble, no explanation.\n"
                "Format exactly as plain text:\n"
                "RULE SET\n"
                "- <rule 1>\n"
                "- <rule 2>\n"
                "- ...\n"
                "Each bullet must be a direct row-level structural rule.\n"
                "Do NOT include commentary such as 'strongest subtype', 'secondary subtype',\n"
                "'not portable', 'heterogeneous', 'no single rule survives', or any similar\n"
                "analysis of the evidence."
            )
        return (
            f"Output ONLY a single-sentence label ({LABEL_WORDS_MIN}-{LABEL_WORDS_MAX} words) "
            "describing the structural / distributional fingerprint. No column names. No domain terms. "
            "No preamble, no explanation."
        )

    @staticmethod
    def _assemble(blocks: dict, order: str) -> str:
        """Join non-empty blocks in the sequence defined by prompt_order."""
        seq = PROMPT_ORDER_SEQUENCES[order]
        parts = []
        for key in seq:
            block = blocks.get(key)
            if block:
                parts.append(block.rstrip())
        return "\n\n".join(parts).strip()

    def round1_prompt(self, ds: str) -> str:
        header = (
            f"You are labeling SAE feature f_{self.feat_idx} for a tabular foundation model.\n"
            f"This is round 1 of ring-consensus mesh labeling; you are the agent for '{ds}'."
        )
        task = (
            "TASK\n"
            "The CSV has a 'band' column: 'top' (highest activations), 'p90' (~90th percentile\n"
            "of activating rows), 'p80' (~80th percentile), or 'contrast' (feature does NOT fire).\n"
            "Each cell is annotated with its marginal position in the SAE training split:\n"
            "  - numeric: 'value (pXX)' means XX-th percentile\n"
            "  - categorical/binary: 'value (freq Y.YY)' means that category/value occurs at rate Y.YY\n"
            "The dataset header shows the training-split target base rates. Compare the target\n"
            "values seen in the activating rows to the base rate to judge target enrichment.\n"
            "Note whether the pattern holds at p80/p90 too, or only at top activations."
        )
        blocks = {
            "header": header,
            "preprocessing": self._preprocessing_block(),
            "data": self._dataset_block(ds, full_csv=True),
            "task": task,
            "shape": self._shape_only_block(),
            "contrast": self._contrast_discipline_block(),
            "output": self._output_block(),
        }
        if self.label_format == "rules":
            blocks["task"] += (
                "\nReturn a monosemantic necessary-and-sufficient rule set, not a polished sentence."
            )
            blocks["shape"] = f"{self._shape_only_block()}\n\n{self._rules_block()}"
        return self._assemble(blocks, self.prompt_order)

    def _peer_evidence_block(self, peer_ds: str, round_num: int) -> str:
        """Render peer evidence as header + 3 activating + 3 contrast rows.

        Replaces passing the full CSV, which (a) bloats context 5× and
        (b) lets peers pattern-match on incidental rows rather than generalize.
        """
        return self._sampled_dataset_block(
            peer_ds,
            round_num=round_num,
            n_act=PEER_SAMPLE_N_ACT,
            n_con=PEER_SAMPLE_N_CON,
            purpose="peer",
            label="peer's",
        )

    @staticmethod
    def _claim_text(c) -> str:
        """Render a claim as either '[certainty] claim' or plain string.

        Accepts both the new {claim, certainty} dict shape and legacy
        plain-string claims in old state files.
        """
        if isinstance(c, dict):
            cert = c.get("certainty", "").strip()
            claim = c.get("claim", "").strip()
            return f"[{cert}] {claim}" if cert else claim
        return str(c)

    def _judge_feedback_block(self, ds: str, round_num: int) -> str:
        """Render the judge's previous-round feedback for one dataset.

        `round_num` here is the *current* (in-progress) round; feedback is
        pulled from judge_verdicts[round_num - 2] (the round just judged).
        Returns an empty string if no prior verdict exists.
        """
        prev_idx = round_num - 2
        if prev_idx < 0 or prev_idx >= len(self.judge_verdicts):
            return ""
        v = self.judge_verdicts[prev_idx]
        per_ds = (v.get("per_dataset") or {}).get(ds)
        if not per_ds:
            return ""
        pr = per_ds.get("portability_risk") or {}
        pr_str = (
            f"{pr.get('level', '?')} — {pr.get('justification', '')}"
            if isinstance(pr, dict) else str(pr)
        )
        def _fmt_list(items):
            items = items or []
            if not items:
                return "  (none)"
            return "\n".join(f"  - {self._claim_text(x)}" for x in items)
        return (
            f"JUDGE FEEDBACK ON '{ds}' (from round {round_num - 1})\n"
            f"overall verdict: {v.get('verdict')}\n"
            f"overall note: {v.get('note', '')}\n"
            f"per-dataset verdict: {per_ds.get('verdict', '?')}\n"
            f"supported claims:\n{_fmt_list(per_ds.get('supported_claims'))}\n"
            f"unsupported claims:\n{_fmt_list(per_ds.get('unsupported_claims'))}\n"
            f"contradicted claims:\n{_fmt_list(per_ds.get('contradicted_claims'))}\n"
            f"missing signal: {self._claim_text(per_ds.get('missing_signal')) or '(none)'}\n"
            f"portability risk: {pr_str}\n"
            f"suggested revision: {per_ds.get('suggested_revision', '') or '(none)'}\n"
        )

    def ring_prompt(self, ds: str, peer_ds: str, peer_label: str, round_num: int) -> str:
        header = (
            f"You are the agent for '{ds}', labeling SAE feature f_{self.feat_idx}.\n"
            f"This is round {round_num} of ring-consensus mesh labeling."
        )
        if self.arch == "baseline":
            data = f"YOUR DATA ROWS\n{self._dataset_block(ds, full_csv=True)}"
        else:
            sampled_self = self._sampled_dataset_block(
                ds,
                round_num=round_num,
                n_act=RINGLITE_SELF_SAMPLE_N_ACT,
                n_con=RINGLITE_SELF_SAMPLE_N_CON,
                purpose="self",
                label="your",
            )
            data = (
                "YOUR DATA ROWS\n"
                f"{sampled_self}"
            )
        peer_block = (
            f"PEER EVIDENCE (from '{peer_ds}')\n"
            f"  peer's current label: \"{peer_label}\"\n\n"
            f"{self._peer_evidence_block(peer_ds, round_num)}"
        )
        own_feedback = self._judge_feedback_block(ds, round_num)
        peer_feedback = self._judge_feedback_block(peer_ds, round_num)
        feedback_block = ""
        if own_feedback or peer_feedback:
            parts = []
            if own_feedback:
                parts.append(own_feedback.rstrip())
            if peer_feedback:
                parts.append(peer_feedback.rstrip())
            feedback_block = "\n\n".join(parts)
        task = (
            "TASK\n"
            f"Revise your {'rule set' if self.label_format == 'rules' else 'label'} in light of the peer's hypothesis and the judge's feedback above.\n"
            "Address unsupported / contradicted claims; incorporate the missing signal if the "
            "evidence supports it; apply the suggested revision unless your data contradicts it.\n"
            "If the peer's shape-level pattern also fits your activating rows, merge framings. "
            "If your data contradicts theirs at the shape level, hold your ground but sharpen.\n"
            f"Goal: a {'necessary-and-sufficient rule set' if self.label_format == 'rules' else 'label'} expressible in structural terms that fits both datasets."
        )
        output = self._output_block()
        blocks = {
            "header": header,
            "preprocessing": self._preprocessing_block(),
            "data": data,
            "peer": peer_block,
            "judge_feedback": feedback_block,
            "task": task,
            "shape": self._shape_only_block(),
            "contrast": self._contrast_discipline_block(),
            "output": output,
        }
        if self.label_format == "rules":
            blocks["shape"] = f"{self._shape_only_block()}\n\n{self._rules_block()}"
        return self._assemble(blocks, self.prompt_order)

    def round_prompts(self, round_num: int) -> list[str]:
        n = len(self.datasets)
        if round_num == 1:
            return [self.round1_prompt(ds) for ds in self.datasets]
        if len(self.rounds) < round_num - 1:
            raise ValueError(
                f"Cannot build round {round_num} prompts: "
                f"only {len(self.rounds)} rounds recorded"
            )
        prev_labels = self.rounds[round_num - 2]
        peer_offset = ((round_num - 1) - 1) % max(n - 1, 1) + 1  # 1,2,...,n-1,1,...
        # Prior judge verdict (from the round just judged) may contain
        # explicit pairings that override the default ring rotation per
        # dataset. Unset datasets fall back to the fixed offset.
        prior_pairings: dict = {}
        if len(self.judge_verdicts) >= round_num - 1 and round_num >= 2:
            prior_pairings = (
                self.judge_verdicts[round_num - 2].get("next_round_pairings") or {}
            )
        active = self.active_datasets_for_round(round_num)
        prompts: list[str] = []
        for i, ds in enumerate(self.datasets):
            if ds not in active:
                continue
            if ds in prior_pairings and prior_pairings[ds] in self.datasets:
                peer_ds = prior_pairings[ds]
                peer_idx = self.datasets.index(peer_ds)
            else:
                peer_idx = (i - peer_offset) % n
                peer_ds = self.datasets[peer_idx]
            prompts.append(
                self.ring_prompt(
                    ds=ds,
                    peer_ds=peer_ds,
                    peer_label=prev_labels[peer_idx],
                    round_num=round_num,
                )
            )
        return prompts

    def active_datasets_for_round(self, round_num: int) -> list[str]:
        """Datasets that need worker prompts for a given round."""
        if round_num <= 1 or self.arch != "ringlite_freeze":
            return list(self.datasets)
        prev_idx = round_num - 2
        if prev_idx < 0 or prev_idx >= len(self.judge_verdicts):
            return list(self.datasets)
        per_ds = self.judge_verdicts[prev_idx].get("per_dataset") or {}
        active = [
            ds for ds in self.datasets
            if (per_ds.get(ds) or {}).get("verdict") != "accept"
        ]
        return active or list(self.datasets)

    # ── State updates ──────────────────────────────────────────────────
    def record_round(
        self, labels: list[str], *, round_num: Optional[int] = None
    ) -> float:
        if round_num is None:
            round_num = len(self.rounds) + 1
        if round_num != len(self.rounds) + 1:
            raise ValueError(
                f"Next round is {len(self.rounds) + 1}, got round_num={round_num}"
            )
        active = self.active_datasets_for_round(round_num)
        if round_num == 1 or self.arch != "ringlite_freeze":
            expected = len(self.datasets)
            if len(labels) != expected:
                raise ValueError(f"Expected {expected} labels, got {len(labels)}")
            clean = [_strip_label(l) for l in labels]
        else:
            expected = len(active)
            if len(labels) != expected:
                raise ValueError(
                    f"ARCH={self.arch} round {round_num} expects {expected} labels "
                    f"for active datasets {active}, got {len(labels)}"
                )
            if not self.rounds:
                raise ValueError("Freeze mode requires a prior round to carry frozen labels forward")
            merged = list(self.rounds[-1])
            active_iter = iter(_strip_label(l) for l in labels)
            for i, ds in enumerate(self.datasets):
                if ds in active:
                    merged[i] = next(active_iter)
            clean = merged
        self.rounds.append(clean)
        sim = mean_pairwise_cosine(clean)
        self.similarities.append(sim)
        self._save_state()
        return sim

    # ── Judge-gated iteration ──────────────────────────────────────────
    def _sample_rows_for(
        self,
        round_num: int,
        ds: str,
        n_act: int,
        n_con: int,
        purpose: str = "judge",
    ) -> dict:
        """Deterministically sample activating + contrast rows for (round, ds).

        Activating rows are stratified across bands (top / p90 / p80) when
        possible so the sample reflects the full firing range.
        Seeds are derived from round + stable hash(ds) + purpose so peer
        and judge samples are independent (prevents agents from memorising
        the judge's sample set).
        Returns {'act': csv_text_with_header, 'con': csv_text_with_header}.
        """
        import hashlib
        seed_key = f"{purpose}:{ds}:{round_num}"
        seed = int(hashlib.md5(seed_key.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.default_rng(seed)
        df = pd.read_csv(self.csv_paths[ds])
        act = df[df.label == "activating"].reset_index(drop=True)
        con = df[df.label == "contrast"].reset_index(drop=True)

        # Stratified activating sample: one per band first, then top-up randomly.
        act_rows = []
        if "band" in act.columns:
            for band in ("top", "p90", "p80"):
                pool = act[act.band == band]
                if len(pool) == 0 or len(act_rows) >= n_act:
                    continue
                act_rows.append(pool.iloc[int(rng.integers(len(pool)))])
        remaining_act_ids = set(range(len(act))) - {r.name for r in act_rows}
        while len(act_rows) < n_act and remaining_act_ids:
            pick = int(rng.choice(list(remaining_act_ids)))
            act_rows.append(act.iloc[pick])
            remaining_act_ids.discard(pick)

        # Contrast: simple random without replacement.
        n_con_take = min(n_con, len(con))
        con_idx = rng.choice(len(con), size=n_con_take, replace=False) if n_con_take else []
        con_rows = [con.iloc[int(i)] for i in con_idx]

        def _to_csv(rows):
            if not rows:
                return ""
            return self._prompt_csv_text(pd.DataFrame(rows))

        return {"act": _to_csv(act_rows), "con": _to_csv(con_rows)}

    def _prior_rounds_history_block(self, ds: str, up_to_round: int) -> str:
        """Render prior-round judge observations for one dataset, for stability judging.

        Shows the agent's past labels and the judge's past supported /
        unsupported / contradicted claims across rounds 1..up_to_round-1,
        so the current judge can mark claims as stable-across-rounds vs
        one-round observations.
        """
        if up_to_round <= 1 or not self.judge_verdicts:
            return ""
        ds_i = self.datasets.index(ds)
        lines = [f"PRIOR-ROUND OBSERVATIONS for '{ds}' (rounds 1 .. {up_to_round - 1}):"]
        for r_idx in range(up_to_round - 1):
            if r_idx >= len(self.judge_verdicts):
                break
            prior_label = self.rounds[r_idx][ds_i] if r_idx < len(self.rounds) else ""
            pd_block = (self.judge_verdicts[r_idx].get("per_dataset") or {}).get(ds, {})
            sup = pd_block.get("supported_claims") or []
            unsup = pd_block.get("unsupported_claims") or []
            contr = pd_block.get("contradicted_claims") or []
            lines.append(f"  round {r_idx + 1}:")
            lines.append(f"    label: \"{prior_label}\"")
            if sup:
                lines.append(f"    supported: {'; '.join(self._claim_text(c) for c in sup)}")
            if unsup:
                lines.append(f"    unsupported: {'; '.join(self._claim_text(c) for c in unsup)}")
            if contr:
                lines.append(f"    contradicted: {'; '.join(self._claim_text(c) for c in contr)}")
        return "\n".join(lines)

    def judge_prompt(self, round_num: Optional[int] = None) -> str:
        """Build a judge prompt over the current (or specified) round's labels.

        The judge sees each agent's label plus sampled activating + contrast
        rows. The judge does NOT see the full CSVs — by design — so it
        verifies each label's claims against fresh samples rather than
        becoming the synthesizer. From round 2 on, the judge also sees
        PRIOR-ROUND observations per dataset to tag claim certainty
        (local observation / emerging pattern / stable cross-round pattern).
        """
        if round_num is None:
            round_num = len(self.rounds)
        if round_num < 1 or round_num > len(self.rounds):
            raise ValueError(f"Round {round_num} not recorded yet")
        labels = self.rounds[round_num - 1]
        sections = []
        for ds, lab in zip(self.datasets, labels):
            samples = self._sample_rows_for(
                round_num, ds, JUDGE_SAMPLE_N_ACT, JUDGE_SAMPLE_N_CON, purpose="judge",
            )
            history = self._prior_rounds_history_block(ds, round_num)
            section = (
                f"=== {ds} ===\n"
                f"label: \"{lab}\"\n\n"
                f"sampled activating rows ({JUDGE_SAMPLE_N_ACT}):\n{samples['act']}\n"
                f"sampled contrast rows ({JUDGE_SAMPLE_N_CON}):\n{samples['con']}"
            )
            if history:
                section += f"\n\n{history}"
            sections.append(section)
        data_block = "\n\n".join(sections)
        remaining = MAX_ROUNDS - round_num
        ds_list = ", ".join(f'"{ds}"' for ds in self.datasets)
        return (
            f"You are the judge for SAE feature f_{self.feat_idx} ({self.preprocessing.get('model', self.model)}) "
            f"labeling round {round_num} of max {MAX_ROUNDS}.\n\n"
            f"{self._preprocessing_block()}\n\n"
            f"Each of the {len(self.datasets)} per-dataset agents produced a label this round and "
            "randomly sampled one activating row + one contrast row from its own contrastive CSV.\n"
            "Row cells are annotated: numeric 'value (pXX)' = percentile in training; "
            "categorical 'value (freq Y.YY)' = training prevalence.\n"
            "Rows do not include downstream prediction columns.\n\n"
            f"{data_block}\n\n"
            "TASK\n"
            f"For each dataset independently, check whether the agent's {'rule set' if self.label_format == 'rules' else 'label'} claims distinguish "
            "that dataset's activating row from its contrast row — in any combination of inputs "
            "(percentile positions, categorical frequencies), or labels (target).\n"
            "A claim survives only if it separates the two rows.\n\n"
            f"Then decide whether the {len(self.datasets)} per-dataset {'rule sets' if self.label_format == 'rules' else 'labels'} share a cross-dataset distinguishing pattern the "
            "sampled rows corroborate. If so, the overall verdict is 'done'; otherwise 'continue' "
            f"(rounds remaining: {remaining}).\n\n"
            f"FINAL CONSENSUS {'RULE SET' if self.label_format == 'rules' else 'LABEL'}\n"
            "If overall_verdict is 'done', OR this is the final round "
            f"({'YES' if remaining == 0 else 'no'}), you must also write a 'final_label' — "
            f"{'a plain-text RULE SET with bullets under a \"RULE SET\" header' if self.label_format == 'rules' else f'a single-sentence consensus label ({LABEL_WORDS_MIN}-{LABEL_WORDS_MAX} words)'} "
            "that describes what distinguishes activating from contrast rows across the datasets, in "
            "shape-only language. You have watched the agents revise for up to 5 rounds, you have "
            "sampled fresh evidence each round, and you have the cross-dataset view; this is "
            "your synthesis, not a vote among the agents' outputs.\n"
            "  * Capture the concept's MEANING overall, not activation strength.\n"
            "  * If a cross-dataset pattern holds on all sampled pairs, state it plainly.\n"
            "  * If the signal is genuinely heterogeneous, abstract to the shared semantic core rather\n"
            "    than smoothing over contradictions with directional phrasing.\n"
            "  * No column names. No domain terms. Use structural language.\n"
            "  * In rules mode, the final rule set must contain ONLY direct row-level structural rules.\n"
            "    Put all commentary about strongest subtype, secondary subtype, portability, unresolved\n"
            "    disagreement, or heterogeneity into overall_note instead.\n"
            "  * If no portable monosemantic rule survives, overall_note should say that clearly, but the\n"
            "    final rule set should still avoid meta-commentary.\n"
            "Otherwise omit the 'final_label' field.\n\n"
            "SHAPE-ONLY DISCIPLINE FOR YOUR OUTPUT\n"
            f"The agents are required to write shape-level {'rule sets' if self.label_format == 'rules' else 'labels'} only — no column names, no\n"
            "domain terms. Your feedback and suggestions must follow the same rule so agents\n"
            "can apply them directly.\n"
            "  * Refer to columns as 'a specific numeric column', 'a column subset', 'adjacent\n"
            "    categorical positions' — never by name.\n"
            "  * Describe values by percentile / frequency / sign / count / density / spread,\n"
            "    not by domain meaning.\n"
            "  * Forbidden in supported_claims / unsupported_claims / contradicted_claims /\n"
            "    missing_signal / suggested_revision: any column name, any domain term, any\n"
            "    hard-coded index like 'position +1'. Use structural phrasing instead.\n"
            "If you find yourself about to reference a specific column or domain, restate the\n"
            "observation as a structural / statistical property.\n\n"
            "RULE-SET PURITY (rules mode)\n"
            "When LABEL_FORMAT=rules, 'final_label' is not an essay and not a verdict. It is only the\n"
            "rule set itself. Forbidden inside final_label in rules mode:\n"
            "  * 'strongest recurring subtype', 'secondary subtype', 'treat as heterogeneous',\n"
            "    'no single rule survives', 'not portable', 'across the full set', or similar\n"
            "    meta-analysis.\n"
            "Those belong in overall_note, not in the rule set.\n\n"
            "EVIDENCE CALIBRATION — tag every claim with a certainty level\n"
            "Each claim in supported_claims / unsupported_claims / contradicted_claims /\n"
            "missing_signal must be an object {\"claim\": str, \"certainty\": str} where\n"
            "certainty is one of:\n"
            "  * 'local observation' — visible only in this round's sampled pair; one-shot evidence.\n"
            "  * 'emerging pattern'  — observed this round and plausibly aligns with prior-round\n"
            "                         observations, but not yet confirmed across >=2 rounds.\n"
            "  * 'stable cross-round pattern' — same direction of claim appears in supported_claims\n"
            "                         across >=2 rounds' fresh samples (consult PRIOR-ROUND\n"
            "                         OBSERVATIONS above).\n"
            "Use sample-specific numeric anchors (e.g. exact percentile values like 'p85', 'p100',\n"
            "exact frequencies like 'freq 0.07') ONLY when repeated rounds' samples keep selecting\n"
            "that value — otherwise use range language ('extreme upper-tail values',\n"
            "'rare-prevalence levels', 'low-frequency categorical levels'). A single sample\n"
            "contradicting a number does NOT establish a new number as the true mechanism; it only\n"
            "invalidates the old one.\n\n"
            "OUTPUT FORMAT (strict)\n"
            "Output a single fenced JSON block (```json ... ```). Nothing before or after the "
            "fence. Schema:\n\n"
            "```json\n"
            "{\n"
            '  "overall_verdict": "done" | "continue",\n'
            '  "overall_note": "<1-3 sentences on cross-dataset pattern or what\'s missing>",\n'
            '  "final_label": "<required if verdict=done OR final round; else omit>",\n'
            '  "per_dataset": {\n'
            f"    // one entry for each of: {ds_list}\n"
            '    "<dataset>": {\n'
            '      "verdict": "accept" | "revise" | "split",\n'
            '      "supported_claims":   [{"claim": str, "certainty": "local observation"|"emerging pattern"|"stable cross-round pattern"}, ...],\n'
            '      "unsupported_claims": [{"claim": str, "certainty": ...}, ...],\n'
            '      "contradicted_claims":[{"claim": str, "certainty": ...}, ...],\n'
            '      "missing_signal":     {"claim": str, "certainty": ...} | null,\n'
            '      "portability_risk":   {"level": "low"|"medium"|"high", "justification": "<short>"},\n'
            f'      "suggested_revision": "<{"one tighter rule set or a short edit instruction" if self.label_format == "rules" else "one tighter sentence or a short edit instruction"} — shape-only>"\n'
            '    }\n'
            '  }'
            + (
                ',\n'
                '  "next_round_pairings": {\n'
                '    // OPTIONAL. Overrides the default ring-rotation peer assignment for\n'
                '    // the next round. Use this when per-dataset claims contradict on the\n'
                '    // same structural axis (e.g. an "activating = high" claim in one\n'
                '    // dataset vs "activating = low" in another). Pair the minority-\n'
                '    // direction agent with a majority-direction peer so both can reconcile\n'
                '    // in front of evidence. Omit for datasets that should keep the default\n'
                '    // rotation. If overall_verdict is "done", omit this field entirely.\n'
                '    "<dataset>": "<peer_dataset>"\n'
                '  }\n'
                if self.pairings_mode == "on" else "\n"
            ) +
            '}\n'
            "```\n"
            "Verdicts: 'accept' = claims hold across sampled pairs (prefer 'stable cross-round' or "
            "'emerging' certainty backing the verdict); 'revise' = tighten per suggested_revision; "
            "'split' = the label is conflating two distinct regimes that should be stated separately.\n"
            "An 'accept' verdict should NOT rest on 'local observation' claims alone — that is a\n"
            "one-sample agreement, not a durable pattern."
            + (
                "\n\nPAIRING GUIDANCE: if per-dataset supported_claims (or the round-1 labels above) "
                "contradict each other on direction / polarity (e.g. 'activating = upper-tail' in "
                "one dataset vs 'activating = low-tail' in another), prefer using "
                "`next_round_pairings` to pair those datasets together. One round of face-to-face "
                "evidence usually resolves whether they are the same concept surface-inverted "
                "(merge to a polarity-agnostic description) or genuinely split (two sub-concepts)."
                if self.pairings_mode == "on" else ""
            )
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Pull the first fenced JSON block out of agent text output.

        Raises ValueError with a truncated excerpt on parse failure so the
        caller can see what the judge actually produced.
        """
        import re as _re
        m = _re.search(r"```json\s*(.+?)```", text, flags=_re.DOTALL)
        if not m:
            raise ValueError(
                "No fenced ```json ... ``` block found in judge output. "
                f"First 400 chars: {text[:400]!r}"
            )
        raw = m.group(1).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from judge output: {e}. "
                f"First 400 chars of block: {raw[:400]!r}"
            ) from e

    def record_judge_response(self, response: dict) -> None:
        """Record the judge's structured response for the latest round.

        Validates required fields and per-dataset keys before appending.
        When overall_verdict='done' or the round just judged is the final
        round, `final_label` is required.
        """
        if len(self.judge_verdicts) >= len(self.rounds):
            raise ValueError(
                f"Already recorded {len(self.judge_verdicts)} judge verdicts for "
                f"{len(self.rounds)} rounds; record a new round first."
            )
        verdict = response.get("overall_verdict")
        if verdict not in ("done", "continue"):
            raise ValueError(
                f"overall_verdict must be 'done' or 'continue', got {verdict!r}"
            )
        per_ds = response.get("per_dataset", {})
        missing = [ds for ds in self.datasets if ds not in per_ds]
        if missing:
            raise ValueError(
                f"Judge response missing per_dataset entries for: {missing}"
            )
        round_just_judged = len(self.judge_verdicts) + 1
        final_label_required = (verdict == "done") or (round_just_judged >= MAX_ROUNDS)
        final_label = (response.get("final_label") or "").strip() or None
        if final_label_required and not final_label:
            raise ValueError(
                f"final_label required when verdict='done' or final round "
                f"(round {round_just_judged} of {MAX_ROUNDS}) but was missing/empty."
            )
        pairings: dict = {}
        if self.pairings_mode == "on":
            pairings_in = response.get("next_round_pairings") or {}
            if isinstance(pairings_in, dict):
                for ds, peer in pairings_in.items():
                    if ds == peer:
                        continue  # agent can't be its own peer
                    if ds in self.datasets and peer in self.datasets:
                        pairings[ds] = peer
        self.judge_verdicts.append({
            "verdict": verdict,
            "note": (response.get("overall_note") or "").strip(),
            "final_label": final_label,
            "per_dataset": {ds: per_ds[ds] for ds in self.datasets},
            "next_round_pairings": pairings,
        })
        self._save_state()

    def is_done(self) -> bool:
        """True if the latest judge verdict is 'done' or we've hit MAX_ROUNDS."""
        if self.judge_verdicts and self.judge_verdicts[-1]["verdict"] == "done":
            return True
        return len(self.rounds) >= MAX_ROUNDS

    # ── Held-out validator ─────────────────────────────────────────────
    def _validator_csv_paths(self) -> dict:
        return {
            ds: self._model_dir / f"f{self.feat_idx}_validator_{ds}.csv"
            for ds in self.datasets
        }

    def _validator_truth(self) -> dict:
        p = self._model_dir / f"f{self.feat_idx}_validator_truth.json"
        if not p.exists():
            return {}
        return json.loads(p.read_text())

    def validator_prompt(self) -> str:
        """Prompt a judge-independent validator with held-out rows + final label.

        The validator sees the final label and, per dataset, an opaque
        shuffled list of held-out rows (the contrastive / judge rows are
        excluded). It outputs a binary fires/doesn't-fire prediction per
        row, which we grade against a stored truth file.
        """
        lab = self.final_label()
        if not lab:
            raise RuntimeError("No final label to validate")
        paths = self._validator_csv_paths()
        present = {
            ds: self._prompt_csv_text(pd.read_csv(p))
            for ds, p in paths.items() if p.exists()
        }
        if not present:
            raise RuntimeError(
                f"No validator CSVs under {self._model_dir}. Run: python -m "
                f"scripts.concepts.build_contrastive_examples --validator "
                f"--model {self.model} --features {self.feat_idx}"
            )
        data_block = "\n\n".join(
            f"=== {ds} ===\n{csv}" for ds, csv in present.items()
        )
        return (
            f"You are the validator for SAE feature f_{self.feat_idx} "
            f"({self.preprocessing.get('model', self.model)}).\n"
            "The labeling agents and the judge did NOT see these rows. Your job is to "
            "predict, per row, whether it activates the concept described by the FINAL "
            f"{'RULE SET' if self.label_format == 'rules' else 'LABEL'} below. Binary output only.\n\n"
            f"FINAL {'RULE SET' if self.label_format == 'rules' else 'LABEL'}\n{lab if self.label_format == 'rules' else f'\"{lab}\"'}\n\n"
            f"{self._preprocessing_block()}\n\n"
            "Row cells: numeric 'value (pXX)' = training percentile; categorical 'value "
            "(freq Y.YY)' = training prevalence. Rows are shuffled; you cannot infer class "
            "from ordering.\n\n"
            f"{data_block}\n\n"
            "TASK\n"
            "For each row_id in each dataset, classify the row into one of two categories "
            f"based solely on whether it fits the FINAL {'RULE SET' if self.label_format == 'rules' else 'LABEL'}'s structural claims:\n"
            f"  * \"fires\"         — the row matches the {'rule set' if self.label_format == 'rules' else 'label'}'s description of activating rows.\n"
            "  * \"does_not_fire\" — the row does not match that description.\n"
            "Judge each row independently. Do not assume any particular number of rows should "
            "fire in a dataset — the true count could be 0, all, or anything in between, and "
            "counting toward a balanced expectation biases the result. Base each decision only "
            "on whether that specific row fits the label.\n"
            "No reasoning, no hedging, no extra fields — row_id and prediction only.\n\n"
            "OUTPUT FORMAT (strict)\n"
            "Single fenced ```json``` block. Nothing before or after.\n\n"
            "```json\n"
            "{\n"
            '  "per_dataset": {\n'
            '    "<dataset>": [\n'
            '      {"row_id": "r000", "prediction": "fires"},\n'
            '      {"row_id": "r001", "prediction": "does_not_fire"}\n'
            '    ]\n'
            '  }\n'
            '}\n'
            "```"
        )

    def record_validator_response(self, response: dict) -> dict:
        """Grade validator predictions against the stored truth file."""
        truth = self._validator_truth()
        if not truth:
            raise RuntimeError(
                "No validator truth file; regenerate validator CSVs first."
            )
        per_ds_resp = response.get("per_dataset") or {}
        missing_ds = [ds for ds in truth if ds not in per_ds_resp]
        if missing_ds:
            raise ValueError(f"Validator response missing datasets: {missing_ds}")

        def _decode(p_entry):
            """Accept 'fires'/'does_not_fire' (current) or bool (legacy)."""
            val = p_entry.get("prediction", p_entry.get("predicted_fires"))
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                s = val.strip().lower()
                if s in ("fires", "fire", "activates", "activating"):
                    return True
                if s in ("does_not_fire", "does not fire", "no", "inactive", "not_firing"):
                    return False
            raise ValueError(f"Cannot decode prediction value: {val!r}")

        per_ds_stats: dict = {}
        tot_tp = tot_fp = tot_tn = tot_fn = 0
        for ds, ds_truth in truth.items():
            preds_list = per_ds_resp.get(ds) or []
            preds = {p.get("row_id"): _decode(p) for p in preds_list}
            missing_rows = set(ds_truth.keys()) - set(preds.keys())
            if missing_rows:
                raise ValueError(
                    f"Validator response for {ds} missing row_ids: {sorted(missing_rows)}"
                )
            tp = fp = tn = fn = 0
            for row_id, actual in ds_truth.items():
                pred = preds[row_id]
                if actual and pred: tp += 1
                elif actual and not pred: fn += 1
                elif (not actual) and pred: fp += 1
                else: tn += 1
            total = tp + fp + tn + fn
            per_ds_stats[ds] = {
                "total": total,
                "correct": tp + tn,
                "accuracy": (tp + tn) / total if total else 0.0,
                "precision": tp / (tp + fp) if (tp + fp) else None,
                "recall": tp / (tp + fn) if (tp + fn) else None,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }
            tot_tp += tp; tot_fp += fp; tot_tn += tn; tot_fn += fn

        tot_total = tot_tp + tot_fp + tot_tn + tot_fn
        micro_acc = (tot_tp + tot_tn) / tot_total if tot_total else 0.0
        micro_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else None
        micro_rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else None
        f1 = None
        if micro_prec and micro_rec and (micro_prec + micro_rec):
            f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        # Macro-average: mean of per-dataset accuracies (prevents one big dataset dominating)
        macro_acc = (
            sum(s["accuracy"] for s in per_ds_stats.values()) / len(per_ds_stats)
            if per_ds_stats else 0.0
        )
        overall = {
            "total": tot_total,
            "correct": tot_tp + tot_tn,
            "accuracy": micro_acc,            # micro (row-weighted)
            "accuracy_macro": macro_acc,      # macro (dataset-weighted)
            "precision": micro_prec,
            "recall": micro_rec,
            "f1": f1,
            "tp": tot_tp, "fp": tot_fp, "tn": tot_tn, "fn": tot_fn,
        }
        self.validator_results = {
            "overall": overall,
            "per_dataset": per_ds_stats,
            "note": (response.get("overall_note") or "").strip(),
            "label_graded": self.final_label(),
        }
        self._save_state()
        return self.validator_results

    def converged(self) -> bool:
        return bool(self.similarities) and self.similarities[-1] >= CONVERGENCE_THRESHOLD

    def plateaued(self) -> bool:
        """Cosine failed to improve for 2 consecutive rounds."""
        s = self.similarities
        if len(s) < 3:
            return False
        return s[-1] <= s[-2] <= s[-3]

    def next_round_number(self) -> int:
        return len(self.rounds) + 1

    def peak_round(self) -> int:
        """1-indexed round number with the highest mean pairwise cosine."""
        if not self.similarities:
            raise RuntimeError("No rounds recorded yet")
        return int(np.argmax(self.similarities)) + 1

    def most_central_from_peak(self) -> str:
        """Return the most-central label from the highest-cosine round."""
        peak = self.peak_round()
        labels = self.rounds[peak - 1]
        if len(labels) == 1:
            return labels[0]
        embs = nomic_embed(labels)
        sims = embs @ embs.T
        np.fill_diagonal(sims, 0.0)
        centrality = sims.mean(axis=1)
        return labels[int(np.argmax(centrality))]

    def _round_for_final(self) -> int:
        """Round index (1-based) whose labels feed the final pick.

        Prefer the round the judge said 'done' on; fall back to the last
        recorded round.
        """
        for i, v in enumerate(self.judge_verdicts):
            if v.get("verdict") == "done":
                return i + 1
        if not self.rounds:
            raise RuntimeError("No rounds recorded yet")
        return len(self.rounds)

    def most_central_from_current(self) -> str:
        """Most-central label from the judge-approved (or last) round."""
        round_num = self._round_for_final()
        labels = self.rounds[round_num - 1]
        if len(labels) == 1:
            return labels[0]
        embs = nomic_embed(labels)
        sims = embs @ embs.T
        np.fill_diagonal(sims, 0.0)
        centrality = sims.mean(axis=1)
        return labels[int(np.argmax(centrality))]

    def _judge_final_label(self) -> Optional[str]:
        """Return the judge-authored final_label from the latest verdict, if any."""
        if not self.judge_verdicts:
            return None
        lab = self.judge_verdicts[-1].get("final_label")
        return lab if lab else None

    def final_label(self) -> str:
        if self.synthesis:
            return self.synthesis
        judge_lab = self._judge_final_label()
        if judge_lab:
            return judge_lab
        if not self.rounds:
            raise RuntimeError("No rounds recorded yet")
        return self.most_central_from_current()

    # ── Synthesizer ─────────────────────────────────────────────────────
    def synthesizer_prompt(self) -> str:
        """Prompt for a single agent that sees all n labels + all n CSVs."""
        if not self.rounds:
            raise RuntimeError("No rounds recorded — cannot synthesize")
        round_num = self._round_for_final()
        round_labels = self.rounds[round_num - 1]
        label_block = "\n".join(
            f"  [{ds}] \"{lab}\""
            for ds, lab in zip(self.datasets, round_labels)
        )
        data_block = "\n\n".join(
            f"=== {ds} ===\n{self._dataset_block(ds, full_csv=True)}"
            for ds in self.datasets
        )
        return (
            f"You are the synthesizer for SAE feature f_{self.feat_idx} "
            f"({self.preprocessing.get('model', self.model)} model).\n"
            f"{len(self.datasets)} per-dataset agents ran a ring-consensus mesh and produced "
            f"the following labels (from round {round_num}, the judge-approved round):\n\n"
            f"{label_block}\n\n"
            f"Each agent saw only its own dataset's rows + one peer's at a time, so they "
            f"biased toward surface density of their own data. You will see ALL {len(self.datasets)} "
            f"datasets' activating vs contrast rows below, plus the preprocessing context, "
            f"so you can abstract past surface specifics.\n\n"
            f"{self._preprocessing_block()}\n\n"
            f"{data_block}\n\n"
            "TASK\n"
            f"Write a {'necessary-and-sufficient RULE SET' if self.label_format == 'rules' else 'single consensus label'} for f_{{idx}} that describes what distinguishes\n"
            "activating from contrast rows across ALL datasets — in STRUCTURAL / DISTRIBUTIONAL\n"
            "terms only.\n"
            "  * IGNORE column names and ordering. Treat all columns as nameless c0..cN.\n"
            "    The concept must be invariant under column permutation and renaming.\n"
            "  * Describe shape: density of non-zero values, within-row spread/variance,\n"
            "    correlation/co-variation between column subsets, magnitude regime, clustering,\n"
            "    repetition/uniformity, extreme-vs-moderate value placement.\n"
            "  * No column names. No domain terms (no 'quasar', 'carat', 'bankruptcy', etc.).\n"
            "  * The previous round-1 agents may have leaked domain language into their labels.\n"
            "    Strip that away and work from the raw CSVs below, not from their framings.\n\n"
            f"{self._rules_block() if self.label_format == 'rules' else ''}"
            "CONTRAST DISCIPLINE — include only distinguishing properties:\n"
            "  * Before adding ANY property to the consensus label, verify activating rows DIFFER\n"
            "    from contrast rows on it — in any combination of inputs (percentile positions,\n"
            "    categorical frequencies, within-row spread, column-subset co-firing), or labels\n"
            "    (target values, base-rate enrichment).\n"
            "  * Check this across ALL datasets: a property may distinguish act from con in one\n"
            "    dataset but not in another — in that case it is not a cross-dataset signal.\n"
            "  * Properties shared between activating and contrast are NOT feature-level signals\n"
            "    and must not appear in the label.\n"
            "  * Keep looking — properties can combine across dimensions (an input × target\n"
            "    joint regime; a percentile × target pair; a column-subset × rarity pattern).\n"
            "    Exhaust those combinations before settling on a label.\n\n"
            f"{'In rules mode, output only direct row-level structural rules. Put any commentary about strongest subtype, secondary subtype, portability, unresolved disagreement, or heterogeneity outside the rule set; do not include it in the final output.\\n\\n' if self.label_format == 'rules' else ''}"
            f"{self._output_block() if self.label_format == 'rules' else f'Output ONLY a single-sentence label ({LABEL_WORDS_MIN}-{LABEL_WORDS_MAX} words). Shape-level language only. No column names. No preamble.'}"
        ).replace("{idx}", str(self.feat_idx))

    def record_synthesis(self, label: str) -> None:
        self.synthesis = _strip_label(label)
        self._save_state()

    def save_label(
        self, *, output_path: Optional[Path] = None, label: Optional[str] = None
    ) -> Path:
        label = label or self.final_label()
        if output_path is None:
            output_path = self._model_dir / f"f{self.feat_idx}_label.json"
        judge_lab = self._judge_final_label()
        output_path.write_text(json.dumps({
            "model": self.model,
            "feat_idx": self.feat_idx,
            "arch": self.arch,
            "prompt_order": self.prompt_order,
            "label_format": self.label_format,
            "pairings_mode": self.pairings_mode,
            "label": label,
            "label_source": (
                "synthesis" if self.synthesis and label == self.synthesis
                else "judge" if judge_lab and label == judge_lab
                else "most_central"
            ),
            "datasets_used": self.datasets,
            "rounds_completed": len(self.rounds),
            "final_round": self._round_for_final() if self.rounds else None,
            "judge_verdicts": self.judge_verdicts,
            "similarity_history": self.similarities,
            "labels_per_round": self.rounds,
            "validator_results": self.validator_results,
        }, indent=2))
        return output_path


def _strip_label(s: str) -> str:
    s = s.strip()
    # Remove common wrappers from agent outputs
    for quote in ('"', "'", "`"):
        if s.startswith(quote) and s.endswith(quote):
            s = s[1:-1].strip()
    for prefix in ("Label:", "LABEL:", "label:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    for prefix in ("RULE SET", "Rule set", "Rules", "RULES"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    return s


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--feat", type=int, required=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser(
        "init",
        help=(
            "Initialize state for a (model, feature) with the chosen knobs and "
            "print round-1 prompts. Knobs are persisted in the state file; later "
            "subcommands read them from state."
        ),
    )
    init_p.add_argument(
        "--arch", default=DEFAULT_ARCH, choices=sorted(VALID_ARCHES),
        help=f"Ring architecture (default: {DEFAULT_ARCH}).",
    )
    init_p.add_argument(
        "--prompt-order", default=DEFAULT_PROMPT_ORDER, choices=sorted(VALID_PROMPT_ORDERS),
        help=f"Section ordering for round/ring prompts (default: {DEFAULT_PROMPT_ORDER}).",
    )
    init_p.add_argument(
        "--label-format", default=DEFAULT_LABEL_FORMAT, choices=sorted(VALID_LABEL_FORMATS),
        help=f"Final label format (default: {DEFAULT_LABEL_FORMAT}).",
    )
    init_p.add_argument(
        "--pairings", default=DEFAULT_PAIRINGS, choices=sorted(VALID_PAIRINGS),
        help=(
            f"Whether the judge may propose next_round_pairings to override default "
            f"ring rotation (default: {DEFAULT_PAIRINGS})."
        ),
    )

    show = sub.add_parser("show-round", help="Show prompts for a specific round")
    show.add_argument("--round", type=int, required=True)

    record = sub.add_parser("record", help="Record labels for the next round")
    record.add_argument("--labels-file", type=Path, required=True,
                         help="JSON list of label strings; baseline/ringlite use datasets_used order, ringlite_freeze uses the active-dataset order printed by show-round")

    sub.add_parser("status", help="Print round-by-round state")
    sub.add_parser("save-label", help="Write final label to f{feat}_label.json")
    sub.add_parser("reset", help="Clear persisted rounds for this feature")
    sub.add_parser("show-synth", help="Print the synthesizer prompt")
    synth_rec = sub.add_parser("record-synth", help="Record the synthesizer's label")
    synth_rec.add_argument("--label", type=str, required=True)

    show_judge = sub.add_parser("show-judge", help="Print the judge prompt for a round")
    show_judge.add_argument("--round", type=int, default=None,
                             help="Round number (default: latest recorded round)")
    rec_judge = sub.add_parser("record-judge", help="Record the judge's structured response for the latest round")
    rec_judge.add_argument("--response-file", type=Path, required=True,
                            help="Path to judge's raw output (with fenced ```json block)")

    sub.add_parser("show-validator", help="Print the validator prompt (held-out rows vs final label)")
    rec_val = sub.add_parser("record-validator", help="Grade the validator's response against the truth file")
    rec_val.add_argument("--response-file", type=Path, required=True)

    args = parser.parse_args()

    if args.cmd == "init":
        # `init` defines the run's knobs; wipe any prior state first so the
        # knobs are authoritative (resume-and-change is intentionally not
        # supported — force an explicit reset instead).
        stale = ContrastiveMeshPipeline(args.model, args.feat)
        stale.reset()
        pipe = ContrastiveMeshPipeline(
            model=args.model,
            feat_idx=args.feat,
            arch=args.arch,
            prompt_order=args.prompt_order,
            label_format=args.label_format,
            pairings_mode=args.pairings,
        )
        pipe._save_state()
        print(f"arch={pipe.arch}  prompt_order={pipe.prompt_order}  label_format={pipe.label_format}  pairings={pipe.pairings_mode}")
        print(f"Datasets (n={len(pipe.datasets)}): {pipe.datasets}")
        prompts = pipe.round_prompts(1)
        for ds, p in zip(pipe.datasets, prompts):
            print(f"\n===== {ds} =====\n{p}")
        return

    # All non-init subcommands load knobs from the state file written by init.
    pipe = ContrastiveMeshPipeline(args.model, args.feat)

    if args.cmd == "show-round":
        active = pipe.active_datasets_for_round(args.round)
        print(f"arch={pipe.arch}  prompt_order={pipe.prompt_order}  label_format={pipe.label_format}  pairings={pipe.pairings_mode}")
        print(f"Active datasets for round {args.round} (n={len(active)}): {active}")
        prompts = pipe.round_prompts(args.round)
        for ds, p in zip(active, prompts):
            print(f"\n===== round {args.round} / {ds} =====\n{p}")
    elif args.cmd == "record":
        labels = json.loads(args.labels_file.read_text())
        sim = pipe.record_round(labels)
        print(f"Round {len(pipe.rounds)} recorded (pairwise cosine diagnostic: {sim:.4f})")
        print(f"Next step: show-judge and record the verdict.")
    elif args.cmd == "status":
        for i, round_labels in enumerate(pipe.rounds, 1):
            sim = pipe.similarities[i - 1] if i - 1 < len(pipe.similarities) else None
            verdict = pipe.judge_verdicts[i - 1] if i - 1 < len(pipe.judge_verdicts) else None
            sim_str = f"cosine={sim:.4f}" if sim is not None else "cosine=-"
            v_str = f"judge={verdict['verdict']}" if verdict else "judge=pending"
            print(f"Round {i}: {sim_str}  {v_str}")
            for ds, lab in zip(pipe.datasets, round_labels):
                print(f"  [{ds}] {lab}")
            if verdict and verdict.get("note"):
                print(f"  judge note: {verdict['note']}")
        if pipe.synthesis:
            print(f"\nSynthesized: {pipe.synthesis}")
        print(f"arch={pipe.arch}  prompt_order={pipe.prompt_order}  label_format={pipe.label_format}  pairings={pipe.pairings_mode}")
        print(f"Next active datasets: {pipe.active_datasets_for_round(pipe.next_round_number())}")
        print(f"Done: {pipe.is_done()}   Rounds: {len(pipe.rounds)}/{MAX_ROUNDS}")
    elif args.cmd == "save-label":
        path = pipe.save_label()
        print(f"Saved final label to {path}")
        print(f"Label: {pipe.final_label()}")
    elif args.cmd == "reset":
        pipe.reset()
        print("State cleared")
    elif args.cmd == "show-synth":
        print(pipe.synthesizer_prompt())
    elif args.cmd == "record-synth":
        pipe.record_synthesis(args.label)
        print(f"Recorded synthesis: {pipe.synthesis}")
    elif args.cmd == "show-judge":
        print(pipe.judge_prompt(args.round))
    elif args.cmd == "record-judge":
        raw = args.response_file.read_text()
        parsed = ContrastiveMeshPipeline._extract_json(raw)
        pipe.record_judge_response(parsed)
        v = pipe.judge_verdicts[-1]
        print(f"Recorded round-{len(pipe.judge_verdicts)} verdict: {v['verdict']}")
        for ds, pd_block in v["per_dataset"].items():
            print(f"  [{ds}] {pd_block.get('verdict', '?')}")
        print(f"Done: {pipe.is_done()}   Rounds: {len(pipe.rounds)}/{MAX_ROUNDS}")
    elif args.cmd == "show-validator":
        print(pipe.validator_prompt())
    elif args.cmd == "record-validator":
        raw = args.response_file.read_text()
        parsed = ContrastiveMeshPipeline._extract_json(raw)
        result = pipe.record_validator_response(parsed)
        o = result["overall"]
        # Headline metric for HP sweeps: micro accuracy + macro + f1
        prec_str = f"{o['precision']:.3f}" if o['precision'] is not None else "--"
        rec_str  = f"{o['recall']:.3f}"    if o['recall']    is not None else "--"
        f1_str   = f"{o['f1']:.3f}"        if o['f1']        is not None else "--"
        print(f"HEADLINE  accuracy(micro)={o['accuracy']:.3f}  "
              f"accuracy(macro)={o['accuracy_macro']:.3f}  f1={f1_str}")
        print(f"          {o['correct']}/{o['total']} correct  "
              f"precision={prec_str}  recall={rec_str}")
        print()
        for ds, s in result["per_dataset"].items():
            prec = f"{s['precision']:.2f}" if s['precision'] is not None else "--"
            rec = f"{s['recall']:.2f}" if s['recall'] is not None else "--"
            print(f"  [{ds}] acc={s['accuracy']:.2f} ({s['correct']}/{s['total']}) "
                  f"prec={prec} rec={rec}")


if __name__ == "__main__":
    _cli()
