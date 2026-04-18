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
NOMIC_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
JUDGE_SAMPLE_SEED = 13
PEER_SAMPLE_N_ACT = 3
PEER_SAMPLE_N_CON = 3
JUDGE_SAMPLE_N_ACT = 2
JUDGE_SAMPLE_N_CON = 2

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

    def __post_init__(self):
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
        self._load_state_if_exists()

    # ── State persistence ──────────────────────────────────────────────
    def _load_state_if_exists(self):
        if not self._state_path.exists():
            return
        data = json.loads(self._state_path.read_text())
        if data.get("feat_idx") != self.feat_idx or data.get("model") != self.model:
            return
        self.rounds = data.get("rounds", [])
        self.similarities = data.get("similarities", [])
        self.judge_verdicts = data.get("judge_verdicts", [])
        self.synthesis = data.get("synthesis")

    def _save_state(self):
        self._state_path.write_text(json.dumps({
            "model": self.model,
            "feat_idx": self.feat_idx,
            "datasets": self.datasets,
            "rounds": self.rounds,
            "similarities": self.similarities,
            "judge_verdicts": self.judge_verdicts,
            "synthesis": self.synthesis,
        }, indent=2))

    def reset(self):
        self.rounds = []
        self.similarities = []
        self.judge_verdicts = []
        self.synthesis = None
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
            csv_text = self.csv_paths[ds].read_text()
            note = (
                "Each numeric cell is annotated as 'value (pXX)' = pXX percentile in the SAE "
                "training split. Each categorical/binary cell is 'value (freq Y.YY)' = prevalence "
                "of that value in training.\n"
                "Each row also carries the model's prediction on that row:\n"
                "  - classification: pred_class, pred_conf (max softmax prob), pred_correct (bool)\n"
                "  - regression:     pred_value, pred_abs_err (|pred - target|)\n"
                "Use these to judge whether the feature fires on confident-correct, "
                "confident-wrong, or uncertain rows.\n"
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

    def round1_prompt(self, ds: str) -> str:
        return (
            f"You are labeling SAE feature f_{self.feat_idx} for a tabular foundation model.\n"
            f"This is round 1 of ring-consensus mesh labeling; you are the agent for '{ds}'.\n\n"
            f"{self._preprocessing_block()}\n\n"
            f"{self._dataset_block(ds, full_csv=True)}\n\n"
            "TASK\n"
            "The CSV has a 'band' column: 'top' (highest activations), 'p90' (~90th percentile\n"
            "of activating rows), 'p80' (~80th percentile), or 'contrast' (feature does NOT fire).\n"
            "Each cell is annotated with its marginal position in the SAE training split:\n"
            "  - numeric: 'value (pXX)' means XX-th percentile\n"
            "  - categorical/binary: 'value (freq Y.YY)' means that category/value occurs at rate Y.YY\n"
            "The dataset header shows the training-split target base rates. Compare the target\n"
            "values seen in the activating rows to the base rate to judge target enrichment.\n\n"
            "CRITICAL INSTRUCTION — describe SHAPE and STATISTICAL FINGERPRINT only:\n"
            "  * IGNORE column names and the domain meaning they suggest.\n"
            "  * IGNORE column ordering. The pattern should be invariant under permutation of columns.\n"
            "  * Use the marginal annotations (pXX / freq) as your primary source — these are\n"
            "    dataset-agnostic: 'p85' means the same thing whether the column is a magnitude,\n"
            "    a permission bit, or an age.\n"
            "  * Describe distributional properties: do activating rows concentrate at high/low/mid\n"
            "    percentiles? At rare/common categorical levels? Across many columns or a narrow\n"
            "    subset? With tight or dispersed within-row spread of percentiles?\n"
            "  * Do NOT mention specific column names or domain terms.\n\n"
            "CONTRAST DISCIPLINE — include only distinguishing properties:\n"
            "  * Before adding ANY property to the label, verify activating rows DIFFER from\n"
            "    contrast rows on it — in any combination of inputs (percentile positions,\n"
            "    categorical frequencies, within-row spread, column-subset co-firing), labels\n"
            "    (target values, base-rate enrichment), or predictions (pred_class, pred_conf,\n"
            "    pred_correct / pred_value, pred_abs_err).\n"
            "  * Properties shared between activating and contrast are NOT feature-level signals\n"
            "    and must not appear in the label.\n"
            "  * Keep looking — properties can combine (an interaction between two inputs; a\n"
            "    pred_class × target pair; a percentile × prediction-confidence joint regime).\n"
            "    Exhaust the combinations before concluding anything is shared.\n\n"
            "Note whether the pattern holds at p80/p90 too, or only at top activations.\n\n"
            f"Output ONLY a single-sentence label ({LABEL_WORDS_MIN}-{LABEL_WORDS_MAX} words) "
            "describing the structural / distributional fingerprint. No column names. No domain terms. "
            "No preamble, no explanation."
        )

    def _peer_evidence_block(self, peer_ds: str, round_num: int) -> str:
        """Render peer evidence as header + 3 activating + 3 contrast rows.

        Replaces passing the full CSV, which (a) bloats context 5× and
        (b) lets peers pattern-match on incidental rows rather than generalize.
        """
        meta = self.feat_context["dataset_stats"].get(peer_ds, {})
        target = meta.get("target_summary") or {}
        header = (
            f"Dataset: {peer_ds} ({meta.get('task_type', '?')}), "
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
        samples = self._sample_rows_for(
            round_num, peer_ds, PEER_SAMPLE_N_ACT, PEER_SAMPLE_N_CON, purpose="peer",
        )
        note = (
            "Numeric cells: 'value (pXX)' = pXX percentile in training.\n"
            "Categorical cells: 'value (freq Y.YY)' = training prevalence.\n"
            "Prediction columns: pred_class/pred_conf/pred_correct (classification) or\n"
            "pred_value/pred_abs_err (regression).\n"
        )
        return (
            f"{header}\n\n{note}\n"
            f"peer's {PEER_SAMPLE_N_ACT} sampled activating rows:\n{samples['act']}\n"
            f"peer's {PEER_SAMPLE_N_CON} sampled contrast rows:\n{samples['con']}"
        )

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
            return "\n".join(f"  - {x}" for x in items)
        return (
            f"JUDGE FEEDBACK ON '{ds}' (from round {round_num - 1})\n"
            f"overall verdict: {v.get('verdict')}\n"
            f"overall note: {v.get('note', '')}\n"
            f"per-dataset verdict: {per_ds.get('verdict', '?')}\n"
            f"supported claims:\n{_fmt_list(per_ds.get('supported_claims'))}\n"
            f"unsupported claims:\n{_fmt_list(per_ds.get('unsupported_claims'))}\n"
            f"contradicted claims:\n{_fmt_list(per_ds.get('contradicted_claims'))}\n"
            f"missing signal: {per_ds.get('missing_signal', '') or '(none)'}\n"
            f"portability risk: {pr_str}\n"
            f"suggested revision: {per_ds.get('suggested_revision', '') or '(none)'}\n"
        )

    def ring_prompt(self, ds: str, peer_ds: str, peer_label: str, round_num: int) -> str:
        own_feedback = self._judge_feedback_block(ds, round_num)
        peer_feedback = self._judge_feedback_block(peer_ds, round_num)
        feedback_sections = ""
        if own_feedback or peer_feedback:
            feedback_sections = (
                (f"\n{own_feedback}" if own_feedback else "") +
                (f"\n{peer_feedback}" if peer_feedback else "") +
                "\n"
            )
        return (
            f"You are the agent for '{ds}', labeling SAE feature f_{self.feat_idx}.\n"
            f"This is round {round_num} of ring-consensus mesh labeling.\n\n"
            f"{self._preprocessing_block()}\n\n"
            f"YOUR DATA ROWS\n{self._dataset_block(ds, full_csv=True)}\n\n"
            f"PEER EVIDENCE (from '{peer_ds}')\n"
            f"  peer's current label: \"{peer_label}\"\n\n"
            f"{self._peer_evidence_block(peer_ds, round_num)}\n"
            f"{feedback_sections}\n"
            "TASK\n"
            "Revise your label in light of the peer's hypothesis and the judge's feedback above.\n"
            "Address unsupported / contradicted claims; incorporate the missing signal if the "
            "evidence supports it; apply the suggested revision unless your data contradicts it.\n"
            "CRITICAL: describe SHAPE and STATISTICAL FINGERPRINT only.\n"
            "  * IGNORE column names and ordering. The pattern must be invariant under column\n"
            "    permutation and renaming.\n"
            "  * Describe distributional / structural properties (density, spread, co-variation,\n"
            "    magnitude regime, clustering).\n"
            "  * No column names. No domain terms.\n\n"
            "If the peer's shape-level pattern also fits your activating rows, merge framings.\n"
            "If your data contradicts theirs at the shape level, hold your ground but sharpen.\n\n"
            "CONTRAST DISCIPLINE — include only distinguishing properties:\n"
            "  * Before adding ANY property to the label, verify activating rows DIFFER from\n"
            "    contrast rows on it — in any combination of inputs, labels, or predictions.\n"
            "  * Properties shared between activating and contrast are NOT feature-level signals.\n"
            "  * Keep looking — properties can combine (interactions, joint regimes). Exhaust the\n"
            "    combinations; don't settle for a weak claim.\n\n"
            "Goal: a label expressible in structural terms that fits both datasets.\n\n"
            f"Output ONLY a single-sentence label ({LABEL_WORDS_MIN}-{LABEL_WORDS_MAX} words). "
            "No prefix, no explanation."
        )

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
        return [
            self.ring_prompt(
                ds=ds,
                peer_ds=self.datasets[(i - peer_offset) % n],
                peer_label=prev_labels[(i - peer_offset) % n],
                round_num=round_num,
            )
            for i, ds in enumerate(self.datasets)
        ]

    # ── State updates ──────────────────────────────────────────────────
    def record_round(
        self, labels: list[str], *, round_num: Optional[int] = None
    ) -> float:
        if round_num is None:
            round_num = len(self.rounds) + 1
        if len(labels) != len(self.datasets):
            raise ValueError(
                f"Expected {len(self.datasets)} labels, got {len(labels)}"
            )
        clean = [_strip_label(l) for l in labels]
        if round_num != len(self.rounds) + 1:
            raise ValueError(
                f"Next round is {len(self.rounds) + 1}, got round_num={round_num}"
            )
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
            return pd.DataFrame(rows).to_csv(index=False)

        return {"act": _to_csv(act_rows), "con": _to_csv(con_rows)}

    def judge_prompt(self, round_num: Optional[int] = None) -> str:
        """Build a judge prompt over the current (or specified) round's labels.

        The judge sees each agent's label plus the 1 activating + 1 contrast
        row sampled for that agent's dataset at this round. The judge does
        NOT see the full CSVs — by design — so it verifies each label's
        claims against fresh samples rather than becoming the synthesizer.
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
            sections.append(
                f"=== {ds} ===\n"
                f"label: \"{lab}\"\n\n"
                f"sampled activating rows ({JUDGE_SAMPLE_N_ACT}):\n{samples['act']}\n"
                f"sampled contrast rows ({JUDGE_SAMPLE_N_CON}):\n{samples['con']}"
            )
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
            "Rows include the model's prediction columns (pred_class/pred_conf/pred_correct or "
            "pred_value/pred_abs_err).\n\n"
            f"{data_block}\n\n"
            "TASK\n"
            "For each dataset independently, check whether the agent's label claims distinguish "
            "that dataset's activating row from its contrast row — in any combination of inputs "
            "(percentile positions, categorical frequencies), labels (target), or predictions "
            "(pred_class/pred_conf/pred_correct or pred_value/pred_abs_err).\n"
            "A claim survives only if it separates the two rows.\n\n"
            "Then decide whether the 5 labels share a cross-dataset distinguishing pattern the "
            "sampled rows corroborate. If so, the overall verdict is 'done'; otherwise 'continue' "
            f"(rounds remaining: {remaining}).\n\n"
            "FINAL CONSENSUS LABEL\n"
            "If overall_verdict is 'done', OR this is the final round "
            f"({'YES' if remaining == 0 else 'no'}), you must also write a 'final_label' — a "
            f"single-sentence consensus label ({LABEL_WORDS_MIN}-{LABEL_WORDS_MAX} words) that "
            "describes what distinguishes activating from contrast rows across the datasets, in "
            "shape-only language. You have watched the agents revise for up to 5 rounds, you have "
            "sampled fresh evidence each round, and you have the cross-dataset view; this label is "
            "your synthesis, not a vote among the agents' labels.\n"
            "  * If a cross-dataset pattern holds on all sampled pairs, state it plainly.\n"
            "  * If the signal is genuinely heterogeneous, name the shared meta-structure (e.g. "
            "    \"activating rows place tail-mass on a localized column subset; which subset\n"
            "    differs by schema\") — but only if the sampled pairs actually corroborate it.\n"
            "  * No column names. No domain terms. Use structural language.\n"
            "Otherwise omit the 'final_label' field.\n\n"
            "SHAPE-ONLY DISCIPLINE FOR YOUR OUTPUT\n"
            "The agents are required to write shape-level labels only — no column names, no\n"
            "domain terms. Your feedback and suggestions must follow the same rule so agents\n"
            "can apply them directly.\n"
            "  * Refer to columns as 'a specific numeric column', 'a column subset', 'adjacent\n"
            "    categorical positions' — never by name.\n"
            "  * Describe values by percentile / frequency / sign / count / density / spread,\n"
            "    not by domain meaning.\n"
            "  * Forbidden in supported_claims / unsupported_claims / contradicted_claims /\n"
            "    missing_signal / suggested_revision: any column name (e.g. 'GET_ACCOUNTS',\n"
            "    'Income', 'Kidhome', 'Education', 'Debtor', 'admission_grade', 'splice junction',\n"
            "    'Year_Birth'), any domain term, any hard-coded index like 'position +1'. Use\n"
            "    structural phrasing instead ('a rare-frequency binary column', 'one numeric\n"
            "    column at the low tail', 'an adjacent categorical position').\n"
            "If you find yourself about to reference a specific column or domain, restate the\n"
            "observation as a structural / statistical property.\n\n"
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
            '      "supported_claims":   ["clause clearly backed by act vs con rows", ...],\n'
            '      "unsupported_claims": ["clause too broad / vague / not visible in the sample", ...],\n'
            '      "contradicted_claims":["clause false for at least this dataset", ...],\n'
            '      "missing_signal":     "<sharper distinction you see in the rows but the label missed, or empty>",\n'
            '      "portability_risk":   {"level": "low"|"medium"|"high", "justification": "<short>"},\n'
            '      "suggested_revision": "<one tighter sentence or a short edit instruction>"\n'
            '    }\n'
            '  }\n'
            '}\n'
            "```\n"
            "Verdicts: 'accept' = claims hold; 'revise' = tighten per suggested_revision; "
            "'split' = the label is conflating two distinct regimes that should be stated separately."
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
        self.judge_verdicts.append({
            "verdict": verdict,
            "note": (response.get("overall_note") or "").strip(),
            "final_label": final_label,
            "per_dataset": {ds: per_ds[ds] for ds in self.datasets},
        })
        self._save_state()

    def is_done(self) -> bool:
        """True if the latest judge verdict is 'done' or we've hit MAX_ROUNDS."""
        if self.judge_verdicts and self.judge_verdicts[-1]["verdict"] == "done":
            return True
        return len(self.rounds) >= MAX_ROUNDS

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
            "Write a single consensus label for f_{idx} that describes what distinguishes\n"
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
            "CONTRAST DISCIPLINE — include only distinguishing properties:\n"
            "  * Before adding ANY property to the consensus label, verify activating rows DIFFER\n"
            "    from contrast rows on it — in any combination of inputs (percentile positions,\n"
            "    categorical frequencies, within-row spread, column-subset co-firing), labels\n"
            "    (target values, base-rate enrichment), or predictions (pred_class, pred_conf,\n"
            "    pred_correct / pred_value, pred_abs_err).\n"
            "  * Check this across ALL datasets: a property may distinguish act from con in one\n"
            "    dataset but not in another — in that case it is not a cross-dataset signal.\n"
            "  * Properties shared between activating and contrast are NOT feature-level signals\n"
            "    and must not appear in the label.\n"
            "  * Keep looking — properties can combine across dimensions (an input × prediction\n"
            "    joint regime; a percentile × target pair; a column-subset × confidence pattern).\n"
            "    Exhaust those combinations before settling on a label.\n\n"
            f"Output ONLY a single-sentence label ({LABEL_WORDS_MIN}-{LABEL_WORDS_MAX} words). "
            "Shape-level language only. No column names. No preamble."
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
    return s


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--feat", type=int, required=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Show datasets and round-1 prompts")

    show = sub.add_parser("show-round", help="Show prompts for a specific round")
    show.add_argument("--round", type=int, required=True)

    record = sub.add_parser("record", help="Record labels for the next round")
    record.add_argument("--labels-file", type=Path, required=True,
                         help="JSON list of label strings, one per dataset (in feat_context.datasets_used order)")

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

    args = parser.parse_args()
    pipe = ContrastiveMeshPipeline(args.model, args.feat)

    if args.cmd == "init":
        print(f"Datasets (n={len(pipe.datasets)}): {pipe.datasets}")
        prompts = pipe.round_prompts(1)
        for ds, p in zip(pipe.datasets, prompts):
            print(f"\n===== {ds} =====\n{p}")
    elif args.cmd == "show-round":
        prompts = pipe.round_prompts(args.round)
        for ds, p in zip(pipe.datasets, prompts):
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


if __name__ == "__main__":
    _cli()
