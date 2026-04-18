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
CONVERGENCE_THRESHOLD = 0.80
MAX_ROUNDS = 6
LABEL_WORDS_MIN, LABEL_WORDS_MAX = 10, 25
NOMIC_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

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
        self.synthesis = data.get("synthesis")

    def _save_state(self):
        self._state_path.write_text(json.dumps({
            "model": self.model,
            "feat_idx": self.feat_idx,
            "datasets": self.datasets,
            "rounds": self.rounds,
            "similarities": self.similarities,
            "synthesis": self.synthesis,
        }, indent=2))

    def reset(self):
        self.rounds = []
        self.similarities = []
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
            "    subset? With tight or dispersed within-row spread of percentiles? Is the target\n"
            "    enriched relative to base rate in activating vs contrast?\n"
            "  * Do NOT mention specific column names or domain terms.\n\n"
            "Note whether the pattern holds at p80/p90 too, or only at top activations.\n"
            "Also note whether activating rows concentrate on confident-correct,\n"
            "confident-wrong, or uncertain predictions (vs. contrast) — this sharpens\n"
            "whether the feature tracks a decisive signal, a confounder, or ambiguity.\n\n"
            f"Output ONLY a single-sentence label ({LABEL_WORDS_MIN}-{LABEL_WORDS_MAX} words) "
            "describing the structural / distributional fingerprint. No column names. No domain terms. "
            "No preamble, no explanation."
        )

    def ring_prompt(self, ds: str, peer_ds: str, peer_label: str, round_num: int) -> str:
        return (
            f"You are the agent for '{ds}', labeling SAE feature f_{self.feat_idx}.\n"
            f"This is round {round_num} of ring-consensus mesh labeling.\n\n"
            f"{self._preprocessing_block()}\n\n"
            f"YOUR DATA ROWS\n{self._dataset_block(ds, full_csv=True)}\n\n"
            f"PEER EVIDENCE (from '{peer_ds}')\n"
            f"  peer's current label: \"{peer_label}\"\n"
            f"  {self._dataset_block(peer_ds, full_csv=False)}\n\n"
            "TASK\n"
            "Revise your label in light of the peer's hypothesis.\n"
            "CRITICAL: describe SHAPE and STATISTICAL FINGERPRINT only.\n"
            "  * IGNORE column names and ordering. The pattern must be invariant under column\n"
            "    permutation and renaming.\n"
            "  * Describe distributional / structural properties (density, spread, co-variation,\n"
            "    magnitude regime, clustering).\n"
            "  * No column names. No domain terms.\n\n"
            "If the peer's shape-level pattern also fits your activating rows, merge framings.\n"
            "If your data contradicts theirs at the shape level, hold your ground but sharpen.\n"
            "Use the pred_class / pred_conf / pred_correct columns (or pred_value / pred_abs_err\n"
            "for regression) to check whether the feature fires on confident-correct vs.\n"
            "confident-wrong vs. uncertain rows — include that in the label when it distinguishes\n"
            "activating from contrast.\n"
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

    def final_label(self) -> str:
        if self.synthesis:
            return self.synthesis
        if not self.rounds:
            raise RuntimeError("No rounds recorded yet")
        return self.most_central_from_peak()

    # ── Synthesizer ─────────────────────────────────────────────────────
    def synthesizer_prompt(self) -> str:
        """Prompt for a single agent that sees all n labels + all n CSVs."""
        if not self.rounds:
            raise RuntimeError("No rounds recorded — cannot synthesize")
        peak = self.peak_round()
        peak_labels = self.rounds[peak - 1]
        label_block = "\n".join(
            f"  [{ds}] \"{lab}\""
            for ds, lab in zip(self.datasets, peak_labels)
        )
        data_block = "\n\n".join(
            f"=== {ds} ===\n{self._dataset_block(ds, full_csv=True)}"
            for ds in self.datasets
        )
        return (
            f"You are the synthesizer for SAE feature f_{self.feat_idx} "
            f"({self.preprocessing.get('model', self.model)} model).\n"
            f"{len(self.datasets)} per-dataset agents ran a ring-consensus mesh and produced "
            f"the following labels (from the peak-cosine round, round {peak}):\n\n"
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
            "    Strip that away. If their domain descriptions share a shape-level commonality,\n"
            "    describe that commonality. If not, the feature is genuinely polysemantic at\n"
            "    the structural level — say so.\n"
            "  * Check the pred_class / pred_conf / pred_correct columns (or pred_value /\n"
            "    pred_abs_err) across datasets: if activating rows cluster on confident-correct,\n"
            "    confident-wrong, or uncertain predictions — relative to contrast — that\n"
            "    behavioral regime belongs in the label.\n\n"
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
        output_path.write_text(json.dumps({
            "model": self.model,
            "feat_idx": self.feat_idx,
            "label": label,
            "synthesis_used": self.synthesis is not None and label == self.synthesis,
            "datasets_used": self.datasets,
            "rounds_completed": len(self.rounds),
            "peak_round": self.peak_round() if self.similarities else None,
            "converged": self.converged(),
            "plateaued": self.plateaued(),
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
        print(f"Round {len(pipe.rounds)}: mean pairwise cosine = {sim:.4f}")
        print(f"Converged: {pipe.converged()}")
    elif args.cmd == "status":
        for i, (round_labels, sim) in enumerate(zip(pipe.rounds, pipe.similarities), 1):
            print(f"Round {i}: cosine={sim:.4f}")
            for ds, lab in zip(pipe.datasets, round_labels):
                print(f"  [{ds}] {lab}")
        if pipe.synthesis:
            print(f"\nSynthesized: {pipe.synthesis}")
        if pipe.similarities:
            print(f"\nPeak round: {pipe.peak_round()} (cosine {max(pipe.similarities):.4f})")
        print(f"Converged: {pipe.converged()}   Plateaued: {pipe.plateaued()}")
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


if __name__ == "__main__":
    _cli()
