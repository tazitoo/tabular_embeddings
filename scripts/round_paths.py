"""Where one SAE round's artifacts and results live.

A round is the unit of reproducibility: the SAE corpus and checkpoints
(`sae_training_round{N}`, `sae_tabarena_sweep_round{N}`) plus every result derived
from them under `output/round{N}/`. Bumping DEFAULT_SAE_ROUND moves the whole
pipeline onto a fresh tree, so a rerun can never overwrite a published result.
Round 10 is the NeurIPS 2026 submission; its results predate this layout and stay
at their historical paths (output/perrow_importance, output/rebuttal/...).

Kept dependency-free (only PROJECT_ROOT) so the lightest script can import it.
"""
from pathlib import Path

from scripts._project_root import PROJECT_ROOT

# Round 11 = round 10 with TabDPT's corpus and SAE regenerated on the pinned
# extraction path (see scripts/sae_corpus/promote_round.py); other models are
# links to their round-10 artifacts.
DEFAULT_SAE_ROUND = 11


def sae_sweep_dir(round: int | None = None) -> Path:
    """SAE checkpoint root ({model}/sae_*.pt) for a round."""
    r = round if round is not None else DEFAULT_SAE_ROUND
    return PROJECT_ROOT / "output" / f"sae_tabarena_sweep_round{r}"


def sae_training_dir(round: int | None = None) -> Path:
    """SAE corpus root ({model}_taskaware_{sae_training,sae_test,norm_stats}.npz)."""
    r = round if round is not None else DEFAULT_SAE_ROUND
    return PROJECT_ROOT / "output" / f"sae_training_round{r}"


def random_sae_dir(round: int | None = None) -> Path:
    """Geometry-matched random-SAE controls ({model}/sae_*_validated.pt), one per
    trained SAE of the same round. Round 10's live at the untagged
    output/sae_random_baseline; output/sae_random_baseline_round10 links to it."""
    r = round if round is not None else DEFAULT_SAE_ROUND
    return PROJECT_ROOT / "output" / f"sae_random_baseline_round{r}"


RESULTS_DIR = PROJECT_ROOT / "output" / f"round{DEFAULT_SAE_ROUND}"

# matching (scripts/matching)
DEFAULT_MATCHING_FILE = RESULTS_DIR / "sae_feature_matching_mnn_floor_p90.json"
CROSS_CORR_DIR = RESULTS_DIR / "sae_cross_correlations"
CROSS_MODEL_BASELINE_FILE = RESULTS_DIR / "sae_cross_model_random_baseline.json"

# intervention chain (scripts/intervention, scripts/rebuttal)
IMPORTANCE_DIR = RESULTS_DIR / "perrow_importance"
BASELINE_PREDICTIONS_DIR = RESULTS_DIR / "baseline_predictions"
SYMMETRIC_ABLATION_DIR = RESULTS_DIR / "symmetric_ablation"
SYMMETRIC_TRANSFER_DIR = RESULTS_DIR / "symmetric_transfer"
FORWARD_DELTAS_DIR = RESULTS_DIR / "forward_deltas"
TRANSFER_CACHES_DIR = RESULTS_DIR / "transfer_caches"
FUNCTIONAL_DECOMPOSITION_DIR = RESULTS_DIR / "functional_decomposition"

# random-SAE control arms of the same stages
IMPORTANCE_RANDOM_DIR = RESULTS_DIR / "perrow_importance_random"
SYMMETRIC_ABLATION_RANDOM_DIR = RESULTS_DIR / "symmetric_ablation_random"
SYMMETRIC_TRANSFER_RANDOM_DIR = RESULTS_DIR / "symmetric_transfer_random"
FORWARD_DELTAS_RANDOM_DIR = RESULTS_DIR / "forward_deltas_random"
FUNCTIONAL_DECOMPOSITION_RANDOM_DIR = RESULTS_DIR / "functional_decomposition_random"
PATCHING_BURNDOWN_FILE = RESULTS_DIR / "patching_burndown.csv"
PATCH_SEARCH_FILE = RESULTS_DIR / "patch_search.json"

# patching prerequisites and labeling caches (scripts/sae, scripts/concepts). Round 10's
# live at their untagged legacy paths (output/concept_activations_cache,
# output/concept_labeling, output/contrastive_examples); promote_round links the
# per-model directories that a new round does not regenerate.
CONCEPT_ACTIVATIONS_DIR = RESULTS_DIR / "concept_activations_cache"
QUALITY_CACHE_FILE = RESULTS_DIR / "concept_labeling" / "dataset_quality_cache.json"
CONTRASTIVE_EXAMPLES_DIR = RESULTS_DIR / "contrastive_examples"


def off_manifold_dump_file(arm: str) -> Path:
    """The locked patching cell (off_manifold_concept_stratification --dump) per arm."""
    return RESULTS_DIR / f"off_manifold_concept_dump_{arm}.csv"
