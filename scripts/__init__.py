"""Backward-compatible import shim for scripts reorganization.

After moving scripts into thematic subdirectories, this meta-path hook
redirects old-style imports like ``from scripts.compare_sae_cross_model``
to the new location ``from scripts.sae.compare_sae_cross_model``.

No caller needs to change on day one — existing imports keep working.
"""
import importlib
import importlib.abc
import importlib.util
import sys

# old module name -> new dotted path (relative to scripts/)
_REDIRECTS: dict[str, str] = {
    # embeddings/
    "extract_layer_embeddings": "embeddings.extract_layer_embeddings",
    "extract_all_layers": "embeddings.extract_all_layers",
    "extract_optimal_layers": "embeddings.extract_optimal_layers",
    "verify_embedding_counts": "embeddings.verify_embedding_counts",
    "build_sae_training_data": "embeddings.build_sae_training_data",
    "layerwise_cka_mitra_regression": "embeddings.layerwise_cka_mitra_regression",
    # sae/
    "compare_sae_cross_model": "sae.compare_sae_cross_model",
    "compare_sae_architectures": "sae.compare_sae_architectures",
    "analyze_sae_concepts_deep": "sae.analyze_sae_concepts_deep",
    "analyze_sae_concepts": "sae.analyze_sae_concepts",
    "sae_sweep": "sae.sae_sweep",
    "sae_tabarena_sweep": "sae.sae_tabarena_sweep",
    "sae_pareto_analysis": "sae.sae_pareto_analysis",
    "show_sae_metrics": "sae.show_sae_metrics",
    "revalidate_sae_models": "sae.revalidate_sae_models",
    "save_seed_models": "sae.save_seed_models",
    "validate_efficiency_configs": "sae.validate_efficiency_configs",
    "table_sae_architecture_comparison": "sae.table_sae_architecture_comparison",
    "ablate_matryoshka_scales": "sae.ablate_matryoshka_scales",
    "test_ghost_grads_hypothesis": "sae.test_ghost_grads_hypothesis",
    "train_tabicl_no_ghost_grads": "sae.train_tabicl_no_ghost_grads",
    "analyze_tabicl_results": "sae.analyze_tabicl_results",
    "tabicl_metrics_complete": "sae.tabicl_metrics_complete",
    # concepts/
    "label_concepts": "concepts.label_concepts",
    "label_selective_features": "concepts.label_selective_features",
    "analyze_concept_hierarchy": "concepts.analyze_concept_hierarchy",
    "analyze_concept_variance": "concepts.analyze_concept_variance",
    "build_concept_hierarchy": "concepts.build_concept_hierarchy",
    "concept_description_utils": "concepts.concept_description_utils",
    "concept_fingerprint": "concepts.concept_fingerprint",
    "detect_concept_splitting": "concepts.detect_concept_splitting",
    "embed_concept_descriptions": "concepts.embed_concept_descriptions",
    "generate_concept_descriptions": "concepts.generate_concept_descriptions",
    "generate_concept_dictionary": "concepts.generate_concept_dictionary",
    "validate_concept_embeddings": "concepts.validate_concept_embeddings",
    "row_level_probes": "concepts.row_level_probes",
    "compute_pymfe_cache": "matching.03_compute_pymfe_cache",
    "analyze_concept_regression": "matching.04_analyze_concept_regression",
    # matching/ (numbered for execution order)
    "match_sae_features": "matching.01_match_sae_concepts_mnn",
    "match_cross_model_features": "matching.archived.compute_cross_correlations",
    "build_feature_match_graph": "matching.02_build_concept_graph",
    "extend_sae_matches": "matching.archived.extend_concept_matches",
    "annotate_feature_matches": "matching.archived.annotate_concept_matches",
    "label_cross_model_concepts": "matching.05_label_cross_model_concepts",
    # intervention/
    "intervene_sae": "intervention.intervene_sae",
    "concept_causal_intervention": "intervention.concept_causal_intervention",
    "concept_importance": "intervention.concept_importance",
    "concept_performance_diagnostic": "intervention.concept_performance_diagnostic",
    "ablate_unique_concepts": "intervention.archived.ablate_unique_concepts",
    "diagnose_mispredictions": "intervention.diagnose_mispredictions",
    "transfer_concepts": "intervention.transfer_concepts",
    "transfer_virtual_nodes": "intervention.transfer_virtual_nodes",
    "embedding_translator": "intervention.embedding_translator",
    # figures/
    "figure_concept_hierarchy": "figures.figure_concept_hierarchy",
    "figure_concept_universality": "figures.figure_concept_universality",
    "figure_cross_model_heatmap": "figures.figure_cross_model_heatmap",
    "figure_cross_model_matching": "figures.figure_cross_model_matching",
    "figure_dictionary_comparison": "figures.figure_dictionary_comparison",
    "figure_feature_matching": "figures.figure_feature_matching",
    "plot_archetypal_figures": "figures.plot_archetypal_figures",
    "plot_complexity_vs_loss": "figures.plot_complexity_vs_loss",
    "plot_domain_reconstruction": "figures.plot_domain_reconstruction",
    "plot_optuna_response_surfaces": "figures.plot_optuna_response_surfaces",
    "plot_prediction_scatter": "figures.plot_prediction_scatter",
    "plot_sae_pareto_frontier": "figures.plot_sae_pareto_frontier",
    # concepts/ (was section43/)
    "section43.universal_concepts": "concepts.universal_concepts",
}


class _ScriptsFinder(importlib.abc.MetaPathFinder):
    """Redirect ``scripts.<old_name>`` imports to ``scripts.<subdir>.<name>``."""

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith("scripts."):
            return None
        suffix = fullname[len("scripts."):]
        if suffix not in _REDIRECTS:
            return None
        new_name = f"scripts.{_REDIRECTS[suffix]}"
        return importlib.util.spec_from_loader(
            fullname, loader=_RedirectLoader(new_name),
        )


class _RedirectLoader(importlib.abc.Loader):
    """Load a module by importing its redirect target."""

    def __init__(self, target):
        self._target = target

    def create_module(self, spec):
        return None  # use default semantics

    def exec_module(self, module):
        real = importlib.import_module(self._target)
        module.__dict__.update(real.__dict__)
        module.__file__ = real.__file__
        module.__loader__ = self
        module.__path__ = getattr(real, "__path__", [])
        module.__spec__ = module.__spec__
        sys.modules[module.__name__] = module


sys.meta_path.insert(0, _ScriptsFinder())
