#!/usr/bin/env python3
"""
Auto-label SAE concepts using an LLM.

For each dictionary atom (concept):
1. Find samples with highest activation
2. Show LLM the feature patterns
3. Generate interpretable label

Usage:
    python scripts/label_concepts.py --model tabpfn --dataset adult --n-concepts 10
    python scripts/label_concepts.py --model tabpfn --dataset credit-g --dry-run
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from scripts._project_root import PROJECT_ROOT

from analysis.sparse_autoencoder import (
    SAEConfig,
    train_sae,
    measure_dictionary_richness,
)


@dataclass
class ConceptInfo:
    """Information about a single SAE concept."""
    concept_idx: int
    activation_freq: float  # How often it activates
    mean_activation: float  # Mean activation when active
    top_samples: pd.DataFrame  # Samples with highest activation
    top_activations: np.ndarray  # Activation values for top samples
    feature_correlations: Dict[str, float]  # Correlation with input features
    label: Optional[str] = None  # LLM-generated label
    description: Optional[str] = None  # LLM-generated description
    # Activation range for context
    activation_min: float = 0.0
    activation_max: float = 0.0
    activation_p25: float = 0.0
    activation_p75: float = 0.0
    # Linear probe: how predictive is this concept of the target?
    target_coefficient: Optional[float] = None  # From linear probe
    target_pvalue: Optional[float] = None


def fit_linear_probes(
    activations: np.ndarray,  # (n_samples, hidden_dim)
    y: np.ndarray,  # (n_samples,) target
    task_type: str = "classification",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit linear probes from each concept to the target.

    Returns:
        coefficients: (hidden_dim,) coefficient for each concept
        pvalues: (hidden_dim,) p-value for each concept
    """
    from scipy import stats

    n_concepts = activations.shape[1]
    coefficients = np.zeros(n_concepts)
    pvalues = np.ones(n_concepts)

    if task_type == "classification":
        # For binary classification, use logistic regression coefficient
        # or point-biserial correlation as a proxy
        from sklearn.linear_model import LogisticRegression

        # Encode target if needed
        if y.dtype == object or (hasattr(y.dtype, 'name') and 'category' in y.dtype.name):
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y.astype(str))

        # Fit probe for each concept individually
        for i in range(n_concepts):
            x = activations[:, i].reshape(-1, 1)
            if x.std() < 1e-8:
                continue
            try:
                # Point-biserial correlation (faster than logistic regression)
                corr, pval = stats.pointbiserialr(y, activations[:, i])
                coefficients[i] = corr
                pvalues[i] = pval
            except:
                pass
    else:
        # For regression, use Pearson correlation
        for i in range(n_concepts):
            if activations[:, i].std() < 1e-8:
                continue
            try:
                corr, pval = stats.pearsonr(activations[:, i], y)
                coefficients[i] = corr
                pvalues[i] = pval
            except:
                pass

    return coefficients, pvalues


def load_dataset_with_features(dataset_name: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], str]:
    """
    Load dataset with original feature names and target.

    Returns:
        (dataframe, X_processed, y, column_names, task_type)
    """
    import openml
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    # Map dataset names to OpenML IDs
    dataset_ids = {
        "adult": 1590,
        "credit-g": 31,
        "bank-marketing": 1461,
        "diabetes": 37,
        "heart": 1565,
    }

    if dataset_name in dataset_ids:
        dataset = openml.datasets.get_dataset(dataset_ids[dataset_name], download_data=True)
    else:
        # Try as OpenML ID
        try:
            dataset = openml.datasets.get_dataset(int(dataset_name), download_data=True)
        except:
            dataset = openml.datasets.get_dataset(dataset_name, download_data=True)

    X, y, _, attr_names = dataset.get_data(target=dataset.default_target_attribute)

    # Store original for display
    X_display = X.copy()

    # Determine task type
    y_array = y.values if hasattr(y, 'values') else np.array(y)
    if y_array.dtype == object or (hasattr(y_array.dtype, 'name') and 'category' in str(y_array.dtype)):
        task_type = "classification"
    elif len(np.unique(y_array[~pd.isna(y_array)])) <= 10:
        task_type = "classification"
    else:
        task_type = "regression"

    # Preprocess for model
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X[[col]])

    X_processed = X.values.astype(np.float32)
    X_processed = np.nan_to_num(X_processed, nan=0.0)

    return X_display, X_processed, y_array, list(X.columns), task_type


def load_embeddings(model_name: str, dataset_name: str) -> Optional[np.ndarray]:
    """Load cached embeddings."""
    paths = [
        PROJECT_ROOT / f"output/embeddings/tabarena/{model_name}/tabarena_{dataset_name}.npz",
        PROJECT_ROOT / f"output/embeddings/{model_name}_{dataset_name}.npz",
    ]

    for path in paths:
        if path.exists():
            data = np.load(path, allow_pickle=True)
            return data['embeddings'].astype(np.float32)

    return None


def analyze_concept(
    concept_idx: int,
    activations: np.ndarray,  # (n_samples, hidden_dim)
    X_display: pd.DataFrame,
    X_processed: np.ndarray,
    top_k: int = 10,
) -> ConceptInfo:
    """
    Analyze a single concept to prepare for labeling.

    Includes both positive (high activation) and negative (zero/low activation)
    examples for contrastive understanding.
    """
    concept_acts = activations[:, concept_idx]

    # Basic stats
    is_active = concept_acts > 0
    activation_freq = is_active.mean()
    mean_activation = concept_acts[is_active].mean() if is_active.any() else 0.0

    # Top activating samples (POSITIVE examples)
    top_indices = np.argsort(concept_acts)[-top_k:][::-1]
    top_samples = X_display.iloc[top_indices].copy()
    top_activations = concept_acts[top_indices]

    # Bottom activating samples (NEGATIVE examples - zero or lowest activation)
    bottom_indices = np.argsort(concept_acts)[:top_k]
    bottom_samples = X_display.iloc[bottom_indices].copy()
    bottom_activations = concept_acts[bottom_indices]

    # Feature correlations (what input features correlate with this concept?)
    feature_correlations = {}
    for i, col in enumerate(X_display.columns):
        if X_processed[:, i].std() > 1e-8:
            corr = np.corrcoef(concept_acts, X_processed[:, i])[0, 1]
            if not np.isnan(corr):
                feature_correlations[col] = float(corr)

    # Sort by absolute correlation
    feature_correlations = dict(
        sorted(feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    # Activation range statistics (for context in prompt)
    active_acts = concept_acts[concept_acts > 0]
    if len(active_acts) > 0:
        activation_min = float(active_acts.min())
        activation_max = float(active_acts.max())
        activation_p25 = float(np.percentile(active_acts, 25))
        activation_p75 = float(np.percentile(active_acts, 75))
    else:
        activation_min = activation_max = activation_p25 = activation_p75 = 0.0

    # Store in ConceptInfo
    concept_info = ConceptInfo(
        concept_idx=concept_idx,
        activation_freq=float(activation_freq),
        mean_activation=float(mean_activation),
        top_samples=top_samples,
        top_activations=top_activations,
        feature_correlations=feature_correlations,
        activation_min=activation_min,
        activation_max=activation_max,
        activation_p25=activation_p25,
        activation_p75=activation_p75,
    )
    # Add negative examples as extra attributes
    concept_info.bottom_samples = bottom_samples
    concept_info.bottom_activations = bottom_activations

    return concept_info


def format_concept_prompt(concept: ConceptInfo, dataset_name: str) -> str:
    """
    Format a prompt for the LLM to label a concept.

    Includes both positive (activating) and negative (non-activating) examples
    for contrastive understanding.
    """
    # Get top correlated features
    top_features = list(concept.feature_correlations.items())[:5]

    prompt = f"""You are analyzing learned features from a neural network trained on tabular data.

Dataset: {dataset_name}

I'll show you samples that ACTIVATE this neuron (positive examples) and samples that DO NOT activate it (negative examples). Your task is to identify what distinguishes them - what pattern does this neuron detect?

## Top Correlated Input Features
"""

    for feat, corr in top_features:
        direction = "higher" if corr > 0 else "lower"
        prompt += f"- {feat}: r={corr:.3f} ({direction} values → stronger activation)\n"

    prompt += f"""
## POSITIVE EXAMPLES (high activation)

"""

    # Show top activating samples
    for i, (idx, row) in enumerate(concept.top_samples.head(4).iterrows()):
        act = concept.top_activations[i]
        prompt += f"Sample P{i+1} (activation={act:.3f}):\n"
        for col in concept.top_samples.columns[:8]:  # Limit columns shown
            val = row[col]
            marker = " ⭐" if col in dict(top_features[:3]) else ""
            prompt += f"  {col}: {val}{marker}\n"
        prompt += "\n"

    prompt += """## NEGATIVE EXAMPLES (zero/low activation)

"""

    # Show non-activating samples
    if hasattr(concept, 'bottom_samples'):
        for i, (idx, row) in enumerate(concept.bottom_samples.head(4).iterrows()):
            act = concept.bottom_activations[i]
            prompt += f"Sample N{i+1} (activation={act:.3f}):\n"
            for col in concept.bottom_samples.columns[:8]:
                val = row[col]
                marker = " ⭐" if col in dict(top_features[:3]) else ""
                prompt += f"  {col}: {val}{marker}\n"
            prompt += "\n"

    prompt += f"""## Activation Statistics
- Activates on {concept.activation_freq*100:.1f}% of samples
- When active: min={concept.activation_min:.2f}, max={concept.activation_max:.2f}
- Interquartile range: [{concept.activation_p25:.2f}, {concept.activation_p75:.2f}]
- The positive examples above are in the TOP of the activation range
- The negative examples have ZERO activation
"""

    # Add target predictiveness if available
    if concept.target_coefficient is not None:
        direction = "positively" if concept.target_coefficient > 0 else "negatively"
        strength = abs(concept.target_coefficient)
        if strength > 0.3:
            pred_desc = f"strongly {direction}"
        elif strength > 0.1:
            pred_desc = f"moderately {direction}"
        else:
            pred_desc = "weakly"
        sig = "significant" if concept.target_pvalue and concept.target_pvalue < 0.05 else "not significant"
        prompt += f"""
## Target Predictiveness
- This concept is {pred_desc} correlated with the target (r={concept.target_coefficient:.3f}, {sig})
"""

    prompt += """
## Your Task
Compare the POSITIVE and NEGATIVE examples above. What distinguishes them?

Provide:
1. A short label (2-5 words) for this concept
2. A one-sentence description explaining what pattern causes activation vs non-activation

Format your response as:
LABEL: <your label>
DESCRIPTION: <your description>
"""

    return prompt


def call_llm(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """
    Call Claude API to generate concept label.

    Falls back to saving prompts if API unavailable.
    """
    # Try Anthropic API
    try:
        import anthropic
        client = anthropic.Anthropic()

        message = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.content[0].text

    except ImportError:
        pass
    except Exception as e:
        print(f"    API error: {e}")

    # Try OpenAI API as fallback
    try:
        import openai
        client = openai.OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=256,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except ImportError:
        pass
    except Exception as e:
        print(f"    OpenAI error: {e}")

    # Generate mock response based on correlations (heuristic fallback)
    return generate_heuristic_label(prompt)


def generate_heuristic_label(prompt: str) -> str:
    """
    Generate a heuristic label based on feature correlations in the prompt.
    Used as fallback when no LLM API is available.
    """
    import re

    # Extract feature correlations from prompt
    correlations = re.findall(r"- (\w+): r=([+-]?\d+\.\d+)", prompt)

    if not correlations:
        return "LABEL: Unknown\nDESCRIPTION: Could not extract features"

    # Get top feature and direction
    top_feat, top_corr = correlations[0]
    corr_val = float(top_corr)
    direction = "high" if corr_val > 0 else "low"

    # Generate simple label
    label = f"{direction.title()} {top_feat.replace('_', ' ')}"

    # Generate description
    feat_list = ", ".join([f[0].replace('_', ' ') for f in correlations[:3]])
    desc = f"Activates for samples with {direction} values of {top_feat.replace('_', ' ')}, also correlates with {feat_list}."

    return f"LABEL: {label}\nDESCRIPTION: {desc}"


def parse_llm_response(response: str) -> Tuple[str, str]:
    """Parse label and description from LLM response."""
    label = "Unknown"
    description = "Could not parse response"

    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("LABEL:"):
            label = line[6:].strip()
        elif line.upper().startswith("DESCRIPTION:"):
            description = line[12:].strip()

    return label, description


def label_concepts(
    model_name: str,
    dataset_name: str,
    n_concepts: int = 10,
    sae_config: Optional[SAEConfig] = None,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
) -> List[ConceptInfo]:
    """
    Main function to analyze and label SAE concepts.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "concept_labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_name}")
    X_display, X_processed, y, columns, task_type = load_dataset_with_features(dataset_name)
    print(f"  Shape: {X_display.shape}, Columns: {columns[:5]}...")
    print(f"  Task: {task_type}")

    print(f"\nLoading {model_name} embeddings...")
    embeddings = load_embeddings(model_name, dataset_name)

    if embeddings is None:
        print(f"  No cached embeddings found, extracting fresh...")
        # Would need to extract embeddings here
        raise ValueError(f"No embeddings found for {model_name}/{dataset_name}")

    # Subsample to match embedding count
    n_samples = len(embeddings)
    X_display = X_display.iloc[:n_samples]
    X_processed = X_processed[:n_samples]
    y = y[:n_samples]
    print(f"  Embeddings: {embeddings.shape}")

    # Normalize embeddings
    embeddings = embeddings / (embeddings.std(axis=0, keepdims=True) + 1e-8)

    # Train SAE
    print("\nTraining SAE...")
    if sae_config is None:
        sae_config = SAEConfig(
            input_dim=embeddings.shape[1],
            hidden_dim=embeddings.shape[1] * 4,
            sparsity_penalty=1e-3,
            sparsity_type="archetypal",  # Use A-SAE for stability
            archetypal_n_archetypes=min(500, n_samples),
            archetypal_simplex_temp=0.1,
            use_aux_loss=False,
            n_epochs=100,
            batch_size=64,
            learning_rate=1e-2,
        )

    torch.manual_seed(42)
    model, result = train_sae(embeddings, sae_config, verbose=False)

    richness = measure_dictionary_richness(result, input_features=embeddings, sae_model=model)
    print(f"  R²: {richness.get('explained_variance', 0):.4f}")
    print(f"  Alive features: {result.alive_features}/{sae_config.hidden_dim}")
    print(f"  L0 sparsity: {richness['l0_sparsity']:.1f}")

    # Get activations
    activations = result.feature_activations  # (n_samples, hidden_dim)

    # Fit linear probes: which concepts predict the target?
    print("\nFitting linear probes to target...")
    probe_coefficients, probe_pvalues = fit_linear_probes(activations, y, task_type)
    n_significant = (probe_pvalues < 0.05).sum()
    print(f"  {n_significant}/{len(probe_pvalues)} concepts significantly predict target (p<0.05)")

    # Find most interesting concepts (high activation variance, not dead)
    concept_scores = []
    for i in range(activations.shape[1]):
        freq = (activations[:, i] > 0).mean()
        if freq > 0.01 and freq < 0.99:  # Not dead, not always on
            variance = activations[:, i].var()
            concept_scores.append((i, variance * freq))

    concept_scores.sort(key=lambda x: x[1], reverse=True)
    top_concept_indices = [idx for idx, _ in concept_scores[:n_concepts]]

    print(f"\nAnalyzing top {n_concepts} concepts...")
    concepts = []

    for concept_idx in top_concept_indices:
        concept = analyze_concept(
            concept_idx, activations, X_display, X_processed, top_k=10
        )

        # Add linear probe results
        concept.target_coefficient = float(probe_coefficients[concept_idx])
        concept.target_pvalue = float(probe_pvalues[concept_idx])

        print(f"\n{'='*60}")
        print(f"Concept {concept_idx}")
        print(f"  Activation freq: {concept.activation_freq*100:.1f}%")
        print(f"  Activation range: [{concept.activation_min:.2f}, {concept.activation_max:.2f}]")
        print(f"  Target correlation: r={concept.target_coefficient:.3f} (p={concept.target_pvalue:.3f})")
        print(f"  Top feature correlations: {list(concept.feature_correlations.items())[:3]}")

        # Generate prompt
        prompt = format_concept_prompt(concept, dataset_name)

        if dry_run:
            print(f"\n[DRY RUN] Would send prompt ({len(prompt)} chars):")
            print("-" * 40)
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            concept.label = "[DRY RUN]"
            concept.description = "[DRY RUN]"
        else:
            print("  Calling LLM...")
            response = call_llm(prompt)
            label, description = parse_llm_response(response)
            concept.label = label
            concept.description = description
            print(f"  LABEL: {label}")
            print(f"  DESCRIPTION: {description}")

        concepts.append(concept)

    # Save results
    results = {
        "model": model_name,
        "dataset": dataset_name,
        "sae_config": {
            "type": sae_config.sparsity_type,
            "hidden_dim": sae_config.hidden_dim,
            "r2": richness.get('explained_variance', 0),
        },
        "concepts": [
            {
                "idx": c.concept_idx,
                "label": c.label,
                "description": c.description,
                "activation_freq": c.activation_freq,
                "activation_range": [c.activation_min, c.activation_max],
                "activation_iqr": [c.activation_p25, c.activation_p75],
                "target_correlation": c.target_coefficient,
                "target_pvalue": c.target_pvalue,
                "top_feature_correlations": dict(list(c.feature_correlations.items())[:5]),
            }
            for c in concepts
        ]
    }

    output_file = output_dir / f"concepts_{model_name}_{dataset_name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")

    return concepts


def main():
    parser = argparse.ArgumentParser(description="Auto-label SAE concepts with LLM")
    parser.add_argument("--model", type=str, default="tabpfn", help="TFM model name")
    parser.add_argument("--dataset", type=str, default="credit-g", help="Dataset name")
    parser.add_argument("--n-concepts", type=int, default=10, help="Number of concepts to label")
    parser.add_argument("--dry-run", action="store_true", help="Don't call LLM, just show prompts")
    parser.add_argument("--sae-type", type=str, default="archetypal",
                       choices=["l1", "topk", "archetypal"], help="SAE architecture")
    args = parser.parse_args()

    # Configure SAE based on type
    sae_config = None  # Will use defaults based on embeddings

    label_concepts(
        model_name=args.model,
        dataset_name=args.dataset,
        n_concepts=args.n_concepts,
        sae_config=sae_config,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
