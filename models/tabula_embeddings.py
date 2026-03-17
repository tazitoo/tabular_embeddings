"""
Tabula-8B embedding extraction.

Tabula-8B is a Llama-3 8B model fine-tuned for tabular prediction via text
serialization. Each row is converted to text like "The <col> is <val>."

The model leverages semantic information in both column names AND values.
Column names and categorical string values carry meaning the LLM can exploit.
Using generic names (f0, f1) or integer-coded categoricals (3 instead of
"Sales") strips this semantic signal.

We extract embeddings from the final hidden state of the LLM.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .base import EmbeddingExtractor, EmbeddingResult


class TabulaEmbeddingExtractor(EmbeddingExtractor):
    """Extract embeddings from Tabula-8B LLM."""

    def __init__(self, device: str = "cuda", model_path: str = None):
        super().__init__(device)
        self._model_path = model_path or "/data/models/tabula-8b"
        self._tokenizer = None

    @property
    def model_name(self) -> str:
        return "Tabula-8B"

    @property
    def available_layers(self) -> List[str]:
        return ["last_hidden_state", "pooled"]

    def load_model(self) -> None:
        """Load Tabula-8B with 8-bit quantization for 24GB VRAM."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load with 8-bit quantization to fit in 24GB
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            device_map="auto",
            quantization_config=bnb_config,
            dtype=torch.float16,
        )
        self._model.eval()

    def _serialize_row_from_series(self, row: pd.Series) -> str:
        """Convert a DataFrame row to Tabula-8B text format.

        Uses the actual column names and values (including string categoricals).
        Format: "The <column_name> is <value>." per the RTFM/Tabula paper.
        """
        parts = []
        for col_name, val in row.items():
            if pd.isna(val):
                continue
            parts.append(f"The {col_name} is {val}.")
        return " ".join(parts)

    def _serialize_row(self, row: np.ndarray, feature_names: List[str]) -> str:
        """Convert a numpy row to Tabula-8B text format (legacy fallback)."""
        parts = []
        for name, val in zip(feature_names, row):
            try:
                if np.isnan(float(val)):
                    continue
            except (ValueError, TypeError):
                pass
            parts.append(f"The {name} is {val}.")
        return " ".join(parts)

    def _serialize_context_df(
        self,
        X_context: pd.DataFrame,
        y_context: np.ndarray,
        target_name: str = "target",
    ) -> str:
        """Serialize context examples for ICL from DataFrame."""
        lines = []
        for (_, row), label in zip(X_context.iterrows(), y_context):
            row_text = self._serialize_row_from_series(row)
            lines.append(f"{row_text} The {target_name} is {label}.")
        return "\n".join(lines)

    def _serialize_context(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        feature_names: List[str],
        target_name: str = "target",
    ) -> str:
        """Serialize context examples for ICL (legacy fallback)."""
        lines = []
        for row, label in zip(X_context, y_context):
            row_text = self._serialize_row(row, feature_names)
            lines.append(f"{row_text} The {target_name} is {label}.")
        return "\n".join(lines)

    def extract_embeddings(
        self,
        X_context: Union[np.ndarray, pd.DataFrame],
        y_context: np.ndarray,
        X_query: Union[np.ndarray, pd.DataFrame],
        layers: Optional[List[str]] = None,
        task: str = "classification",
        cat_feature_indices: Optional[List[int]] = None,
    ) -> EmbeddingResult:
        """Extract embeddings from Tabula-8B.

        Accepts DataFrames with proper dtypes. Uses real column names and
        original categorical string values for text serialization, which
        provides semantic signal the LLM can leverage.
        """
        if self._model is None:
            self.load_model()

        y_context = np.asarray(y_context)
        use_df = isinstance(X_context, pd.DataFrame)

        # Limit context to avoid exceeding 8k token limit
        max_ctx = min(32, len(X_context))

        if use_df:
            ctx_text = self._serialize_context_df(
                X_context.iloc[:max_ctx], y_context[:max_ctx]
            )
        else:
            X_context = np.asarray(X_context, dtype=np.float32)
            X_query_np = np.asarray(X_query, dtype=np.float32)
            feature_names = [f"f{i}" for i in range(X_context.shape[1])]
            ctx_text = self._serialize_context(
                X_context[:max_ctx], y_context[:max_ctx], feature_names
            )

        embeddings = []
        n_query = len(X_query)

        for i in range(n_query):
            if use_df:
                query_text = self._serialize_row_from_series(X_query.iloc[i])
            else:
                query_text = self._serialize_row(X_query_np[i], feature_names)

            full_text = f"{ctx_text}\n{query_text}"

            inputs = self._tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=8000,
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model(
                    **inputs,
                    output_hidden_states=True,
                )
                # Use last hidden state of final token
                hidden = outputs.hidden_states[-1]
                emb = hidden[0, -1, :].cpu().numpy()
                embeddings.append(emb)

        embeddings = np.stack(embeddings)

        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.model_name,
            extraction_point="last_hidden_state",
            embedding_dim=embeddings.shape[1],
            n_samples=n_query,
            layer_embeddings={"last_hidden_state": embeddings},
        )


if __name__ == "__main__":
    print("Testing Tabula-8B extraction...")

    extractor = TabulaEmbeddingExtractor(device="cuda")
    extractor.load_model()

    np.random.seed(42)
    X_ctx = np.random.randn(20, 5).astype(np.float32)
    y_ctx = (np.random.rand(20) > 0.5).astype(int)
    X_query = np.random.randn(5, 5).astype(np.float32)

    result = extractor.extract_embeddings(X_ctx, y_ctx, X_query)
    print(f"Embedding shape: {result.embeddings.shape}")
