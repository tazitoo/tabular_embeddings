"""
Tabula-8B embedding extraction.

Tabula-8B is a Llama-3 8B model fine-tuned for tabular prediction via text
serialization. Each row is converted to text like "the <col> is <val>".

We extract embeddings from the final hidden state of the LLM.
"""

from typing import Dict, List, Optional

import numpy as np
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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load with 8-bit quantization to fit in 24GB
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16,
        )
        self._model.eval()

    def _serialize_row(self, row: np.ndarray, feature_names: List[str]) -> str:
        """Convert a row to Tabula-8B text format."""
        parts = []
        for name, val in zip(feature_names, row):
            if np.isnan(val) if isinstance(val, float) else False:
                continue
            parts.append(f"the {name} is {val}")
        return ", ".join(parts)

    def _serialize_context(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        feature_names: List[str],
        target_name: str = "target",
    ) -> str:
        """Serialize context examples for ICL."""
        lines = []
        for row, label in zip(X_context, y_context):
            row_text = self._serialize_row(row, feature_names)
            lines.append(f"{row_text}, the {target_name} is {label}")
        return "\n".join(lines)

    def extract_embeddings(
        self,
        X_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        layers: Optional[List[str]] = None,
        task: str = "classification",
    ) -> EmbeddingResult:
        """Extract embeddings from Tabula-8B."""
        if self._model is None:
            self.load_model()

        X_context = np.asarray(X_context, dtype=np.float32)
        y_context = np.asarray(y_context)
        X_query = np.asarray(X_query, dtype=np.float32)

        feature_names = [f"f{i}" for i in range(X_context.shape[1])]

        # Limit context to avoid exceeding 8k token limit
        max_ctx = min(32, len(X_context))
        ctx_text = self._serialize_context(
            X_context[:max_ctx], y_context[:max_ctx], feature_names
        )

        embeddings = []
        for row in X_query:
            query_text = self._serialize_row(row, feature_names)
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
            n_samples=len(X_query),
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
