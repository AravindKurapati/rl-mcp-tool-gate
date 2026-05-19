"""BGE encoder wrapper. Optionally loads a LoRA adapter for the RL-tuned variant.

Embeddings normalized to unit norm so cosine similarity == dot product.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class GateEncoder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        lora_adapter_path: Path | None = None,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        if lora_adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(lora_adapter_path)).eval()
        self._catalog_embeds: np.ndarray | None = None
        self._catalog_names: list[str] | None = None

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            output = self.model(**enc)
            cls = output.last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            out.append(cls.cpu().numpy())
        return np.concatenate(out, axis=0)

    def precompute_catalog(self, catalog: list[dict[str, Any]]) -> None:
        texts = [t["embed_text"] for t in catalog]
        self._catalog_embeds = self.encode(texts)
        self._catalog_names = [t["name"] for t in catalog]

    def score(self, query: str) -> np.ndarray:
        if self._catalog_embeds is None:
            raise RuntimeError("precompute_catalog must be called first")
        q = self.encode([query])
        return (q @ self._catalog_embeds.T).squeeze(0)
