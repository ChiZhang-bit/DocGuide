"""Local embedding encoding with CLS pooling and L2 normalization."""
from __future__ import annotations

from pathlib import Path
import threading

import numpy as np

from ..contracts import require

_CACHE = {}
_CACHE_LOCK = threading.Lock()


class EmbeddingEncoder:
    backend_name = "transformers_cls"
    pooling = "cls"
    query_instruction = ""

    def __init__(self, model_path, device="cpu", max_length=512):
        from transformers import AutoModel, AutoTokenizer

        require(type(max_length) is int and max_length > 0, "Invalid embedding max_length")
        self.model_path, self.device, self.max_length = Path(model_path), device, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(self.model_path), local_files_only=True).eval().to(device)

    @staticmethod
    def pool_and_normalize(hidden_state):
        import torch.nn.functional as functional

        return functional.normalize(hidden_state[:, 0, :], p=2, dim=1)

    def encode(self, texts, batch_size=16, instruction=""):
        import torch

        require(type(batch_size) is int and batch_size > 0, "Invalid embedding batch_size")
        prepared = [" ".join(str(text).split()).strip() for text in texts]
        if not prepared:
            return np.empty((0, 0), dtype=np.float32)
        batches = []
        for offset in range(0, len(prepared), batch_size):
            encoded = self.tokenizer(prepared[offset:offset + batch_size], padding=True,
                truncation=True, max_length=self.max_length, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                batches.append(self.pool_and_normalize(hidden).cpu().numpy().astype(np.float32))
        return np.concatenate(batches, axis=0)


def get_shared_encoder(model_path, device="cpu", max_length=512):
    require(model_path and str(model_path).strip(), "Set embedding_model_path to the local embedding directory")
    key = (str(Path(model_path).expanduser().resolve()), device, max_length)
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = EmbeddingEncoder(*key)
        return _CACHE[key]
