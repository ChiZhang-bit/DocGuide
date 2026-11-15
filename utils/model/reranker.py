"""Lazy local reranker returning raw logits."""
from __future__ import annotations

from pathlib import Path
import threading

from ..contracts import require

_CACHE = {}
_CACHE_LOCK = threading.Lock()


class Reranker:
    backend_name = "transformers_sequence_classification"

    def __init__(self, model_path, device="cpu", max_length=512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_path, self.device, self.max_length = str(Path(model_path).expanduser().resolve()), device, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
            local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path,
            local_files_only=True, trust_remote_code=False).eval().to(device)
        self.lock = threading.Lock()

    def score_pairs(self, pairs, batch_size=8):
        """Return one raw logit per question/evidence pair."""
        import torch

        require(type(batch_size) is int and batch_size > 0, "Invalid reranker batch size")
        scores = []
        with self.lock, torch.inference_mode():
            for offset in range(0, len(pairs), batch_size):
                batch = pairs[offset:offset + batch_size]
                encoded = self.tokenizer([q for q, _ in batch], [text for _, text in batch],
                    padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                require(logits.numel() == len(batch), "Reranker must output one logit per pair")
                scores.extend(logits.reshape(-1).float().cpu().tolist())
        return scores


def get_shared_reranker(config):
    from .encoder_pool import resolve_device

    key = (str(Path(config.reranker_model_path).expanduser().resolve()),
           resolve_device(config.reranker_device), config.reranker_max_length)
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = Reranker(*key)
        return _CACHE[key]
