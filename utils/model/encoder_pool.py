"""Serialize access to the shared encoder across parallel questions."""
from __future__ import annotations

import threading

from .embedding import get_shared_encoder

ENCODER_LOCK = threading.Lock()


def resolve_device(requested):
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


class SharedEncoder:
    def __init__(self, encoder, lock=ENCODER_LOCK):
        self.encoder = encoder
        self.lock = lock

    def encode(self, texts, batch_size, instruction="", role=None):
        with self.lock:
            return self.encoder.encode(texts, batch_size=batch_size, instruction=instruction)

    def __getattr__(self, name):
        return getattr(self.encoder, name)


def get_encoder(config, kind=None):
    with ENCODER_LOCK:
        encoder = get_shared_encoder(config.embedding_model_path,
            config.embedding_device, config.embedding_max_length)
    return SharedEncoder(encoder)
