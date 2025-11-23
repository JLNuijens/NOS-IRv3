# E:\cwm\encoders\factory.py
from __future__ import annotations
from typing import Optional
import numpy as np
from .char_wave import char_to_wave

class CharWaveEncoder:
    def __init__(self, N: int = 1024):
        self.N = int(N)
    def encode_text(self, text: str) -> np.ndarray:
        return char_to_wave(text, N=self.N)

def make_encoder(name: str = "char",
                 N: int = 1024,
                 model_name: Optional[str] = "all-MiniLM-L6-v2",
                 device: Optional[str] = None):
    key = (name or "char").lower()
    if key in ("char", "char_wave"):
        return CharWaveEncoder(N=N)
    if key in ("embed", "embed_wave", "st", "sentence"):
        # Lazy import so char mode works without sentence-transformers
        try:
            from .embed_wave import EmbedWaveEncoder
        except Exception as e:
            raise ImportError(
                "Embed encoder requested but 'sentence-transformers' is not available. "
                "Install it with: pip install sentence-transformers"
            ) from e
        return EmbedWaveEncoder(model_name=model_name, N=N, device=device)
    raise ValueError(f"Unknown encoder: {name}")
