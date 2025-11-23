# store/memory.py

import numpy as np
from typing import Dict, Tuple
from encoders.resonance import resonance_score
import json
from pathlib import Path

Entry = Tuple[str, np.ndarray, int, float]

class MemoryStore:

    def _encode(self, text: str):
        if self.encoder is None:
            raise ValueError("No encoder set. Pass one when creating MemoryStore or call set_encoder().")
        w = self.encoder.encode_text(text)
        return w / (np.linalg.norm(w) + 1e-8)

    def __init__(self, N: int = 1024, eta: float = 0.1, decay: float = 0.5, encoder=None):
        self.N = N
        self.eta = eta
        self.decay = decay
        self.encoder = encoder
        self.step = 0
        self.store: Dict[str, Entry] = {}

    def add_document(self, doc_id: str, text: str, strength: float = 1.0):
        wave = self._encode(text)
        self.store[doc_id] = (text, wave, self.step, float(strength))

    def search(self, query: str, topk: int = 3, K: int = 16, lam: float = 0.5,
               restrict_ids: set[str] | None = None):
        """
        Search the memory for documents matching the query.
        If restrict_ids is provided, only score those doc_ids.
        """
        q_wave = self._encode(query)
        scored = []
        for doc_id, (text, wave, _, strength) in self.store.items():
            if restrict_ids is not None and doc_id not in restrict_ids:
                continue

            base = resonance_score(q_wave, wave, K=K, lam=lam)
            s = base * strength
            scored.append((doc_id, text, s, strength))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:topk], q_wave
