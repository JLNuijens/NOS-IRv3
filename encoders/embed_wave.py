from sentence_transformers import SentenceTransformer
import numpy as np

_EPS = 1e-8

class EmbedWaveEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", N: int = 1024, device: str | None = None):
        self.N = int(N)
        self.model = SentenceTransformer(model_name, device=device)
        self._win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(self.N, dtype=np.float32) / self.N)

    def encode_text(self, text: str) -> np.ndarray:
        # ---- 1. Sentence embedding ----
        emb = self.model.encode([text], normalize_embeddings=False)[0].astype(np.float32)
        emb /= (np.linalg.norm(emb) + _EPS)
        d = emb.shape[0]

        # ---- 2. Vectorized frequency & phase ramps ----
        n = np.arange(self.N, dtype=np.float32)             # (N,)
        k = np.arange(1, d + 1, dtype=np.float32)           # (d,)
        phi = (np.pi * (k - 1)) / max(1, 2 * d)             # (d,)
        theta = 2.0 * np.pi * np.outer(k, n) / self.N + phi[:, None]  # (d, N)

        # ---- 3. Broadcast emb onto cos/sin ----
        cos_terms = np.cos(theta) * emb[:, None]            # (d, N)
        sin_terms = np.sin(theta) * emb[:, None]            # (d, N)

        # ---- 4. Sum over embedding dimensions ----
        real = cos_terms.sum(axis=0) * self._win            # (N,)
        imag = sin_terms.sum(axis=0) * self._win            # (N,)

        # ---- 5. Normalize ----
        norm = np.sqrt((real**2).sum() + (imag**2).sum()) + _EPS
        return (real / norm).astype(np.float32) + 1j * (imag / norm).astype(np.float32)
