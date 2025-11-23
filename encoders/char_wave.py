# encoders/char_wave.py
# Step 4: simple character-to-wave encoder

import numpy as np

def base_freq(ch: str) -> int:
    """
    Deterministic mapping from character -> frequency bin.
    Uses ASCII code as a simple base.
    """
    return ord(ch) % 64 + 1  # keeps it small and stable

def char_to_wave(text: str, N: int = 1024) -> np.ndarray:
    """
    Encode text into a complex-valued waveform of length N.
    """
    wave = np.zeros(N, dtype=np.complex64)
    text = text.lower().strip()

    for i, ch in enumerate(text):
        f = base_freq(ch)
        phi = np.pi * i / max(1, len(text))  # gentle phase progression
        for n in range(N):
            wave[n] += np.exp(1j * (2 * np.pi * f * n / N + phi))

    # Hann window to reduce leakage
    hann = np.hanning(N)
    wave = wave * hann

    # Normalize to unit length
    norm = np.linalg.norm(wave)
    if norm > 0:
        wave = wave / norm

    return wave

if __name__ == "__main__":
    w = char_to_wave("Hello CWM", N=64)
    print("Wave shape:", w.shape)
    print("First 5 samples:", w[:5])
