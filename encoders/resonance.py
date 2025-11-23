# encoders/resonance.py
# Step 5: resonance scoring (FFT, magnitude + phase)

import numpy as np

def resonance_score(q_wave: np.ndarray, m_wave: np.ndarray, K: int = 16, lam: float = 0.5) -> float:
    """
    Compute resonance score between query and memory waves.
    Combines magnitude agreement + phase alignment over top-K FFT bins.
    """
    # FFT both signals
    Q = np.fft.fft(q_wave)
    M = np.fft.fft(m_wave)

    # Magnitudes and phases
    q_mag = np.abs(Q)
    m_mag = np.abs(M)
    q_phase = np.angle(Q)
    m_phase = np.angle(M)

    # Normalize magnitudes
    q_mag = q_mag / (q_mag.max() + 1e-8)
    m_mag = m_mag / (m_mag.max() + 1e-8)

    # Select top-K bins by query magnitude
    idx = np.argsort(q_mag)[-K:]

    # Magnitude agreement
    mag_term = np.sum(q_mag[idx] * m_mag[idx])

    # Phase alignment
    phase_diff = q_phase[idx] - m_phase[idx]
    phase_term = np.sum(np.cos(phase_diff))

    return float(mag_term + lam * phase_term)

if __name__ == "__main__":
    from char_wave import char_to_wave

    q = char_to_wave("hello world", N=128)
    m1 = char_to_wave("hello world", N=128)
    m2 = char_to_wave("complex wave memory", N=128)

    print("Score (q vs m1):", resonance_score(q, m1))
    print("Score (q vs m2):", resonance_score(q, m2))
