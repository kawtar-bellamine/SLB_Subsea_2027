"""Physical layer of a single-carrier acoustic modem (complex baseband).

Band 4 to 8 kHz around fc = 6 kHz, sampled at 24 kHz. Packet: LFM chirp preamble, guard, QPSK training and data
symbols shaped by a root-raised-cosine filter, guard, LFM chirp postamble. The two chirps give the frame timing and the
Doppler time-scale; the training symbols start the adaptive equaliser.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, resample_poly

FS = 24_000
FC = 6_000.0
RS = 3_000.0              # symbol rate (baud)
SPS = int(FS // RS)       # 8 samples per symbol
BETA = 0.25               # RRC roll-off: occupied band 3.75 kHz
SPAN = 8                  # RRC span in symbols
CHIRP_T = 0.05            # s
CHIRP_B = 3_800.0         # Hz, swept from -1.9 to +1.9 kHz in baseband
GUARD = 0.025             # s


def rrc(beta: float = BETA, sps: int = SPS, span: int = SPAN) -> np.ndarray:
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif beta and abs(abs(4 * beta * ti) - 1) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (
                np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def chirp(T: float = CHIRP_T, B: float = CHIRP_B, fs: int = FS) -> np.ndarray:
    t = np.arange(int(T * fs)) / fs
    x = np.exp(1j * np.pi * (B / T) * (t - T / 2) ** 2)
    return x * np.hanning(len(x)) ** 0.25          # gentle taper keeps sidelobes low


QPSK = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.array([0, 1, 3, 2])))   # Gray: index = 2*b0 + b1


def bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
    b = bits.reshape(-1, 2)
    return QPSK[2 * b[:, 0] + b[:, 1]]


def qpsk_to_bits(sym: np.ndarray) -> np.ndarray:
    idx = np.argmin(np.abs(sym[:, None] - QPSK[None, :]), axis=1)
    return np.stack([idx // 2, idx % 2], axis=1).reshape(-1)


def hard(sym: np.ndarray) -> np.ndarray:
    return QPSK[np.argmin(np.abs(np.atleast_1d(sym)[:, None] - QPSK[None, :]), axis=1)]


@dataclass
class Packet:
    signal: np.ndarray        # complex baseband, unit average power over the symbols
    bits: np.ndarray
    symbols: np.ndarray
    n_train: int
    pre_start: int            # sample index of the preamble start
    sym_start: int            # sample index where the first symbol pulse is centred
    post_start: int


def make_packet(n_data: int = 1024, n_train: int = 256, rng=None) -> Packet:
    rng = np.random.default_rng(rng)
    bits = rng.integers(0, 2, 2 * (n_train + n_data))
    sym = bits_to_qpsk(bits)
    up = np.zeros(len(sym) * SPS, complex)
    up[::SPS] = sym
    h = rrc()
    shaped = np.convolve(up, h)[: len(up) + len(h) - 1]
    shaped /= np.sqrt(np.mean(np.abs(shaped) ** 2))
    ch = chirp()
    g = np.zeros(int(GUARD * FS), complex)
    sig = np.concatenate([ch, g, shaped, g, ch])
    sym_start = len(ch) + len(g) + (len(h) - 1) // 2
    post = len(ch) + len(g) + len(shaped) + len(g)
    return Packet(sig, bits, sym, n_train, 0, sym_start, post)


# ------------------------------------------------------------------ receiver building blocks

def matched_filter(x: np.ndarray, template: np.ndarray, normalise: bool = True) -> np.ndarray:
    """|correlation| of x with the template. Normalised version lies in [0, 1] (cosine similarity per lag)."""
    t = template[::-1].conj()
    c = fftconvolve(x, t, mode="valid")
    if not normalise:
        return np.abs(c)
    e = np.cumsum(np.abs(np.concatenate([[0], x])) ** 2)
    win = e[len(template):] - e[: -len(template)]
    return np.abs(c) / (np.linalg.norm(template) * np.sqrt(np.maximum(win, 1e-30)))


def whiten(x: np.ndarray, nfft: int = 4096, smooth: int = 31) -> np.ndarray:
    """Divide the spectrum by a smoothed magnitude estimate (a PHAT-like pre-whitening against coloured noise and
    tonal interference)."""
    X = np.fft.fft(x, n=max(nfft, len(x)))
    mag = np.abs(X)
    k = np.ones(smooth) / smooth
    env = np.convolve(np.concatenate([mag[-smooth:], mag, mag[:smooth]]), k, mode="same")[smooth:-smooth]
    return np.fft.ifft(X / np.maximum(env, 1e-12))[: len(x)]


def parabolic(y: np.ndarray, i: int) -> float:
    """Sub-sample peak position by fitting a parabola through three points."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    return float(i) if den == 0 else i + 0.5 * (a - c) / den


def time_scale(x: np.ndarray, factor: float, fc: float = FC, fs: int = FS, up: int = 4) -> np.ndarray:
    """y[n] = x(n / factor) with the carrier term removed: undoes a Doppler compression of (factor - 1)."""
    xu = resample_poly(x, up, 1)
    n_out = int(np.floor((len(x) - 1) * factor))
    pos = np.arange(n_out) / factor * up
    y = np.interp(pos, np.arange(len(xu)), xu.real) + 1j * np.interp(pos, np.arange(len(xu)), xu.imag)
    a = factor - 1
    t = np.arange(n_out) / fs
    return y * np.exp(-2j * np.pi * fc * a * t / factor)
