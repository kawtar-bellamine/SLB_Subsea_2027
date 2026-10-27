"""Time-varying multipath channel with Doppler, and in-band noise (synthetic Wenz-shaped or a real recording)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, firwin, lfilter, resample_poly

from .environment import Arrival, noise_psd_db
from .phy import FC, FS

REAL = Path(__file__).resolve().parent.parent / "data" / "real" / "glacier-bay-humpback.hq.ogg"
LP = firwin(129, 2_100, fs=FS)       # keeps the 3.75 kHz modem band, rejects the rest


def propagate(x: np.ndarray, arrivals: list[Arrival], speed: float = 0.0, c: float = 1500.0,
              spread_hz: float = 0.0, lead: float = 0.01, tail: float = 0.03, rng=None,
              up: int = 4) -> tuple[np.ndarray, float]:
    """Complex-baseband channel output and the time (s, in the output) where the first arrival starts.

    speed > 0 means the platforms approach: the waveform is compressed by a = speed / c. Rays that touched the
    moving sea surface get a random phase process whose Doppler spread grows with the number of surface bounces.
    """
    rng = np.random.default_rng(rng)
    a = speed / c
    t0 = arrivals[0].delay
    dur = len(x) / FS / (1 + a) + (arrivals[-1].delay - t0) + lead + tail
    n = int(dur * FS)
    t = np.arange(n) / FS
    xu = resample_poly(x, up, 1)
    grid = np.arange(len(xu))
    y = np.zeros(n, complex)
    for ar in arrivals:
        tau = ar.delay - t0 + lead
        pos = ((1 + a) * (t - tau)) * FS * up
        ok = (pos >= 0) & (pos <= len(xu) - 1)
        seg = np.zeros(n, complex)
        seg[ok] = np.interp(pos[ok], grid, xu.real) + 1j * np.interp(pos[ok], grid, xu.imag)
        g = ar.gain
        if spread_hz > 0 and ar.n_surface > 0:
            # Gaussian-shaped Doppler spectrum: filtered complex noise, unit power
            sig = spread_hz * np.sqrt(ar.n_surface)
            w = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            k = np.exp(-0.5 * ((np.arange(-4 * FS // 10, 4 * FS // 10 + 1) / FS) * 2 * np.pi * sig) ** 2)
            k = k[np.abs(k) > 1e-4]
            fade = fftconvolve(w, k / np.linalg.norm(k), mode="same") / np.sqrt(2)
            g = g * (0.6 + 0.4 * fade)          # partly coherent, partly scattered
        y += g * seg
    y *= np.exp(2j * np.pi * FC * a * t)
    return y, lead


def to_baseband(passband: np.ndarray, fs: int = FS) -> np.ndarray:
    t = np.arange(len(passband)) / fs
    bb = 2 * passband * np.exp(-2j * np.pi * FC * t)
    return lfilter(LP, 1, bb)[len(LP) // 2:]


def wenz_noise(n: int, rng=None, shipping: float = 0.5, wind: float = 5.0) -> np.ndarray:
    """Real passband Gaussian noise with the Coates-Wenz spectral shape, taken to baseband."""
    rng = np.random.default_rng(rng)
    m = n + len(LP)
    f = np.fft.rfftfreq(m, 1 / FS)
    amp = 10 ** (noise_psd_db(f / 1000, shipping, wind) / 20)
    spec = amp * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    return to_baseband(np.fft.irfft(spec, m))[:n]


@lru_cache(maxsize=1)
def real_recording() -> np.ndarray:
    """Glacier Bay hydrophone recording (NPS, public domain), resampled from 44.1 kHz to 24 kHz, passband."""
    import soundfile as sf
    x, sr = sf.read(str(REAL), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert sr == 44_100
    return resample_poly(x, 160, 294)


def real_noise(n: int, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    rec = real_recording()
    m = n + len(LP)
    i = rng.integers(0, len(rec) - m)
    return to_baseband(rec[i:i + m])[:n]


def add_noise(y: np.ndarray, snr_db: float, kind: str = "wenz", rng=None, ref_power: float | None = None):
    """Adds in-band noise so that (signal power / noise power) = snr_db. ref_power: signal power to use (else the
    mean power of y over its non-silent part)."""
    rng = np.random.default_rng(rng)
    if ref_power is None:
        p = np.abs(y) ** 2
        ref_power = p[p > 0.05 * p.max()].mean()
    w = wenz_noise(len(y), rng) if kind == "wenz" else real_noise(len(y), rng)
    w = w / np.sqrt(np.mean(np.abs(w) ** 2))
    return y + w * np.sqrt(ref_power / 10 ** (snr_db / 10))
