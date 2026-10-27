"""Acoustic ranging by two-way travel time (TWTT) with LFM pings, and the Cramér-Rao bound of the time of arrival."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import phy
from .channel import add_noise, propagate
from .environment import Waveguide, noise_psd_db, transmission_loss_db
from .receiver import first_path

SOURCE_LEVEL = 175.0      # dB re 1 µPa at 1 m, small modem / transponder
BAND = 3_800.0            # Hz
TURNAROUND = 0.050        # s, transponder reply delay (known)


def snr_db(slant_m: float, sl: float = SOURCE_LEVEL) -> float:
    """Passive sonar equation: SL - TL - NL in the band (Coates-Wenz noise at the band centre)."""
    nl = noise_psd_db(phy.FC / 1000) + 10 * np.log10(BAND)
    return float(sl - transmission_loss_db(slant_m, phy.FC / 1000, k=2.0) - nl)


def rms_bandwidth(x: np.ndarray, fs: int = phy.FS) -> float:
    X = np.abs(np.fft.fft(x, 8 * len(x))) ** 2
    f = np.fft.fftfreq(len(X), 1 / fs)
    m = np.sum(f * X) / np.sum(X)
    return float(np.sqrt(np.sum((f - m) ** 2 * X) / np.sum(X)))


def crlb_toa_std(snr_in_band_db, T: float = phy.CHIRP_T, noise_bw: float = 4_200.0) -> np.ndarray:
    """Lower bound of the TOA standard deviation (s): var >= 1 / (8 pi^2 beta^2 E/N0), E/N0 = SNR * T * B_noise."""
    beta = rms_bandwidth(phy.chirp())
    e_n0 = 10 ** (np.asarray(snr_in_band_db) / 10) * T * noise_bw
    return 1 / np.sqrt(8 * np.pi ** 2 * beta ** 2 * e_n0)


def doppler_bias(radial_speed: float, c: float) -> float:
    """TOA bias of an LFM up-chirp received with time compression a = v / c: the range-Doppler coupling
    (-a fc T / B) plus the shift of the compressed pulse (-a T / 2). Measured and predicted agree within 1 µs."""
    a = radial_speed / c
    return -a * (phy.FC * phy.CHIRP_T / phy.CHIRP_B + phy.CHIRP_T / 2)


@dataclass
class Toa:
    t_first: float    # s, first-path estimator
    t_max: float      # s, strongest-peak estimator
    features: dict


def estimate_toa(y: np.ndarray, lead: float, whiten: bool = False) -> Toa:
    ch = phy.chirp()
    c = phy.matched_filter(phy.whiten(y) if whiten else y, ch)
    i_max = int(np.argmax(c))
    i_first = first_path(c, i_max)
    above = c >= 0.5 * c[i_max]
    peaks = np.flatnonzero((c[1:-1] >= c[:-2]) & (c[1:-1] >= c[2:]) & above[1:-1]) + 1
    srt = np.sort(c[peaks])[::-1] if len(peaks) else np.array([c[i_max]])
    w = i_first
    while w + 1 < len(c) and c[w + 1] >= 0.5 * c[i_first]:
        w += 1
    feats = {
        "peak": float(c[i_max]),
        "first_to_max_ratio": float(c[i_first] / c[i_max]),
        "first_to_max_ms": float((i_max - i_first) / phy.FS * 1000),
        "n_peaks_half": int(len(peaks)),
        "second_peak_ratio": float(srt[1] / srt[0]) if len(srt) > 1 else 0.0,
        "first_width": int(w - i_first),
        "peak_to_median": float(c[i_max] / (np.median(c) + 1e-12)),
    }
    return Toa((phy.parabolic(c, i_first)) / phy.FS - lead, phy.parabolic(c, i_max) / phy.FS - lead, feats)


def twtt_ping(wg: Waveguide, horiz: float, z_vehicle: float, z_beacon: float, radial_speed: float = 0.0,
              noise: str = "wenz", snr: float | None = None, spread_hz: float = 0.3, rng=None,
              whiten: bool = False, sl: float = SOURCE_LEVEL, speed_known: float | None = None) -> dict:
    """One interrogation and reply. Returns range estimates (first path and strongest peak), the true slant range,
    and matched-filter features."""
    rng = np.random.default_rng(rng)
    arr = wg.arrivals(horiz, z_vehicle, z_beacon, phy.FC)
    slant = float(np.hypot(horiz, z_beacon - z_vehicle))
    s = snr_db(slant, sl) if snr is None else snr
    ch = phy.chirp()
    legs = []
    for _ in range(2):
        y, lead = propagate(ch, arr, speed=radial_speed, c=wg.c, spread_hz=spread_hz, rng=rng, lead=0.02,
                            tail=0.02)
        p = np.abs(y) ** 2
        y = add_noise(y, s, noise, rng, ref_power=p[p > 0.05 * p.max()].mean())
        legs.append(estimate_toa(y, lead, whiten))
    direct = arr[0].delay
    corr = 0.0 if speed_known is None else doppler_bias(speed_known, wg.c)
    # each leg's TOA estimate is relative to the direct-path arrival; the true one-way time is `direct`
    t_first = 2 * direct + legs[0].t_first + legs[1].t_first - 2 * corr
    t_max = 2 * direct + legs[0].t_max + legs[1].t_max - 2 * corr
    f = {k: (legs[0].features[k] + legs[1].features[k]) / 2 for k in legs[0].features}
    f["snr_db"] = s
    return {"slant": slant, "r_first": wg.c * t_first / 2, "r_max": wg.c * t_max / 2, "snr_db": s, **f}
