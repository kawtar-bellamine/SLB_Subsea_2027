"""Shallow-water acoustic environment: sound speed, absorption, ambient noise and multipath arrivals.

Textbook models, each with its reference:
* sound speed: Mackenzie (1981), nine-term equation, valid 2 to 30 °C, 25 to 40 ppt, 0 to 8,000 m;
* absorption: Thorp (1967), in dB/km with f in kHz;
* ambient noise: Coates (1990) fit of the Wenz curves (turbulence, shipping, wind, thermal);
* multipath: image method in an isovelocity waveguide (flat surface and bottom), with Rayleigh roughness loss at the
  surface and the Rayleigh reflection coefficient of a fluid sediment at the bottom.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def mackenzie(T: float = 10.0, S: float = 35.0, D: float = 25.0) -> float:
    """Sound speed in m/s; T in °C, S in ppt, D (depth) in m."""
    return (1448.96 + 4.591 * T - 5.304e-2 * T ** 2 + 2.374e-4 * T ** 3 + 1.340 * (S - 35) + 1.630e-2 * D
            + 1.675e-7 * D ** 2 - 1.025e-2 * T * (S - 35) - 7.139e-13 * T * D ** 3)


def thorp(f_khz):
    """Absorption in dB/km."""
    f2 = np.asarray(f_khz, dtype=float) ** 2
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003


def noise_psd_db(f_khz, shipping: float = 0.5, wind: float = 5.0):
    """Ambient noise PSD in dB re 1 µPa²/Hz (Coates). shipping in [0, 1], wind in m/s."""
    f = np.maximum(np.asarray(f_khz, dtype=float), 1e-3)
    turb = 17 - 30 * np.log10(f)
    ship = 40 + 20 * (shipping - 0.5) + 26 * np.log10(f) - 60 * np.log10(f + 0.03)
    wnd = 50 + 7.5 * np.sqrt(wind) + 20 * np.log10(f) - 40 * np.log10(f + 0.4)
    th = -15 + 20 * np.log10(f)
    return 10 * np.log10(10 ** (turb / 10) + 10 ** (ship / 10) + 10 ** (wnd / 10) + 10 ** (th / 10))


def transmission_loss_db(r_m, f_khz, k: float = 1.5):
    """Spreading (k = 1.5, practical) plus absorption."""
    r = np.maximum(np.asarray(r_m, dtype=float), 1.0)
    return k * 10 * np.log10(r) + thorp(f_khz) * r / 1000


def rayleigh_bottom(grazing_rad, c1: float, c2: float = 1700.0, rho1: float = 1.0, rho2: float = 1.9) -> complex:
    """Plane-wave reflection coefficient of a fluid half-space (sand by default). Complex below the critical angle."""
    m, n = rho2 / rho1, c1 / c2
    s = np.sin(grazing_rad)
    root = np.sqrt(complex(n ** 2 - np.cos(grazing_rad) ** 2))
    return complex((m * s - root) / (m * s + root))


@dataclass
class Arrival:
    delay: float          # s, absolute
    gain: complex         # complex baseband gain, including carrier phase
    n_surface: int
    n_bottom: int
    grazing_deg: float
    length: float


@dataclass
class Waveguide:
    depth: float = 50.0
    temperature: float = 10.0
    salinity: float = 35.0
    wave_rms: float = 0.2            # m, surface roughness
    bottom_c: float = 1700.0
    bottom_rho: float = 1.9
    max_bounces: int = 6
    c: float = field(init=False)

    def __post_init__(self):
        self.c = mackenzie(self.temperature, self.salinity, self.depth / 2)

    def arrivals(self, rng_m: float, z_tx: float, z_rx: float, fc: float, min_rel_db: float = -30.0) -> list[Arrival]:
        """Eigenrays of the isovelocity waveguide by the image method, strongest first removed below min_rel_db."""
        h, c, k = self.depth, self.c, 2 * np.pi * fc / self.c
        out = []
        for n in range(self.max_bounces + 1):
            # the four image families for n full round trips (Jensen et al., Computational Ocean Acoustics, §2.4)
            fams = [(2 * n * h + z_rx - z_tx, n, n), (2 * n * h + z_rx + z_tx, n + 1, n),
                    (2 * (n + 1) * h - z_rx - z_tx, n, n + 1), (2 * (n + 1) * h - z_rx + z_tx, n + 1, n + 1)]
            for dz, ns, nb in fams:
                if dz < 0:
                    continue
                L = float(np.hypot(rng_m, dz))
                graze = float(np.arctan2(dz, rng_m))
                amp = 10 ** (-transmission_loss_db(L, fc / 1000, k=2.0) / 20)
                g = complex(amp) * (-1) ** ns
                g *= np.exp(-2 * (k * self.wave_rms * np.sin(graze)) ** 2) ** ns
                g *= rayleigh_bottom(graze, c, self.bottom_c, 1.0, self.bottom_rho) ** nb
                g *= np.exp(-2j * np.pi * fc * L / c)
                out.append(Arrival(L / c, g, ns, nb, float(np.degrees(graze)), L))
        # equal path lengths can appear twice (z_tx = z_rx): keep one ray per delay
        uniq = {}
        for a in out:
            key = round(a.delay, 9)
            if key not in uniq or abs(a.gain) > abs(uniq[key].gain):
                uniq[key] = a
        arr = sorted(uniq.values(), key=lambda a: a.delay)
        peak = max(abs(a.gain) for a in arr)
        return [a for a in arr if 20 * np.log10(abs(a.gain) / peak + 1e-30) >= min_rel_db]
