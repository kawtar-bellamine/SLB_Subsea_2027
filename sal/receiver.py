"""Receiver: detection, Doppler estimation and compensation, timing, and adaptive equalisation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import phy

MODES = {
    # name: (feed-forward taps at T/2, cursor index, feedback taps)
    "single-tap": (1, 0, 0),
    "linear": (48, 8, 0),
    "dfe": (48, 8, 32),
}


@dataclass
class Result:
    ber: float
    a_hat: float
    detected: bool
    eq_out: np.ndarray = field(repr=False)
    mse_db: float = np.nan
    sync_peak: float = np.nan


class RlsDfe:
    """Fractionally spaced decision-feedback equaliser, RLS adaptation, second-order PLL (Stojanovic et al. 1994)."""

    def __init__(self, n_ff: int, n_fb: int, lam: float = 0.995, delta: float = 10.0, k1: float = 0.02,
                 k2: float = 0.002):
        self.n_ff, self.n_fb, self.lam = n_ff, n_fb, lam
        n = n_ff + n_fb
        self.w = np.zeros(n, complex)
        self.P = np.eye(n, dtype=complex) * delta
        self.k1, self.k2 = k1, k2
        self.theta, self.acc = 0.0, 0.0
        self.past = np.zeros(n_fb, complex)

    def step(self, u: np.ndarray, d_train: complex | None):
        v = np.concatenate([u * np.exp(-1j * self.theta), self.past])
        z = np.vdot(self.w, v)
        d = d_train if d_train is not None else phy.hard(z)[0]
        e = d - z
        Pv = self.P @ v
        k = Pv / (self.lam + np.vdot(v, Pv).real)
        self.w = self.w + k * np.conj(e)
        self.P = (self.P - np.outer(k, v.conj() @ self.P)) / self.lam
        eps = np.imag(z * np.conj(d))
        self.acc += eps
        self.theta += self.k1 * eps + self.k2 * self.acc
        if self.n_fb:
            self.past = np.concatenate([[d], self.past[:-1]])
        return z, d, e


def first_path(c: np.ndarray, i_max: int, rel: float = 0.5, back: float = 0.015) -> int:
    """Earliest local maximum within `back` seconds before the strongest peak that reaches rel * peak."""
    lo = max(1, i_max - int(back * phy.FS))
    thr = rel * c[i_max]
    for i in range(lo, i_max):
        if c[i] >= thr and c[i] >= c[i - 1] and c[i] >= c[i + 1]:
            return i
    return i_max


def detect_and_doppler(rx: np.ndarray, pkt: phy.Packet, whiten: bool = False):
    ch = phy.chirp()
    x = phy.whiten(rx) if whiten else rx
    c = phy.matched_filter(x, ch)
    L = pkt.post_start - pkt.pre_start
    lim = max(1, len(c) - int(L * 0.98))
    i1 = int(np.argmax(c[:lim]))
    lo, hi = i1 + int(L * 0.985), min(len(c), i1 + int(L * 1.015))
    if hi <= lo:
        return None
    # Doppler from the shift between the whole multipath signatures of preamble and postamble (cross-correlation of
    # the two correlation envelopes), not from two single peaks that fading can swap between paths.
    W = int(0.015 * phy.FS)
    a0 = max(0, i1 - W)
    sig1 = c[a0:i1 + W]
    seg = c[max(0, lo - W):min(len(c), hi + W)]
    if len(seg) <= len(sig1):
        return None
    xc = np.correlate(seg - seg.mean(), sig1 - sig1.mean(), mode="valid")
    k = int(np.argmax(xc))
    lag = phy.parabolic(xc, k) + max(0, lo - W) - a0
    t1 = phy.parabolic(c, i1)
    return t1, t1 + lag, L / lag - 1, float(c[i1])


def receive(rx: np.ndarray, pkt: phy.Packet, mode: str = "dfe", doppler: bool = True, whiten: bool = False,
            threshold: float = 0.3) -> Result:
    det = detect_and_doppler(rx, pkt, whiten)
    if det is None or det[3] < threshold:
        return Result(0.5, np.nan, False, np.zeros(0, complex), sync_peak=np.nan if det is None else det[3])
    t1, t2, a_hat, peak = det
    y = rx
    if doppler:
        y = phy.time_scale(rx, 1 + a_hat)
        c2 = phy.matched_filter(phy.whiten(y) if whiten else y, phy.chirp())
        g = int(round(t1 * (1 + a_hat)))
        lo = max(0, g - phy.SPS * 20)
        i = lo + int(np.argmax(c2[lo:g + phy.SPS * 20]))
        t1 = phy.parabolic(c2, first_path(c2, i))
    else:
        c2 = phy.matched_filter(phy.whiten(y) if whiten else y, phy.chirp())
        t1 = phy.parabolic(c2, first_path(c2, int(round(t1))))
    h = phy.rrc()
    z = np.convolve(y, h)
    gd = (len(h) - 1) / 2
    first = int(t1 + pkt.sym_start - pkt.pre_start + gd)
    span = z[max(0, first):first + len(pkt.symbols) * phy.SPS]
    z = z / np.sqrt(np.mean(np.abs(span) ** 2) + 1e-30)
    n_ff, cursor, n_fb = MODES[mode]
    eq = RlsDfe(n_ff, n_fb)
    half = phy.SPS / 2
    offs = (np.arange(n_ff) - cursor) * half
    grid = np.arange(len(z))
    out, dec = [], []
    n_sym = len(pkt.symbols)
    for n in range(n_sym):
        centre = t1 + (pkt.sym_start - pkt.pre_start) + n * phy.SPS + gd
        pos = centre - offs[::-1] if n_ff > 1 else np.array([centre])
        pos = np.clip(pos, 0, len(z) - 1)
        u = np.interp(pos, grid, z.real) + 1j * np.interp(pos, grid, z.imag)
        train = pkt.symbols[n] if n < pkt.n_train else None
        zn, dn, _ = eq.step(u, train)
        out.append(zn)
        dec.append(dn)
    out = np.array(out)
    data_out = out[pkt.n_train:]
    bits_hat = phy.qpsk_to_bits(data_out)
    bits = pkt.bits[2 * pkt.n_train:]
    ber = float(np.mean(bits_hat != bits))
    err = data_out - pkt.symbols[pkt.n_train:]
    return Result(ber, a_hat, True, out, float(10 * np.log10(np.mean(np.abs(err) ** 2))), peak)
