"""Long-baseline (LBL) navigation of an AUV: dead reckoning, per-epoch least squares and an EKF with outlier gating."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BEACONS = np.array([[0.0, 0.0, 49.0], [1500.0, 0.0, 49.0], [1500.0, 1500.0, 49.0], [0.0, 1500.0, 49.0]])
DEPTH = 30.0


def lawnmower(x0=250.0, y0=250.0, length=1000.0, spacing=140.0, legs=8, speed=1.5, dt=1.0) -> np.ndarray:
    """Survey path sampled every dt seconds: columns t, x, y, heading (rad), vx, vy."""
    pts = []
    x, y = x0, y0
    for k in range(legs):
        pts.append((x, y))
        y = y0 + length if k % 2 == 0 else y0
        pts.append((x, y))
        x += spacing
    pts = np.array(pts)
    out, t = [], 0.0
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        d = np.hypot(xb - xa, yb - ya)
        hdg = np.arctan2(yb - ya, xb - xa)
        n = max(1, int(d / (speed * dt)))
        for i in range(n):
            f = i / n
            out.append((t, xa + f * (xb - xa), ya + f * (yb - ya), hdg, speed * np.cos(hdg), speed * np.sin(hdg)))
            t += dt
    return np.array(out)


def dvl_velocity(track: np.ndarray, rng=None, noise=0.02, scale=0.003, heading_bias_deg=1.0, heading_noise_deg=0.3):
    """Measured north-east velocity from a DVL (speed noise and scale error) and a compass (bias and noise)."""
    rng = np.random.default_rng(rng)
    speed = np.hypot(track[:, 4], track[:, 5]) * (1 + scale) + rng.normal(0, noise, len(track))
    hdg = track[:, 3] + np.radians(heading_bias_deg) + rng.normal(0, np.radians(heading_noise_deg), len(track))
    return np.c_[speed * np.cos(hdg), speed * np.sin(hdg)]


def dead_reckoning(start_xy, vel, dt=1.0):
    return np.asarray(start_xy) + np.vstack([[0, 0], np.cumsum(vel[:-1] * dt, axis=0)])


def slant(p_xy, beacon, z=DEPTH):
    return float(np.sqrt((p_xy[0] - beacon[0]) ** 2 + (p_xy[1] - beacon[1]) ** 2 + (z - beacon[2]) ** 2))


def least_squares(ranges, beacons=BEACONS, guess=(750.0, 750.0), z=DEPTH, iters=10):
    """Gauss-Newton trilateration with known depth."""
    p = np.array(guess, float)
    for _ in range(iters):
        H, r = [], []
        for rho, b in zip(ranges, beacons):
            d = slant(p, b, z)
            H.append([(p[0] - b[0]) / d, (p[1] - b[1]) / d])
            r.append(rho - d)
        H, r = np.array(H), np.array(r)
        dp, *_ = np.linalg.lstsq(H, r, rcond=None)
        p += dp
        if np.linalg.norm(dp) < 1e-4:
            break
    return p


def gdop(p_xy, beacons=BEACONS, z=DEPTH):
    H = np.array([[(p_xy[0] - b[0]) / slant(p_xy, b, z), (p_xy[1] - b[1]) / slant(p_xy, b, z)] for b in beacons])
    return float(np.sqrt(np.trace(np.linalg.inv(H.T @ H))))


class Ekf:
    """State [east, north, heading bias (rad), DVL scale error]. The measured velocity drives the prediction after
    correction by the two calibration states, which the ranges make observable as the vehicle turns."""

    def __init__(self, xy, pos_var=1.0, hdg_var=np.radians(3) ** 2, scale_var=0.01 ** 2, q_pos=0.02 ** 2,
                 q_cal=1e-7, r=0.15 ** 2, gate=6.63, estimate_bias=True):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.est = estimate_bias
        self.P = np.diag([pos_var, pos_var, hdg_var if estimate_bias else 0.0, scale_var if estimate_bias else 0.0])
        self.q_pos, self.q_cal, self.r, self.gate = q_pos, (q_cal if estimate_bias else 0.0), r, gate
        self.rejected_in_row = 0

    @property
    def xy(self):
        return self.x[:2]

    def predict(self, v, dt=1.0):
        d, s = self.x[2], self.x[3]
        c, sn = np.cos(d), np.sin(d)
        vc = (1 - s) * np.array([c * v[0] + sn * v[1], -sn * v[0] + c * v[1]])     # undo heading bias and scale
        F = np.eye(4)
        if self.est:
            F[0:2, 2] = (1 - s) * np.array([-sn * v[0] + c * v[1], -c * v[0] - sn * v[1]]) * dt
            F[0:2, 3] = -np.array([c * v[0] + sn * v[1], -sn * v[0] + c * v[1]]) * dt
        self.x = self.x + np.r_[vc * dt, 0, 0]
        Q = np.diag([self.q_pos * dt] * 2 + [self.q_cal * dt] * 2)
        self.P = F @ self.P @ F.T + Q

    def update(self, rho, beacon, z=DEPTH, use_gate=True):
        d = slant(self.xy, beacon, z)
        H = np.array([[(self.x[0] - beacon[0]) / d, (self.x[1] - beacon[1]) / d, 0.0, 0.0]])
        S = float((H @ self.P @ H.T)[0, 0]) + self.r
        nu = rho - d
        nis = nu ** 2 / S
        if use_gate and nis > self.gate:
            self.rejected_in_row += 1
            if self.rejected_in_row < 12:          # after 3 epochs of rejections, accept again (no lock-out)
                return False, nis
        self.rejected_in_row = 0
        K = (self.P @ H.T / S).ravel()
        self.x = self.x + K * nu
        I_KH = np.eye(4) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * self.r      # Joseph form
        return True, nis
