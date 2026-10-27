"""Redraw every figure from the saved results, without re-running the simulations."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.argv = sys.argv[:1]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_all as R  # noqa: E402

O = R.OUT
R.fig_environment()
R.fig_ber(pd.read_csv(O / "link_ber.csv"))
R.fig_constellation()
R.fig_doppler(pd.read_csv(O / "doppler.csv"))
R.fig_real(pd.read_csv(O / "real_noise_ber.csv"), pd.read_csv(O / "noise_only_detection.csv"))
R.fig_ranging(pd.read_csv(O / "ranging_crlb.csv"), pd.read_csv(O / "ranging_multipath.csv"),
              pd.read_csv(O / "ranging_doppler.csv"))
z = np.load(O / "navigation_tracks.npz")
names = {"dead_reckoning": "dead reckoning", "least_squares_per_epoch": "least squares per epoch"}
res = pd.read_csv(O / "navigation_results.csv")
est = {}
for m in res["method"]:
    key = m.replace(" ", "_").replace("+", "p")
    est[m] = z[key]
R.fig_navigation(z["truth"], est, res, pd.read_csv(O / "navigation_pings.csv"))
print("figures redrawn")
