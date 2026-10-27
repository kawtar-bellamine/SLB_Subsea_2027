"""Machine learning on matched-filter features: flag range measurements that a multipath arrival has corrupted."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["peak", "first_to_max_ratio", "first_to_max_ms", "n_peaks_half", "second_peak_ratio", "first_width",
            "peak_to_median", "snr_db"]
THRESHOLD_M = 0.5     # a range error above this is an outlier for navigation


def label(df: pd.DataFrame) -> np.ndarray:
    return (np.abs(df["r_first"] - df["slant"]) > THRESHOLD_M).astype(int).to_numpy()


def train(df: pd.DataFrame, seed: int = 0):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=seed, verbose=-1)
    m.fit(df[FEATURES], label(df))
    return m


def importance(model) -> pd.DataFrame:
    g = model.booster_.feature_importance(importance_type="gain")
    return (pd.DataFrame({"feature": FEATURES, "gain_pct": 100 * g / g.sum()})
            .sort_values("gain_pct", ascending=False).reset_index(drop=True))
