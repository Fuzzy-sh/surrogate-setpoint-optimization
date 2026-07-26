"""Feature construction for the setpoint surrogate.

Kept deliberately small and legible. Every engineered feature is here because it
encodes something physical about the process, not because it improved a metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATE_NUMERIC = [
    "before_output_rate",
    "before_water_fraction_pct",
    "before_header_pressure_kpa",
    "before_casing_pressure_kpa",
    "before_line_pressure_kpa",
    "before_output_volatility",
    "before_reliability_pct",
    "before_missed_cycles_7d",
    "current_injection_rate",
    "current_dwell_min",
    "current_cycle_min",
    "days_since_last_change",
    "data_quality_score",
    "downtime_flag",
    "facility_constraint_flag",
]

ACTION_COLS = [
    "action_delta_injection_rate",
    "action_delta_dwell_min",
    "action_delta_cycle_min",
]

TARGET = "target_delta_output_48h"
GROUP = "asset_id"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """State + action -> model matrix.

    Engineered terms:
      * relative actions -- a +3 unit injection change means something different on
        an asset currently at 8 than at 45, so express the action as a fraction of
        the current setpoint as well as in absolute units;
      * pressure differentials -- the driving force for flow is a difference, not a
        level;
      * regime flags -- the response saturates differently in high-water-fraction
        and constrained operation;
      * action magnitude -- how far from "do nothing" this intervention is, which
        is the single most useful predictor of how much can go wrong.
    """
    X = pd.DataFrame(index=df.index)

    for c in STATE_NUMERIC:
        X[c] = df[c].astype(float)
    for c in ACTION_COLS:
        X[c] = df[c].astype(float)

    eps = 1e-6
    X["rel_injection_change"] = df["action_delta_injection_rate"] / (df["current_injection_rate"] + eps)
    X["rel_dwell_change"] = df["action_delta_dwell_min"] / (df["current_dwell_min"] + eps)
    X["rel_cycle_change"] = df["action_delta_cycle_min"] / (df["current_cycle_min"] + eps)

    X["casing_minus_header"] = df["before_casing_pressure_kpa"] - df["before_header_pressure_kpa"]
    X["header_minus_line"] = df["before_header_pressure_kpa"] - df["before_line_pressure_kpa"]

    X["high_water_regime"] = (df["before_water_fraction_pct"] > 65).astype(int)
    X["unstable_regime"] = (df["before_output_volatility"] > 3.0).astype(int)

    # L1 action magnitude, scaled per channel so the three setpoints are comparable.
    X["action_magnitude"] = (
        (df["action_delta_injection_rate"].abs() / 9.0)
        + (df["action_delta_dwell_min"].abs() / 6.0)
        + (df["action_delta_cycle_min"].abs() / 14.0)
    )

    # Interaction the physics implies: injection and cycle changes are coupled.
    X["inj_x_cycle"] = df["action_delta_injection_rate"] * (-df["action_delta_cycle_min"]) / 10.0

    mode = pd.get_dummies(df["operating_mode"], prefix="mode", dtype=float)
    for c in ["mode_continuous_injection", "mode_cyclic", "mode_assisted_cyclic"]:
        X[c] = mode[c] if c in mode.columns else 0.0

    return X


def sample_weights(df: pd.DataFrame, floor: float = 0.25) -> np.ndarray:
    """Down-weight low-quality records rather than discarding them.

    Dropping them loses real signal; trusting them equally lets measurement noise
    masquerade as process behaviour. Weighting is the honest middle.
    """
    w = df["data_quality_score"].to_numpy().astype(float)
    w = np.where(df["downtime_flag"].to_numpy() == 1, w * 0.5, w)
    return np.clip(w, floor, 1.0)
