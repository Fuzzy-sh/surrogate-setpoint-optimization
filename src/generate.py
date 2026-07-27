"""
Synthetic data generator for an industrial setpoint-response problem.

The scenario
------------
An industrial site operates a fleet of production assets. Each asset has three
controllable setpoints. Periodically an operator changes one or more setpoints
(an "intervention"), and the change in output is measured over the following
48 hours.

We want a *surrogate model* of the response function

    f(state, action) -> delta_output_48h

so that candidate actions can be scored before they are applied to real equipment.

Why this is not a plain regression problem
------------------------------------------
Three properties are deliberately baked into the generator, because they are what
make the real version of this problem hard:

1. **Confounded action selection.** Operators do not pick actions at random. They
   intervene more aggressively on assets that are already struggling. So the
   historical action distribution is correlated with the state, and a model that
   ignores this will attribute the *asset's condition* to the *action*.

2. **Heteroscedastic, quality-dependent noise.** Measurement reliability varies by
   asset and period. Low-quality records are noisier, so a model that treats all
   rows as equally trustworthy will be overconfident exactly where it should not be.

3. **Regime interactions and diminishing returns.** The response to an action is
   non-linear and depends on the operating regime (e.g. a high-water-fraction asset
   responds differently). Effects saturate: doubling an action does not double the
   response, and past a point it reverses.

The generator also produces a set of *candidate* actions with no observed target,
covering action ranges that are deliberately under-represented in the history.
That is where a naive "maximise the predicted mean" recommender gets into trouble.

All parameters are fixed by `seed` so the dataset is fully reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- fleet configuration -------------------------------------------------------

N_ASSETS = 24
N_SITES = 5
N_EVENTS = 2600
N_CANDIDATES = 400

MODES = ["continuous_injection", "cyclic", "assisted_cyclic"]

# Setpoint action bounds seen in normal operation.
ACTION_BOUNDS = {
    "delta_injection_rate": (-9.0, 9.0),
    "delta_dwell_min": (-6.0, 6.0),
    "delta_cycle_min": (-14.0, 14.0),
}


def _asset_table(rng: np.random.Generator, n_assets: int = N_ASSETS) -> pd.DataFrame:
    """Per-asset latent characteristics. These are never exposed to the model."""
    N_ASSETS = n_assets
    asset_ids = [f"ASSET_{i:02d}" for i in range(1, N_ASSETS + 1)]
    return pd.DataFrame(
        {
            "asset_id": asset_ids,
            "site_id": [f"SITE_{rng.integers(1, N_SITES + 1)}" for _ in asset_ids],
            "mode": rng.choice(MODES, size=N_ASSETS, p=[0.45, 0.30, 0.25]),
            # Latent responsiveness: how strongly this asset reacts to injection.
            "latent_gain": rng.normal(1.0, 0.45, size=N_ASSETS).clip(0.25, 2.2),
            # Latent decline: assets late in life respond less and drift down.
            "latent_decline": rng.uniform(0.0, 1.0, size=N_ASSETS),
            # Per-asset measurement reliability.
            "latent_noise": rng.uniform(0.45, 1.15, size=N_ASSETS),
            # Persistent per-asset response offset (tubing condition, completion design,
            # local facility constraints). Large and asset-specific, which is exactly
            # why validation must hold out whole assets rather than random rows.
            "latent_offset": rng.normal(0.0, 0.75, size=N_ASSETS),
        }
    )


def _draw_state(rng: np.random.Generator, assets: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pre-intervention operating state for n events."""
    idx = rng.integers(0, len(assets), size=n)
    a = assets.iloc[idx].reset_index(drop=True)

    water_fraction = rng.beta(2.2, 2.0, size=n) * 100.0
    decline_penalty = 1.0 - 0.45 * a["latent_decline"].to_numpy()

    base_output = (
        rng.gamma(shape=3.0, scale=6.0, size=n) * decline_penalty
        * (1.0 - 0.004 * water_fraction)
    ).clip(0.4, None)

    header_pressure = rng.normal(2400, 430, size=n).clip(900, 4200)
    casing_pressure = header_pressure + rng.normal(1900, 520, size=n).clip(200, None)
    line_pressure = rng.normal(1150, 240, size=n).clip(400, 2400)

    # Instability rises with water fraction and with back-pressure.
    volatility = (
        0.6
        + 0.022 * water_fraction
        + 0.0009 * (line_pressure - 900).clip(0, None)
        + rng.gamma(2.0, 0.35, size=n)
    )

    reliability = (100 - rng.gamma(2.4, 4.0, size=n)).clip(45, 100)
    missed_cycles_7d = rng.poisson(np.clip((100 - reliability) / 12.0, 0, None))

    current_injection = (rng.normal(26, 9, size=n)).clip(2, 60)
    current_dwell_min = (rng.normal(22, 8, size=n)).clip(3, 55)
    current_cycle_min = (rng.normal(72, 22, size=n)).clip(20, 160)

    days_since_last_change = rng.geometric(p=0.09, size=n).clip(1, 180)

    # Data quality: a function of reliability and random operational disruption.
    quality = (
        0.55
        + 0.004 * (reliability - 45)
        + rng.normal(0, 0.11, size=n)
    ).clip(0.05, 1.0)

    downtime_flag = (rng.uniform(size=n) < 0.11).astype(int)
    constraint_flag = (rng.uniform(size=n) < 0.14).astype(int)

    return pd.DataFrame(
        {
            "asset_id": a["asset_id"].to_numpy(),
            "site_id": a["site_id"].to_numpy(),
            "operating_mode": a["mode"].to_numpy(),
            "before_output_rate": base_output.round(2),
            "before_water_fraction_pct": water_fraction.round(1),
            "before_header_pressure_kpa": header_pressure.round(0),
            "before_casing_pressure_kpa": casing_pressure.round(0),
            "before_line_pressure_kpa": line_pressure.round(0),
            "before_output_volatility": volatility.round(3),
            "before_reliability_pct": reliability.round(1),
            "before_missed_cycles_7d": missed_cycles_7d,
            "current_injection_rate": current_injection.round(2),
            "current_dwell_min": current_dwell_min.round(1),
            "current_cycle_min": current_cycle_min.round(1),
            "days_since_last_change": days_since_last_change,
            "data_quality_score": quality.round(3),
            "downtime_flag": downtime_flag,
            "facility_constraint_flag": constraint_flag,
            # latent columns, dropped before the model sees the data
            "_latent_gain": a["latent_gain"].to_numpy(),
            "_latent_decline": a["latent_decline"].to_numpy(),
            "_latent_noise": a["latent_noise"].to_numpy(),
            "_latent_offset": a["latent_offset"].to_numpy(),
        }
    )


def _choose_actions(rng: np.random.Generator, state: pd.DataFrame) -> pd.DataFrame:
    """
    Historical action selection -- deliberately confounded with state.

    Operators push injection harder on unstable, high-water-fraction assets, and
    they tend to make small adjustments on assets that were recently changed.
    """
    n = len(state)
    distress = (
        0.35 * (state["before_water_fraction_pct"].to_numpy() / 100.0)
        + 0.45 * np.tanh(state["before_output_volatility"].to_numpy() / 4.0)
        + 0.20 * (1.0 - state["before_reliability_pct"].to_numpy() / 100.0)
    )

    recency_damping = np.clip(state["days_since_last_change"].to_numpy() / 25.0, 0.55, 1.0)

    d_inj = (rng.normal(3.0 * distress - 1.0, 4.2, size=n) * recency_damping)
    d_dwell = rng.normal(0.8 * distress - 0.25, 2.8, size=n) * recency_damping
    d_cycle = rng.normal(-3.6 * distress + 1.2, 7.0, size=n) * recency_damping

    lo, hi = ACTION_BOUNDS["delta_injection_rate"]
    d_inj = d_inj.clip(lo, hi)
    lo, hi = ACTION_BOUNDS["delta_dwell_min"]
    d_dwell = d_dwell.clip(lo, hi)
    lo, hi = ACTION_BOUNDS["delta_cycle_min"]
    d_cycle = d_cycle.clip(lo, hi)

    return pd.DataFrame(
        {
            "action_delta_injection_rate": d_inj.round(2),
            "action_delta_dwell_min": d_dwell.round(2),
            "action_delta_cycle_min": d_cycle.round(2),
        }
    )


def true_response(state: pd.DataFrame, action: pd.DataFrame) -> np.ndarray:
    """
    The ground-truth response function.

    Exposed so the notebook can compare the surrogate against the true mean
    response on candidate actions -- an evaluation you never get in production,
    which is exactly why it is useful in a teaching repo.
    """
    gain = state["_latent_gain"].to_numpy()
    decline = state["_latent_decline"].to_numpy()
    water = state["before_water_fraction_pct"].to_numpy() / 100.0
    vol = state["before_output_volatility"].to_numpy()
    base = state["before_output_rate"].to_numpy()
    line_p = state["before_line_pressure_kpa"].to_numpy()

    d_inj = action["action_delta_injection_rate"].to_numpy()
    d_dwell = action["action_delta_dwell_min"].to_numpy()
    d_cycle = action["action_delta_cycle_min"].to_numpy()

    # Injection: saturating benefit that *reverses* when overdriven. Physically the
    # gain saturates quickly while the penalty keeps growing -- past a point extra
    # injection destabilises the asset rather than helping it. This is why
    # extrapolating beyond the historical envelope is dangerous rather than merely
    # uncertain.
    inj_effect = gain * (2.60 * np.tanh(d_inj / 3.4) - 0.195 * (d_inj / 3.0) ** 2)
    # High water fraction blunts the injection response.
    inj_effect *= (1.0 - 0.55 * water)

    # Dwell: mild benefit that also saturates, penalised on unstable assets.
    dwell_effect = 1.5 * np.tanh(d_dwell / 3.0) - 0.085 * d_dwell * np.tanh(vol / 3.0) - 0.035 * d_dwell**2

    # Cycle time: shortening helps up to a point, then hurts.
    cycle_effect = -0.090 * d_cycle - 0.0090 * d_cycle**2

    # Interaction: injection and cycle changes are not independent.
    interaction = 0.030 * d_inj * (-d_cycle) / 3.0

    # Back-pressure suppresses everything.
    pressure_damping = 1.0 - 0.00016 * (line_p - 800).clip(0, None)

    # Assets in decline drift down regardless of the action.
    drift = -0.55 * decline + state["_latent_offset"].to_numpy()

    scale = 0.70 + 0.030 * base
    mean = (
        scale * pressure_damping * (inj_effect + dwell_effect + cycle_effect + interaction)
        + drift
    )
    return mean


def _observe(rng: np.random.Generator, state: pd.DataFrame, mean: np.ndarray) -> np.ndarray:
    """Add heteroscedastic noise that scales with poor data quality."""
    quality = state["data_quality_score"].to_numpy()
    asset_noise = state["_latent_noise"].to_numpy()
    vol = state["before_output_volatility"].to_numpy()

    sigma = asset_noise * (0.30 + 0.95 * (1.0 - quality) + 0.055 * vol)
    noise = rng.normal(0.0, sigma)

    # Downtime events corrupt a fraction of measurements outright.
    corrupt = state["downtime_flag"].to_numpy().astype(bool) & (rng.uniform(size=len(state)) < 0.35)
    noise = np.where(corrupt, noise + rng.normal(0, 2.6, size=len(state)), noise)
    return mean + noise


def _quality_label(state: pd.DataFrame) -> np.ndarray:
    q = state["data_quality_score"].to_numpy()
    down = state["downtime_flag"].to_numpy().astype(bool)
    label = np.where((q < 0.55) | down, "low_quality", "clean")
    return label


def make_history(seed: int = 7, n_events: int = N_EVENTS,
                 n_assets: int = N_ASSETS) -> pd.DataFrame:
    """Historical intervention events, with observed 48-hour response.

    `n_events` / `n_assets` are exposed because the *sample-size regime* changes which
    modelling choices are correct. With a few hundred rows over a dozen assets, a
    regularised linear model beats gradient boosting and single-split validation is
    too noisy to trust -- conclusions that reverse at 2600 rows. Both regimes are
    worth being able to reproduce.
    """
    rng = np.random.default_rng(seed)
    assets = _asset_table(rng, n_assets)
    N_EVENTS = n_events
    state = _draw_state(rng, assets, N_EVENTS)
    action = _choose_actions(rng, state)

    mean = true_response(state, action)
    observed = _observe(rng, state, mean)

    start = np.datetime64("2023-10-01")
    offsets = np.sort(rng.integers(0, 640, size=N_EVENTS))
    event_time = start + offsets.astype("timedelta64[D]")

    df = pd.concat([state.reset_index(drop=True), action], axis=1)
    df.insert(0, "event_id", [f"EVT_{i:05d}" for i in range(1, N_EVENTS + 1)])
    df.insert(3, "event_time", event_time)
    df["event_quality_label"] = _quality_label(state)
    df["true_mean_response"] = mean.round(4)     # oracle column, for teaching only
    df["target_delta_output_48h"] = observed.round(3)
    return df


def make_candidates(seed: int = 21, n_candidates: int = N_CANDIDATES,
                    n_assets: int = N_ASSETS, history_seed: int = 7,
                    history_events: int = N_EVENTS) -> pd.DataFrame:
    """
    Candidate actions to be scored. No target is provided.

    Action ranges here extend beyond what operators historically tried, so part of
    the candidate set sits in a region where the surrogate has little support.
    Detecting that is the point.
    """
    rng = np.random.default_rng(seed)
    N_CANDIDATES = n_candidates
    assets = _asset_table(np.random.default_rng(history_seed), n_assets)  # same fleet
    state = _draw_state(rng, assets, N_CANDIDATES)

    # Two thirds of candidates sit inside the action envelope operators actually
    # used; the rest deliberately push beyond it, so the support check has
    # something real to catch. The envelope is measured from the history rather
    # than hard-coded, so the two datasets stay consistent if the generator changes.
    hist_actions = make_history(seed=history_seed, n_events=history_events,
                                n_assets=n_assets)[
        ["action_delta_injection_rate", "action_delta_dwell_min", "action_delta_cycle_min"]
    ]
    env = {c: (hist_actions[c].quantile(0.02), hist_actions[c].quantile(0.98))
           for c in hist_actions.columns}

    n_in = int(0.66 * N_CANDIDATES)
    n_out = N_CANDIDATES - n_in

    def _mix(col: str, outward: float = 2.1):
        lo, hi = env[col]
        span = hi - lo
        inside = rng.uniform(lo, hi, size=n_in)
        # Outside: beyond the envelope on either side, up to `outward` * span away.
        side = rng.choice([-1.0, 1.0], size=n_out)
        offset = rng.uniform(0.05, outward, size=n_out) * span
        outside = np.where(side > 0, hi + offset, lo - offset)
        v = np.concatenate([inside, outside])
        rng.shuffle(v)
        return v

    d_inj = _mix("action_delta_injection_rate")
    d_dwell = _mix("action_delta_dwell_min")
    d_cycle = _mix("action_delta_cycle_min")

    action = pd.DataFrame(
        {
            "action_delta_injection_rate": d_inj.round(2),
            "action_delta_dwell_min": d_dwell.round(2),
            "action_delta_cycle_min": d_cycle.round(2),
        }
    )

    df = pd.concat([state.reset_index(drop=True), action], axis=1)
    df.insert(0, "scenario_id", [f"SCEN_{i:04d}" for i in range(1, N_CANDIDATES + 1)])
    df["event_quality_label"] = _quality_label(state)
    df["true_mean_response"] = true_response(state, action).round(4)  # oracle
    return df


LATENT_COLS = ["_latent_gain", "_latent_decline", "_latent_noise", "_latent_offset"]
ORACLE_COLS = ["true_mean_response"]


def drop_oracle(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns a production model could never see."""
    return df.drop(columns=[c for c in LATENT_COLS + ORACLE_COLS if c in df.columns])


if __name__ == "__main__":
    import pathlib

    out = pathlib.Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)

    hist = make_history()
    cand = make_candidates()
    hist.to_csv(out / "history.csv", index=False)
    cand.to_csv(out / "candidates.csv", index=False)
    print(f"history:    {hist.shape[0]} events, {hist.shape[1]} columns")
    print(f"candidates: {cand.shape[0]} scenarios, {cand.shape[1]} columns")
    print(f"target mean {hist['target_delta_output_48h'].mean():.3f} "
          f"sd {hist['target_delta_output_48h'].std():.3f}")
