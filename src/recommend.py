"""Safe scoring of candidate setpoint actions.

The central argument of this repo
---------------------------------
Ranking candidate actions by predicted mean is the wrong objective when the model
is going to drive real equipment. The highest predicted mean tends to sit exactly
where the model is least trustworthy -- far from the historical action distribution,
where the fit is extrapolating and nothing contradicts it.

So we rank on a **lower confidence bound** and apply hard guardrails on top:

1. **Support check.** How much training data resembles this state-action pair? Actions
   outside the historical envelope are flagged, not scored.
2. **Risk-adjusted score.** LCB = mean - kappa * total_sd, penalising uncertainty
   instead of ignoring it.
3. **Downside constraint.** Reject anything whose plausible worst case breaches an
   operational tolerance, however good its mean looks.
4. **Magnitude cap.** Prefer the smallest action achieving the benefit; large moves
   carry operational risk this model never sees.

The result is a recommender that declines to answer when it should.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _mean_knn_distance(query: np.ndarray, reference: np.ndarray, k: int,
                       drop_self: bool = False) -> np.ndarray:
    """Mean Euclidean distance to the k nearest reference points.

    Implemented directly rather than via sklearn's NearestNeighbors: the data is
    small enough that a chunked brute-force pass is fast, and it keeps the repo
    free of a BLAS-threading dependency that breaks on some installs.
    """
    out = np.empty(len(query), dtype=float)
    take = k + 1 if drop_self else k
    chunk = 512
    for start in range(0, len(query), chunk):
        block = query[start:start + chunk]
        d = np.sqrt(((block[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2))
        part = np.partition(d, take - 1, axis=1)[:, :take]
        part.sort(axis=1)
        if drop_self:
            part = part[:, 1:]
        out[start:start + len(block)] = part.mean(axis=1)
    return out

ACTION_COLS = [
    "action_delta_injection_rate",
    "action_delta_dwell_min",
    "action_delta_cycle_min",
]


class SupportChecker:
    """Distance to the historical state-action manifold.

    A candidate far from every training point is one the surrogate has no basis to
    score. We measure that with mean k-NN distance in a standardised feature space
    and calibrate the threshold on the training set itself, so 'far' is defined
    relative to the density the model was actually fitted on.
    """

    def __init__(self, k: int = 12, quantile: float = 0.98):
        self.k = k
        self.quantile = quantile
        self.scaler = StandardScaler()
        self.reference_: np.ndarray | None = None
        self.threshold_: float = np.inf

    def fit(self, X: pd.DataFrame):
        Z = self.scaler.fit_transform(X.to_numpy(dtype=float))
        self.reference_ = Z
        train_dist = _mean_knn_distance(Z, Z, self.k, drop_self=True)
        self.threshold_ = float(np.quantile(train_dist, self.quantile))
        return self

    def distance(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.scaler.transform(X.to_numpy(dtype=float))
        return _mean_knn_distance(Z, self.reference_, self.k)

    def in_support(self, X: pd.DataFrame) -> np.ndarray:
        return self.distance(X) <= self.threshold_


def action_envelope(history: pd.DataFrame, pad: float = 0.05) -> dict:
    """Per-channel action range actually observed in operation."""
    env = {}
    for c in ACTION_COLS:
        lo, hi = history[c].quantile(0.01), history[c].quantile(0.99)
        span = hi - lo
        env[c] = (lo - pad * span, hi + pad * span)
    return env


def within_envelope(df: pd.DataFrame, envelope: dict) -> np.ndarray:
    ok = np.ones(len(df), dtype=bool)
    for c, (lo, hi) in envelope.items():
        ok &= (df[c] >= lo) & (df[c] <= hi)
    return ok


def score_candidates(
    candidates: pd.DataFrame,
    X_cand: pd.DataFrame,
    preds: pd.DataFrame,
    support: SupportChecker,
    envelope: dict,
    kappa: float = 1.0,
    downside_tolerance: float = -1.0,
    max_action_magnitude: float = 2.2,
) -> pd.DataFrame:
    """Attach risk-adjusted scores and an explicit recommend / reject decision.

    Parameters
    ----------
    kappa
        Risk aversion. 0 reproduces naive mean-ranking; 1.0 is a mild penalty;
        2.0 is conservative. Exposed because the right value is a business
        decision about downtime cost, not a statistical one.
    downside_tolerance
        Reject if the plausible worst case is below this. Defaults to -1.0 units,
        i.e. we will not knowingly risk a material production loss.
    """
    out = candidates.copy()
    out["pred_mean"] = preds["pred_mean"].to_numpy()
    out["pred_lo"] = preds["pred_lo"].to_numpy()
    out["pred_hi"] = preds["pred_hi"].to_numpy()
    out["epistemic_sd"] = preds["epistemic_sd"].to_numpy()

    # Aleatoric spread implied by the interval, plus epistemic disagreement.
    aleatoric_sd = (out["pred_hi"] - out["pred_lo"]) / 2.563   # 80% interval -> sd
    out["aleatoric_sd"] = aleatoric_sd
    out["total_sd"] = np.sqrt(aleatoric_sd**2 + out["epistemic_sd"] ** 2)

    out["lcb"] = out["pred_mean"] - kappa * out["total_sd"]

    out["support_distance"] = support.distance(X_cand)
    out["in_support"] = support.in_support(X_cand)
    out["in_envelope"] = within_envelope(candidates, envelope)

    out["action_magnitude"] = (
        candidates["action_delta_injection_rate"].abs() / 9.0
        + candidates["action_delta_dwell_min"].abs() / 6.0
        + candidates["action_delta_cycle_min"].abs() / 14.0
    )

    reasons = []
    for _, r in out.iterrows():
        why = []
        if not r["in_envelope"]:
            why.append("action outside historical envelope")
        if not r["in_support"]:
            why.append("state-action pair unlike training data")
        if r["pred_lo"] < downside_tolerance:
            why.append(f"downside risk below {downside_tolerance}")
        if r["action_magnitude"] > max_action_magnitude:
            why.append("action magnitude too large")
        if r["lcb"] <= 0:
            why.append("no confident expected gain")
        reasons.append("; ".join(why) if why else "")

    out["reject_reason"] = reasons
    out["recommended"] = out["reject_reason"] == ""
    return out.sort_values("lcb", ascending=False)


def compare_policies(scored: pd.DataFrame, truth_col: str = "true_mean_response", top_n: int = 20) -> pd.DataFrame:
    """Naive mean-ranking vs the safe rule, judged against the true response.

    In production the truth column does not exist. Here it does, which lets us show
    the thing that is otherwise an article of faith: what the safety rule costs in
    average upside, and what it buys in avoided losses.
    """
    rows = []

    naive = scored.nlargest(top_n, "pred_mean")
    rows.append({
        "policy": f"naive top-{top_n} by predicted mean",
        "n_selected": len(naive),
        "true_mean_response": naive[truth_col].mean(),
        "true_worst_case": naive[truth_col].min(),
        "n_actually_harmful": int((naive[truth_col] < 0).sum()),
    })

    safe_pool = scored[scored["recommended"]]
    safe = safe_pool.nlargest(min(top_n, len(safe_pool)), "lcb")
    rows.append({
        "policy": f"safe rule, top-{top_n} by LCB",
        "n_selected": len(safe),
        "true_mean_response": safe[truth_col].mean() if len(safe) else np.nan,
        "true_worst_case": safe[truth_col].min() if len(safe) else np.nan,
        "n_actually_harmful": int((safe[truth_col] < 0).sum()) if len(safe) else 0,
    })

    return pd.DataFrame(rows)
