"""Surrogate model with calibrated predictive uncertainty.

Design choice
-------------
Two uncertainty sources matter here and they are not the same thing:

* **Aleatoric** -- irreducible measurement noise, which varies with data quality.
  Estimated with gradient-boosted *quantile* regressors, which model the spread of
  the response directly.
* **Epistemic** -- model ignorance in regions with little training support. Estimated
  from the disagreement of a bagged ensemble fitted on different asset subsets.

A recommender that only sees aleatoric noise will happily extrapolate into action
ranges nobody has ever tried. Keeping the two separate is what makes the safety
rule in `recommend.py` meaningful.

What actually worked
--------------------
Reported honestly, because the result was not the one expected: **the bagged
ensemble turned out to be a weak signal.** Its disagreement barely correlates with
error in-distribution (r ~ 0.03) and only mildly on out-of-distribution candidates
(r ~ 0.16). The k-NN support check in `recommend.py` is what carries the signal
(r ~ 0.71 against candidate error).

The ensemble is kept because it is cheap and contributes to the total-variance term,
but the honest conclusion is that a distance-to-training-data check did the work that
ensemble disagreement is usually assumed to do. Quantile heads plus a conformal
correction handle the aleatoric side well: 80% intervals achieve ~0.80 coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold


@dataclass
class SurrogateModel:
    """Mean model + quantile heads + bagged ensemble for epistemic spread."""

    n_ensemble: int = 8
    lower_q: float = 0.1
    upper_q: float = 0.9
    random_state: int = 0

    mean_model: HistGradientBoostingRegressor | None = None
    q_lo: GradientBoostingRegressor | None = None
    q_hi: GradientBoostingRegressor | None = None
    ensemble: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)
    calibration_scale_: float = 1.0
    _train_X: np.ndarray | None = None

    # -- fitting ----------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: np.ndarray, w: np.ndarray, groups: np.ndarray):
        self.feature_names = list(X.columns)
        Xv = X.to_numpy(dtype=float)
        self._train_X = Xv

        self.mean_model = HistGradientBoostingRegressor(
            max_depth=4, max_iter=320, learning_rate=0.06,
            min_samples_leaf=28, l2_regularization=1.0,
            random_state=self.random_state,
        ).fit(Xv, y, sample_weight=w)

        common = dict(
            n_estimators=280, max_depth=3, learning_rate=0.05,
            min_samples_leaf=28, subsample=0.85, random_state=self.random_state,
        )
        self.q_lo = GradientBoostingRegressor(
            loss="quantile", alpha=self.lower_q, **common
        ).fit(Xv, y, sample_weight=w)
        self.q_hi = GradientBoostingRegressor(
            loss="quantile", alpha=self.upper_q, **common
        ).fit(Xv, y, sample_weight=w)

        # Bagged ensemble over *asset groups*, not rows -- resampling rows would
        # leak the same asset into every member and understate disagreement.
        rng = np.random.default_rng(self.random_state)
        uniq = np.unique(groups)
        self.ensemble = []
        for k in range(self.n_ensemble):
            keep = rng.choice(uniq, size=int(0.75 * len(uniq)), replace=False)
            m = np.isin(groups, keep)
            est = HistGradientBoostingRegressor(
                max_depth=4, max_iter=220, learning_rate=0.07,
                min_samples_leaf=28, l2_regularization=1.0, random_state=k,
            ).fit(Xv[m], y[m], sample_weight=w[m])
            self.ensemble.append(est)
        return self

    # -- prediction -------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.mean_model.predict(X.to_numpy(dtype=float))

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xv = X.to_numpy(dtype=float)
        lo = self.q_lo.predict(Xv)
        hi = self.q_hi.predict(Xv)
        # Quantile heads are fitted independently and can cross on sparse regions.
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        if self.calibration_scale_ != 1.0:
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * self.calibration_scale_
            lo, hi = mid - half, mid + half
        return lo, hi

    def calibrate_intervals(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Rescale interval width on held-out data to hit nominal coverage.

        Gradient-boosted quantile heads are systematically over-tight: they are
        fitted on the training set and inherit its optimism. Rather than pretend
        the raw intervals are calibrated, we measure the miss on data the heads did
        not see and apply a single multiplicative correction -- a simple
        split-conformal adjustment. One scalar, no distributional assumptions.
        """
        self.calibration_scale_ = 1.0                      # measure uncorrected
        lo, hi = self.predict_interval(X)
        nominal = self.upper_q - self.lower_q
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        # Smallest scale s such that |y - mid| <= s * half for `nominal` of points.
        ratio = np.abs(y - mid) / np.maximum(half, 1e-9)
        self.calibration_scale_ = float(np.quantile(ratio, nominal))
        return self.calibration_scale_

    def epistemic_sd(self, X: pd.DataFrame) -> np.ndarray:
        Xv = X.to_numpy(dtype=float)
        preds = np.stack([m.predict(Xv) for m in self.ensemble])
        return preds.std(axis=0)

    def predict_full(self, X: pd.DataFrame) -> pd.DataFrame:
        mean = self.predict(X)
        lo, hi = self.predict_interval(X)
        epi = self.epistemic_sd(X)
        return pd.DataFrame(
            {
                "pred_mean": mean,
                "pred_lo": lo,
                "pred_hi": hi,
                "interval_width": hi - lo,
                "epistemic_sd": epi,
            },
            index=X.index,
        )


# -- validation -----------------------------------------------------------------


def grouped_cv_predict(
    X: pd.DataFrame, y: np.ndarray, w: np.ndarray, groups: np.ndarray, n_splits: int = 5
) -> pd.DataFrame:
    """Out-of-fold predictions with assets held out entirely.

    Random k-fold would put the same asset in train and test. Because assets have
    persistent latent characteristics, that inflates every metric -- the model
    recognises the asset rather than learning the response function. Grouping by
    asset is the only validation that answers the question we actually care about:
    *will this work on an asset we have not modelled before?*
    """
    oof_mean = np.full(len(y), np.nan)
    oof_lo = np.full(len(y), np.nan)
    oof_hi = np.full(len(y), np.nan)
    oof_epi = np.full(len(y), np.nan)

    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups=groups):
        # Hold out a calibration slice *by asset* so the conformal correction is
        # measured on assets the quantile heads never saw.
        tr_groups = np.unique(groups[tr])
        rng = np.random.default_rng(0)
        cal_assets = rng.choice(tr_groups, size=max(2, int(0.2 * len(tr_groups))), replace=False)
        cal_mask = np.isin(groups[tr], cal_assets)
        fit_idx, cal_idx = tr[~cal_mask], tr[cal_mask]

        m = SurrogateModel(n_ensemble=5).fit(
            X.iloc[fit_idx], y[fit_idx], w[fit_idx], groups[fit_idx]
        )
        m.calibrate_intervals(X.iloc[cal_idx], y[cal_idx])
        out = m.predict_full(X.iloc[te])
        oof_mean[te] = out["pred_mean"].to_numpy()
        oof_lo[te] = out["pred_lo"].to_numpy()
        oof_hi[te] = out["pred_hi"].to_numpy()
        oof_epi[te] = out["epistemic_sd"].to_numpy()

    return pd.DataFrame(
        {
            "pred_mean": oof_mean,
            "pred_lo": oof_lo,
            "pred_hi": oof_hi,
            "epistemic_sd": oof_epi,
        },
        index=X.index,
    )


def temporal_holdout(df: pd.DataFrame, frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for a forward-in-time split.

    Grouped CV answers 'does this generalise to a new asset'. A time split answers
    'does this still hold next quarter'. Operating conditions and operator habits
    drift, so both questions need answering and they can disagree.
    """
    order = np.argsort(df["event_time"].to_numpy())
    cut = int(len(df) * (1 - frac))
    train_idx = order[:cut]
    test_idx = order[cut:]
    tr = np.zeros(len(df), dtype=bool)
    te = np.zeros(len(df), dtype=bool)
    tr[train_idx] = True
    te[test_idx] = True
    return tr, te


# -- metrics --------------------------------------------------------------------


def regression_metrics(y: np.ndarray, pred: np.ndarray, w: np.ndarray | None = None) -> dict:
    err = pred - y
    mae = np.average(np.abs(err), weights=w)
    rmse = float(np.sqrt(np.average(err**2, weights=w)))
    ss_res = np.average(err**2, weights=w)
    ss_tot = np.average((y - np.average(y, weights=w)) ** 2, weights=w)
    return {"mae": float(mae), "rmse": rmse, "r2": float(1 - ss_res / ss_tot)}


def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    inside = (y >= lo) & (y <= hi)
    return {
        "coverage": float(inside.mean()),
        "mean_width": float(np.mean(hi - lo)),
    }
