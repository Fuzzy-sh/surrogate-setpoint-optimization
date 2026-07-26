"""End-to-end run: generate data, validate, fit, score candidates, write figures.

    python run_analysis.py

Produces `figures/*.png` and prints the metrics quoted in the README.
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import features as F
from src import generate as G
from src import model as M
from src import recommend as R

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

INK = "#2b3a55"
ACCENT = "#c2643f"
MUTED = "#8896ab"


def main() -> None:
    # ---------------------------------------------------------------- data ----
    hist = G.make_history()
    cand = G.make_candidates()
    hist.to_csv(DATA / "history.csv", index=False)
    cand.to_csv(DATA / "candidates.csv", index=False)

    y = hist[F.TARGET].to_numpy()
    groups = hist[F.GROUP].to_numpy()
    w = F.sample_weights(hist)
    X = F.build_features(G.drop_oracle(hist))

    print(f"history {hist.shape}  candidates {cand.shape}")
    print(f"target: mean {y.mean():+.3f}  sd {y.std():.3f}")

    # Noise ceiling: how much variance is explainable even with the true function.
    truth = hist["true_mean_response"].to_numpy()
    ceiling = 1 - np.mean((y - truth) ** 2) / np.var(y)
    print(f"oracle R2 ceiling (irreducible noise floor): {ceiling:.3f}")

    # ---------------------------------------------------- validation design ----
    print("\n--- validation ---")

    # 1. Naive random k-fold, to demonstrate the leak it causes.
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.ensemble import HistGradientBoostingRegressor

    naive_pred = cross_val_predict(
        HistGradientBoostingRegressor(max_depth=4, max_iter=320, learning_rate=0.06,
                                      min_samples_leaf=28, random_state=0),
        X.to_numpy(dtype=float), y, cv=KFold(5, shuffle=True, random_state=0),
    )
    naive_m = M.regression_metrics(y, naive_pred)

    # 2. Grouped by asset -- the honest number.
    oof = M.grouped_cv_predict(X, y, w, groups, n_splits=5)
    grouped_m = M.regression_metrics(y, oof["pred_mean"].to_numpy())

    # 3. Forward in time.
    tr, te = M.temporal_holdout(hist, frac=0.25)
    tm = M.SurrogateModel(n_ensemble=6).fit(X[tr], y[tr], w[tr], groups[tr])
    temporal_m = M.regression_metrics(y[te], tm.predict(X[te]))

    val = pd.DataFrame([
        {"scheme": "random k-fold (leaky)", **naive_m},
        {"scheme": "grouped by asset", **grouped_m},
        {"scheme": "forward in time", **temporal_m},
    ])
    print(val.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    val.to_csv(DATA / "validation_metrics.csv", index=False)

    cov = M.interval_coverage(y, oof["pred_lo"].to_numpy(), oof["pred_hi"].to_numpy())
    print(f"\n80% interval coverage (out-of-fold): {cov['coverage']:.3f} "
          f"(nominal 0.800), mean width {cov['mean_width']:.2f}")

    # --------------------------------------------------------------- model ----
    full = M.SurrogateModel(n_ensemble=8).fit(X, y, w, groups)

    # --------------------------------------------------------- diagnostics ----
    resid = oof["pred_mean"].to_numpy() - y

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))
    ax[0].scatter(oof["pred_mean"], y, s=7, alpha=0.35, color=INK, edgecolor="none")
    lim = [min(y.min(), oof["pred_mean"].min()), max(y.max(), oof["pred_mean"].max())]
    ax[0].plot(lim, lim, color=ACCENT, lw=1.2)
    ax[0].set_xlabel("predicted"); ax[0].set_ylabel("observed")
    ax[0].set_title(f"Out-of-fold fit (R²={grouped_m['r2']:.2f})")

    ax[1].scatter(oof["pred_mean"], resid, s=7, alpha=0.35, color=INK, edgecolor="none")
    ax[1].axhline(0, color=ACCENT, lw=1.2)
    ax[1].set_xlabel("predicted"); ax[1].set_ylabel("residual")
    ax[1].set_title("Residuals vs fitted")

    q = pd.qcut(hist["data_quality_score"], 5, duplicates="drop")
    by_q = pd.DataFrame({"q": q, "abs_err": np.abs(resid)}).groupby("q", observed=True)["abs_err"].mean()
    ax[2].bar(range(len(by_q)), by_q.to_numpy(), color=MUTED)
    ax[2].set_xticks(range(len(by_q)))
    ax[2].set_xticklabels([f"Q{i+1}" for i in range(len(by_q))])
    ax[2].set_xlabel("data-quality quintile (low → high)")
    ax[2].set_ylabel("mean |error|")
    ax[2].set_title("Error concentrates in poor data")
    fig.tight_layout(); fig.savefig(FIG / "diagnostics.png"); plt.close(fig)

    # Calibration of the predictive interval across nominal levels.
    levels = [0.5, 0.6, 0.7, 0.8, 0.9]
    emp = []
    for lv in levels:
        a = (1 - lv) / 2
        mtmp = M.SurrogateModel(n_ensemble=3, lower_q=a, upper_q=1 - a).fit(X, y, w, groups)
        lo, hi = mtmp.predict_interval(X)
        emp.append(float(((y >= lo) & (y <= hi)).mean()))

    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))
    ax[0].plot([0.4, 1.0], [0.4, 1.0], color=MUTED, ls="--", lw=1)
    ax[0].plot(levels, emp, "o-", color=ACCENT, lw=1.5)
    ax[0].set_xlabel("nominal coverage"); ax[0].set_ylabel("empirical coverage")
    ax[0].set_title("Interval calibration (in-sample)")

    # ------------------------------------------------------ score candidates --
    X_cand = F.build_features(G.drop_oracle(cand))
    preds = full.predict_full(X_cand)

    support = R.SupportChecker(k=12, quantile=0.98).fit(X)
    envelope = R.action_envelope(hist)

    # Which uncertainty signal actually predicts error where it matters -- on
    # candidates, not on in-distribution training rows? Reported honestly: the
    # ensemble adds little, the support distance does the work.
    cand_err = np.abs(preds["pred_mean"].to_numpy() - cand["true_mean_response"].to_numpy())
    sup_dist = support.distance(X_cand)
    c_epi_oof = float(np.corrcoef(oof["epistemic_sd"], np.abs(resid))[0, 1])
    c_epi_cand = float(np.corrcoef(preds["epistemic_sd"], cand_err)[0, 1])
    c_sup_cand = float(np.corrcoef(sup_dist, cand_err)[0, 1])

    ax[1].scatter(sup_dist, cand_err, s=10, alpha=0.35, color=INK, edgecolor="none")
    ax[1].axvline(support.threshold_, color=ACCENT, lw=1.2, ls="--", label="support threshold")
    ax[1].set_xlabel("distance from training support")
    ax[1].set_ylabel("|error| on candidate actions")
    ax[1].set_title(f"Support distance predicts error (r={c_sup_cand:.2f})")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "uncertainty.png"); plt.close(fig)

    print("\nwhich uncertainty signal is informative?")
    print(f"   corr(epistemic sd, |error|)  in-distribution : {c_epi_oof:6.3f}")
    print(f"   corr(epistemic sd, |error|)  on candidates   : {c_epi_cand:6.3f}")
    print(f"   corr(support distance, |error|) on candidates: {c_sup_cand:6.3f}")
    print("   -> ensemble disagreement adds little; the k-NN support check carries the signal.")
    scored = R.score_candidates(cand, X_cand, preds, support, envelope, kappa=1.0)
    scored.to_csv(DATA / "scored_candidates.csv", index=False)

    n_rec = int(scored["recommended"].sum())
    print(f"\ncandidates recommended: {n_rec}/{len(scored)} "
          f"({100*n_rec/len(scored):.1f}%)")
    print("rejection reasons:")
    rej = scored.loc[~scored["recommended"], "reject_reason"]
    for reason, c in rej.str.split("; ").explode().value_counts().items():
        print(f"   {c:4d}  {reason}")

    comparison = R.compare_policies(scored, top_n=20)
    print("\n--- policy comparison (judged against the true response) ---")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    comparison.to_csv(DATA / "policy_comparison.csv", index=False)

    # Figure: where the naive policy goes wrong.
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.5))
    ok = scored["recommended"].to_numpy()
    ax[0].scatter(scored.loc[~ok, "pred_mean"], scored.loc[~ok, "true_mean_response"],
                  s=12, alpha=0.45, color=MUTED, edgecolor="none", label="rejected")
    ax[0].scatter(scored.loc[ok, "pred_mean"], scored.loc[ok, "true_mean_response"],
                  s=14, alpha=0.75, color=ACCENT, edgecolor="none", label="recommended")
    lim = [scored["pred_mean"].min(), scored["pred_mean"].max()]
    ax[0].plot(lim, lim, color=INK, lw=1, ls="--")
    ax[0].axhline(0, color=INK, lw=0.8)
    ax[0].set_xlabel("predicted mean"); ax[0].set_ylabel("true mean response")
    ax[0].set_title("Safety rule filters the tail"); ax[0].legend(frameon=False, fontsize=8)

    ax[1].scatter(scored["support_distance"], scored["pred_mean"] - scored["true_mean_response"],
                  s=12, alpha=0.45, color=INK, edgecolor="none")
    ax[1].axvline(support.threshold_, color=ACCENT, lw=1.2, ls="--", label="support threshold")
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].set_xlabel("distance from training support")
    ax[1].set_ylabel("prediction error on candidates")
    ax[1].set_title("Error grows outside support"); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "policy.png"); plt.close(fig)

    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
