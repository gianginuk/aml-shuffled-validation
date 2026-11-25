"""Tables 1, 2 and 3 (Section 4.1) from the E0 predictions.

Point estimates: each score is computed within each test window, averaged
across the four windows and then across seeds.

Paired bootstrap (Section 3.2): 1,000 resamples, transactions resampled within
each window. The same resample is applied to both designs, to every model and
to the predictions of every seed; metrics are averaged across seeds. Intervals
therefore reflect sampling variation in the test transactions, conditional on
the fitted models.

  python analysis/e0_tables.py               # writes results/e0_table.csv etc.
  python analysis/e0_tables.py --n-boot 200  # quicker
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from _path import RESULTS  # noqa: E402
from amlval.data import WINDOWS, load_elliptic, window_label  # noqa: E402
from amlval.metrics import max_found, top_share  # noqa: E402

ORDER = ["Logistic", "Random Forest", "XGBoost", "TabPFN", "TabPFN (10k cap)"]
BOOT_SEED = 2026   # as in the original E0 notebook


def fast_auc(y, p):
    """ROC AUC via the rank-sum formula (identical to sklearn, with ties averaged)."""
    r = pd.Series(p).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load_predictions(ts):
    preds = {}
    for f in sorted((RESULTS / "e0_predictions").glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        preds.setdefault(str(z["model"]), []).append((z["idx"], z["wf"], z["sh"]))
    if not preds:
        sys.exit("No predictions found: run experiments/e0_baseline.py first.")
    idx = next(iter(preds.values()))[0][0]
    wins = [np.where((ts[idx] >= lo) & (ts[idx] < hi))[0] for lo, hi in WINDOWS]
    return {m: preds[m] for m in ORDER if m in preds}, idx, wins


def scores(y_eval, runs, wins, rows=None):
    """Mean over windows, then seeds, of AUC and AP for both designs."""
    out = np.zeros((len(runs), len(wins), 4))
    for s, (_, wf, sh) in enumerate(runs):
        for w, ix in enumerate(wins):
            ii = ix if rows is None else rows[w]
            yy = y_eval[ii]
            out[s, w] = [fast_auc(yy, sh[ii]), fast_auc(yy, wf[ii]),
                         average_precision_score(yy, sh[ii]), average_precision_score(yy, wf[ii])]
    return out  # seeds x windows x (sh_auc, wf_auc, sh_ap, wf_ap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    _, _, y, ts, _ = load_elliptic()
    preds, idx, wins = load_predictions(ts)
    y_eval = y[idx]

    # ---- point estimates (Table 1 and Table 2)
    t1, t2 = [], []
    point = {}
    for m, runs in preds.items():
        s = scores(y_eval, runs, wins)
        sh_auc, wf_auc, sh_ap, wf_ap = s.mean(axis=(0, 1))
        point[m] = (sh_auc, wf_auc, sh_ap, wf_ap)
        seed_sd = s[:, :, 1].mean(axis=1).std(ddof=1) if len(runs) > 1 else np.nan
        t1.append({"Model": m, "Seeds": len(runs), "Shuffled AUC": sh_auc, "Walk-forward AUC": wf_auc,
                   "WF AUC seed SD": seed_sd, "AUC gap": sh_auc - wf_auc,
                   "Shuffled AP": sh_ap, "Walk-forward AP": wf_ap, "AP gap": sh_ap - wf_ap,
                   "Error ratio": (1 - wf_auc) / (1 - sh_auc)})
        for w, (lo, hi) in enumerate(WINDOWS):
            per_seed = s[:, w, 1]
            t2.append({"Model": m, "Window": window_label(lo, hi), "AUC mean": per_seed.mean(),
                       "AUC SD": per_seed.std(ddof=1) if len(runs) > 1 else np.nan})

    # ---- paired bootstrap
    rng = np.random.default_rng(BOOT_SEED)
    boot = {m: [] for m in preds}
    for b in range(args.n_boot):
        rows = [rng.choice(ix, len(ix), replace=True) for ix in wins]
        for m, runs in preds.items():
            s = scores(y_eval, runs, wins, rows).mean(axis=(0, 1))
            boot[m].append(s)
        if (b + 1) % 100 == 0:
            print(f"bootstrap {b + 1}/{args.n_boot}", flush=True)
    boot = {m: np.array(v) for m, v in boot.items()}
    ci = lambda v: f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"  # noqa: E731
    ratio = {m: (1 - v[:, 1]) / (1 - v[:, 0]) for m, v in boot.items()}
    for row in t1:
        v = boot[row["Model"]]
        row["AUC gap 95% CI"] = ci(v[:, 0] - v[:, 1])
        row["AP gap 95% CI"] = ci(v[:, 2] - v[:, 3])
        row["Error ratio 95% CI"] = ci(ratio[row["Model"]])
    t1 = pd.DataFrame(t1)[["Model", "Seeds", "Shuffled AUC", "Walk-forward AUC", "WF AUC seed SD",
                           "AUC gap", "AUC gap 95% CI", "Shuffled AP", "Walk-forward AP", "AP gap",
                           "AP gap 95% CI", "Error ratio", "Error ratio 95% CI"]]
    t1.round(4).to_csv(RESULTS / "e0_table.csv", index=False)
    pd.DataFrame(t2).round(4).to_csv(RESULTS / "e0_by_window.csv", index=False)
    print("\nTable 1\n", t1.round(4).to_string(index=False))
    print("\nTable 2\n", pd.DataFrame(t2).pivot(index="Model", columns="Window", values="AUC mean")
          .reindex([m for m in ORDER if m in preds]).round(3).to_string())

    # difference in error ratio from logistic regression (Section 4.1 text)
    if "Logistic" in ratio:
        print("\nError-ratio difference from logistic regression [95% CI]:")
        base = point["Logistic"]
        for m in preds:
            if m == "Logistic":
                continue
            pm = point[m]
            d = (1 - pm[1]) / (1 - pm[0]) - (1 - base[1]) / (1 - base[0])
            print(f"  {m}: {d:+.1f} {ci(ratio[m] - ratio['Logistic'])}")

    # ---- Table 3: screening by score (classical models)
    t3 = []
    for m, runs in preds.items():
        if m.startswith("TabPFN"):
            continue
        acc = []
        for _, wf, sh in runs:
            per_w = []
            for ix in wins:
                yy = y_eval[ix]
                f1s, _ = top_share(yy, sh[ix], 0.01)
                f1w, _ = top_share(yy, wf[ix], 0.01)
                f5s, p5s = top_share(yy, sh[ix], 0.05)
                f5w, p5w = top_share(yy, wf[ix], 0.05)
                per_w.append([f1s, f1w, f5s, f5w, p5s, p5w])
            acc.append(per_w)
        acc = np.array(acc).mean(axis=0)  # windows x measures, mean over seeds
        mean = acc.mean(axis=0)
        t3.append({"Model": m, "Found at 1%: shuffled": mean[0], "Found at 1%: walk-forward": mean[1],
                   "Found at 5%: shuffled": mean[2], "Found at 5%: walk-forward": mean[3],
                   "Precision at 5%: shuffled": mean[4], "Precision at 5%: walk-forward": mean[5],
                   "Found at 5%, t=40-49: shuffled": acc[-1, 2], "Found at 5%, t=40-49: walk-forward": acc[-1, 3]})
    t3 = pd.DataFrame(t3)
    mx = [np.mean([max_found(y_eval[ix], s) for ix in wins]) for s in (0.01, 0.05)]
    t3.round(3).to_csv(RESULTS / "e0_screening.csv", index=False)
    print("\nTable 3 (max attainable found at 1% and 5%: "
          f"{mx[0]:.3f}, {mx[1]:.3f})\n", t3.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
