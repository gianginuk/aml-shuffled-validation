"""E2 (Section 4.3): label delay with a random-removal control arm.

For each walk-forward test window and delay k (two-week periods):
  full    : train on all labelled transactions before the window (k = 0)
  delay   : drop the k most recent training periods
  control : drop the SAME number of transactions per class, at random from the
            whole past (repeat r uses random generator r and model seed r)

  data loss    = AUC(full) - AUC(control)
  recency loss = AUC(control) - AUC(delay)
  total loss   = AUC(full) - AUC(delay)

k = 6 is not run for t = 10-19 (fewer than five training periods would remain).
Progress is saved to results/e2_rows.csv after every (model, window, k) step;
finished steps are skipped on a rerun, so an interrupted Kaggle run can resume.

Paper settings
  classical models : 50 control repeats; Random Forest and XGBoost full and
                     delay arms over seeds 42, 1, 7, 21, 99; logistic once
  TabPFN           : 10 control repeats, one seed (42), GPU

  python experiments/e2_label_delay.py --models Logistic "Random Forest" XGBoost
  python experiments/e2_label_delay.py --models TabPFN
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _path import RESULTS
from amlval.data import BOUNDARIES, load_elliptic
from amlval.metrics import auc_ap
from amlval.models import SEEDS, fit_proba

KS = [1, 2, 4, 6]
REPEATS = {"Logistic": 50, "Random Forest": 50, "XGBoost": 50, "TabPFN": 10}
MIN_PERIODS = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["Logistic", "Random Forest", "XGBoost"])
    ap.add_argument("--repeats", type=int, default=None, help="override control repeats for all models")
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    ap.add_argument("--out", default=str(RESULTS / "e2_rows.csv"))
    args = ap.parse_args()

    _, X, y, ts, _ = load_elliptic()
    out = Path(args.out)
    done = set()
    if out.exists():
        d = pd.read_csv(out)
        done = set(zip(d.model, d.window, d.k))

    def save(rows):
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)

    all_idx = np.arange(len(y))
    for name in args.models:
        stochastic = name != "Logistic"
        seeds = [42] if name in ("Logistic", "TabPFN") else SEEDS
        n_rep = args.repeats or REPEATS[name]
        for i in range(len(BOUNDARIES) - 2):
            cut, hi = BOUNDARIES[i + 1], BOUNDARIES[i + 2]
            win = f"t={cut}-{hi - 1}"
            test_mask = (ts >= cut) & (ts < hi)
            test = all_idx[test_mask]
            past = ts < cut

            def score(rows, arm, k, rep, seed, tr_idx):
                p = fit_proba(name, X, y, tr_idx, test, seed)
                auc, apv = auc_ap(y[test], p)
                rows.append({"model": name, "window": win, "arm": arm, "k": k, "rep": rep, "seed": seed,
                             "train_n": len(tr_idx), "train_illicit": int(y[tr_idx].sum()),
                             "auc": auc, "ap": apv})

            if (name, win, 0) not in done:
                t0 = time.time(); rows = []
                for s in seeds:
                    score(rows, "full", 0, 0, s, all_idx[past])
                save(rows); done.add((name, win, 0))
                print(f"{name} {win} k=0 done ({time.time() - t0:.0f}s)", flush=True)

            for k in args.ks:
                if cut - k < MIN_PERIODS or (name, win, k) in done:
                    continue
                t0 = time.time(); rows = []
                for s in seeds:
                    score(rows, "delay", k, 0, s, all_idx[past & (ts < cut - k)])
                lost = past & (ts >= cut - k)
                for r in range(n_rep):
                    rng = np.random.default_rng(r)
                    keep = past.copy()
                    for lab in (0, 1):
                        pool = np.where(past & (y == lab))[0]
                        n = int((lost & (y == lab)).sum())
                        keep[rng.choice(pool, n, replace=False)] = False
                    score(rows, "control", k, r, r if stochastic else 42, all_idx[keep])
                save(rows); done.add((name, win, k))
                print(f"{name} {win} k={k} done ({time.time() - t0:.0f}s)", flush=True)
    print("E2 done")


if __name__ == "__main__":
    main()
