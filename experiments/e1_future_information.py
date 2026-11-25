"""E1 (Section 4.2): controlled comparison isolating the effect of future information.

For each test window, a past-only model is trained on all transactions before
the window. A future-informed model is trained on a set of identical size and
class composition, drawn at random (per class, without replacement) from the
past and from later periods outside the test window. The difference in
test-window AUC and AP isolates the effect of future information.

Designs
  all_later : future data from every later period (t >= end of test window)
  adjacent  : future data from the adjacent window only (t = hi .. hi+9);
              not run for t = 30-39 (the adjacent window is all the later data)

The final window (t = 40-49) is omitted because no later data exist. For the
stochastic models the past-only score is averaged over the five seeds of E0,
and draw b uses random generator 1000 + b and model seed b.

Output: results/e1_rows.csv (one row per model, design, window and draw),
written after every (model, design, window) step; finished steps are skipped
on a rerun.

  python experiments/e1_future_information.py --models Logistic "Random Forest" XGBoost
  python experiments/e1_future_information.py --models TabPFN --draws 25 --designs all_later
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _path import RESULTS
from amlval.data import WINDOWS, load_elliptic, window_label
from amlval.metrics import auc_ap
from amlval.models import SEEDS, fit_proba, seeds_for


def future_pool(ts, lo, hi, design):
    if design == "all_later":
        return ts >= hi
    if design == "adjacent":
        return (ts >= hi) & (ts < hi + (hi - lo))
    raise ValueError(design)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["Logistic", "Random Forest", "XGBoost"])
    ap.add_argument("--designs", nargs="+", default=["all_later", "adjacent"])
    ap.add_argument("--draws", type=int, default=100, help="random draws per cell (paper: 100; 25 for TabPFN)")
    ap.add_argument("--windows", nargs="+", default=None, help="e.g. t=20-29 (default: t=10-19 to t=30-39)")
    ap.add_argument("--out", default=str(RESULTS / "e1_rows.csv"))
    args = ap.parse_args()

    _, X, y, ts, _ = load_elliptic()
    out = Path(args.out)
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev.model, prev.design, prev.window))

    all_idx = np.arange(len(y))
    for name in args.models:
        for design in args.designs:
            for lo, hi in WINDOWS[:-1]:
                win = window_label(lo, hi)
                if args.windows and win not in args.windows:
                    continue
                if design == "adjacent" and hi == WINDOWS[-1][0]:
                    continue
                if (name, design, win) in done:
                    print(f"skip {name} {design} {win}")
                    continue
                t0 = time.time()
                past = ts < lo
                test = all_idx[(ts >= lo) & (ts < hi)]
                later = future_pool(ts, lo, hi, design)
                pool = past | later

                # past-only baseline, mean over seeds
                pa = [auc_ap(y[test], fit_proba(name, X, y, all_idx[past], test, s))
                      for s in seeds_for(name, SEEDS)]
                auc_past, ap_past = np.mean(pa, axis=0)

                rows = []
                for b in range(args.draws):
                    # same draws as the original runs: generator 1000 + b, illicit then licit,
                    # row order kept as drawn (Random Forest's bootstrap depends on row order)
                    rng = np.random.default_rng(1000 + b)
                    pool_idx = all_idx[pool]
                    ill, lic = pool_idx[y[pool_idx] == 1], pool_idx[y[pool_idx] == 0]
                    tr = np.concatenate([rng.choice(ill, int((past & (y == 1)).sum()), replace=False),
                                         rng.choice(lic, int((past & (y == 0)).sum()), replace=False)])
                    auc_fi, ap_fi = auc_ap(y[test], fit_proba(name, X, y, tr, test, seed=b))
                    rows.append({"model": name, "design": design, "window": win, "b": b,
                                 "auc_past": auc_past, "ap_past": ap_past, "auc_fi": auc_fi, "ap_fi": ap_fi,
                                 "share_later": later[tr].mean(), "past_n": int(past.sum()),
                                 "later_pool_n": int(later.sum())})
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                done.add((name, design, win))
                d = np.mean([r["auc_fi"] for r in rows]) - auc_past
                print(f"{name} {design} {win}: mean dAUC {d:+.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
