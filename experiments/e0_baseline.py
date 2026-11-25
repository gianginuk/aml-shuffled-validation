"""E0 (Section 4.1): shuffled five-fold cross-validation versus walk-forward validation.

Saves out-of-sample predictions for every model and seed to
results/e0_predictions/, from which analysis/e0_tables.py computes Tables 1-3,
including the paired bootstrap intervals.

Walk-forward: for each test window, train on all labelled transactions from
earlier time steps. Shuffled: stratified 5-fold CV with shuffling (split seed
42) over all 46,564 transactions (t = 1-49); scores are then computed only on
the transactions in t = 10-49, within each window, so that both designs are
evaluated on the same transactions.

Examples
  python experiments/e0_baseline.py --models Logistic "Random Forest" XGBoost
  python experiments/e0_baseline.py --models TabPFN            # GPU (Kaggle T4)
  python experiments/e0_baseline.py --models "TabPFN (10k cap)"  # one seed
"""
import argparse
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

from _path import RESULTS
from amlval.data import WINDOWS, load_elliptic
from amlval.models import SEEDS, fit_proba, seeds_for

CAP = 10_000          # training-set cap for the TabPFN sensitivity variant
SPLIT_SEED = 42


def capped(train_idx, y, seed, cap=CAP):
    """Stratified random sample of ``cap`` training rows (TabPFN 10k variant)."""
    if len(train_idx) <= cap:
        return train_idx
    rng = np.random.default_rng(seed)
    keep = []
    for lab in (0, 1):
        pool = train_idx[y[train_idx] == lab]
        n = int(round(cap * len(pool) / len(train_idx)))
        keep.append(rng.choice(pool, n, replace=False))
    return np.sort(np.concatenate(keep))


def run(name, X, y, ts, seed):
    base = "TabPFN" if name.startswith("TabPFN") else name
    cap = name == "TabPFN (10k cap)"
    all_idx = np.arange(len(y))
    eval_idx = all_idx[ts >= WINDOWS[0][0]]           # t = 10-49

    # walk-forward
    wf = np.full(len(y), np.nan)
    for lo, hi in WINDOWS:
        tr = all_idx[ts < lo]
        te = all_idx[(ts >= lo) & (ts < hi)]
        if cap:
            tr = capped(tr, y, seed)
        wf[te] = fit_proba(base, X, y, tr, te, seed)

    # shuffled five-fold CV over all transactions
    sh = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    for tr, te in skf.split(X, y):
        if cap:
            tr = capped(tr, y, seed)
        sh[te] = fit_proba(base, X, y, tr, te, seed)

    return eval_idx, wf[eval_idx], sh[eval_idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["Logistic", "Random Forest", "XGBoost"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    _, X, y, ts, _ = load_elliptic()
    out = RESULTS / "e0_predictions"
    out.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        seeds = [42] if name in ("Logistic", "TabPFN (10k cap)") else seeds_for(name, args.seeds)
        for seed in seeds:
            f = out / f"{name.replace(' ', '_')}_seed{seed}.npz"
            if f.exists():
                print(f"skip {f.name} (exists)")
                continue
            t0 = time.time()
            idx, wf, sh = run(name, X, y, ts, seed)
            np.savez_compressed(f, idx=idx, wf=wf, sh=sh, model=name, seed=seed)
            print(f"{name} seed {seed}: saved {f.name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
