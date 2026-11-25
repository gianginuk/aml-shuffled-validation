"""E3 (Section 4.4): graph degree features and leakage (Random Forest, five seeds).

Degree constructions added to the 165 transaction features (in-, out- and
total degree):
  A  : labelled-only edges (36,624 edges; as in the dissertation)
  B  : full network incl. unlabelled neighbours (234,355 edges); needs Kaggle's
       elliptic_txs_edgelist.csv in data/ (skipped otherwise)
  C  : edges whose BOTH endpoints precede the training cutoff (rebuilt per
       window); invalid by design, every test transaction gets zero degree
  C2 : correctly time-bounded control, edges observable by the end of the test
       window; equals A exactly because every labelled edge joins two
       transactions in the same time step

Outputs: results/e3_results_long.csv, results/e3_table.csv (Table 8) and
results/e3_degree_by_class.csv (Table 9). About 10 minutes on a 2-core CPU
(A, C and C2), longer with B.

  python experiments/e3_graph_degree.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from _path import RESULTS
from amlval.data import DATA_DIR, WINDOWS, load_elliptic, load_labelled_edges, window_label
from amlval.metrics import auc_ap
from amlval.models import SEEDS

KAGGLE_EDGES = Path(DATA_DIR) / "elliptic_txs_edgelist.csv"


def make_rf(seed):
    return RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                                  random_state=seed, n_jobs=-1)


def main():
    ell, X, y, ts, _ = load_elliptic()
    edges = load_labelled_edges()
    print(f"Transactions {len(y):,}, illicit {y.sum():,}, labelled-labelled edges {len(edges):,}")

    tmap = dict(zip(ell["txId"], ell["time_step"]))
    e = edges.copy()
    e["t1"] = e["txId1"].map(tmap)
    e["t2"] = e["txId2"].map(tmap)
    same_step = (e["t1"] == e["t2"]).mean()
    print(f"Share of labelled edges joining two transactions in the same time step: {same_step:.4f}")

    def degrees(edge_df):
        ind = edge_df["txId2"].value_counts()
        outd = edge_df["txId1"].value_counts()
        return ell["txId"].map(ind).fillna(0).values, ell["txId"].map(outd).fillna(0).values

    def stack(i, o):
        return np.hstack([X, i.reshape(-1, 1), o.reshape(-1, 1), (i + o).reshape(-1, 1)])

    iA, oA = degrees(edges)
    XA = stack(iA, oA)
    e_t = e.dropna(subset=["t1", "t2"]).copy()
    e_t["edge_time"] = e_t[["t1", "t2"]].max(axis=1)
    X_C = lambda cut: stack(*degrees(e_t[e_t["edge_time"] < cut]))   # noqa: E731
    X_C2 = lambda hi: stack(*degrees(e_t[e_t["edge_time"] < hi]))    # noqa: E731

    XB, iB, oB = None, None, None
    if KAGGLE_EDGES.exists():
        full = pd.read_csv(KAGGLE_EDGES)
        full.columns = ["txId1", "txId2"]
        assert len(full) == 234_355, "unexpected edge count, wrong file?"
        iB, oB = degrees(full)
        XB = stack(iB, oB)
        print(f"Approach B: full edge list loaded, {len(full):,} edges")
    else:
        print(f"Approach B skipped: put Kaggle's elliptic_txs_edgelist.csv in {DATA_DIR}")

    # zero-degree diagnostic (why C is not a valid test)
    zd = []
    for lo, hi in WINDOWS:
        deg_c = X_C(lo)[:, -1]
        tr, te = ts < lo, (ts >= lo) & (ts < hi)
        zd.append([window_label(lo, hi), (deg_c[tr] == 0).mean(), (deg_c[te] == 0).mean(),
                   (XA[tr, -1] == 0).mean(), (XA[te, -1] == 0).mean()])
    print(pd.DataFrame(zd, columns=["window", "C train zero-deg", "C test zero-deg",
                                    "A train zero-deg", "A test zero-deg"]).round(3).to_string(index=False))

    out, t0 = [], time.time()
    for seed in SEEDS:
        for lo, hi in WINDOWS:
            tr, te = ts < lo, (ts >= lo) & (ts < hi)
            variants = {"base": X, "A": XA, "C": X_C(lo), "C2": X_C2(hi)}
            if XB is not None:
                variants["B"] = XB
            for name, M in variants.items():
                p = make_rf(seed).fit(M[tr], y[tr]).predict_proba(M[te])[:, 1]
                auc, apv = auc_ap(y[te], p)
                out.append({"seed": seed, "window": window_label(lo, hi), "variant": name, "AUC": auc, "AP": apv})
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(out)
    res.to_csv(RESULTS / "e3_results_long.csv", index=False)

    per_seed = res.groupby(["seed", "variant"])[["AUC", "AP"]].mean().reset_index()
    base = per_seed[per_seed.variant == "base"].set_index("seed")
    rows = []
    for v in [c for c in ["base", "A", "B", "C", "C2"] if c in set(res.variant)]:
        d = per_seed[per_seed.variant == v].set_index("seed")
        dA, dP = d["AUC"] - base["AUC"], d["AP"] - base["AP"]
        rows.append([v, d["AUC"].mean(), dA.mean(), dA.std(), dP.mean(), dP.std()])
    summ = pd.DataFrame(rows, columns=["variant", "mean AUC", "dAUC vs base", "sd dAUC", "dAP vs base", "sd dAP"])
    summ.round(4).to_csv(RESULTS / "e3_table.csv", index=False)
    print("\nTable 8\n", summ.round(4).to_string(index=False))

    # Table 9: degree by class
    t9 = []
    for label, (i, o) in [("A (labelled-only edges)", (iA, oA)), ("B (full network)", (iB, oB))]:
        if i is None:
            continue
        d = pd.DataFrame({"illicit": y, "in": i, "out": o, "total": i + o})
        g = d.groupby("illicit")[["in", "out", "total"]].mean()
        g["share zero-degree"] = d.groupby("illicit")["total"].apply(lambda s: (s == 0).mean())
        g.insert(0, "construction", label)
        t9.append(g.reset_index())
    t9 = pd.concat(t9)
    t9.round(3).to_csv(RESULTS / "e3_degree_by_class.csv", index=False)
    print("\nTable 9\n", t9.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
