"""Tables 4 and 5 and Figure 1 (Section 4.2) from results/e1_rows.csv.

Intervals are the 2.5th and 97.5th percentiles across random draws: they show
how much the effect varies from one draw of the training sample to the next,
not a confidence interval for the mean effect.

  python analysis/e1_tables.py                                 # own run
  python analysis/e1_tables.py --rows results/paper/e1_rows.csv  # paper outputs
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from _path import FIGURES, RESULTS  # noqa: E402
from plot_style import MODEL_COLOURS, ORDER, apply_style  # noqa: E402

q = lambda s, p: np.percentile(s, p)  # noqa: E731


def main():
    ap = argparse.ArgumentParser()
    default = RESULTS / "e1_rows.csv"
    ap.add_argument("--rows", default=str(default if default.exists() else RESULTS / "paper" / "e1_rows.csv"))
    args = ap.parse_args()
    d = pd.read_csv(args.rows)
    print(f"reading {args.rows} ({len(d)} rows)")
    d["dauc"] = d.auc_fi - d.auc_past
    d["dap"] = d.ap_fi - d.ap_past

    t = (d.groupby(["design", "model", "window"])
         .agg(draws=("b", "size"), past_auc=("auc_past", "first"),
              dAUC=("dauc", "mean"), dAUC_lo=("dauc", lambda s: q(s, 2.5)), dAUC_hi=("dauc", lambda s: q(s, 97.5)),
              dAP=("dap", "mean"), dAP_lo=("dap", lambda s: q(s, 2.5)), dAP_hi=("dap", lambda s: q(s, 97.5)),
              fi_auc=("auc_fi", "mean"))
         .reset_index())
    t["model"] = pd.Categorical(t.model, ORDER, ordered=True)
    t = t.sort_values(["design", "model", "window"])
    t.round(4).to_csv(RESULTS / "e1_table.csv", index=False)

    t4 = t[t.design == "all_later"]
    print("\nTable 4 (future data from all later periods)\n",
          t4.drop(columns=["design", "fi_auc"]).round(3).to_string(index=False))

    # share of out-of-time error removed in t = 20-29 (from unrounded means)
    w = t4[t4.window == "t=20-29"]
    removed = (w.fi_auc - w.past_auc) / (1 - w.past_auc) * 100
    print("\nShare of out-of-time error (1 - AUC) removed in t=20-29:")
    for m, r in zip(w.model, removed):
        print(f"  {m}: {r:.1f}%")
    print("\nMean gain across windows (AUC, AP):")
    print(t4.groupby("model", observed=True)[["dAUC", "dAP"]].mean().round(3).to_string())

    adj = t[t.design == "adjacent"]
    if len(adj):
        t5 = adj.merge(t4[["model", "window", "dAUC", "dAP"]], on=["model", "window"],
                       suffixes=(" adjacent", " all later"))
        print("\nTable 5 (all later periods vs adjacent window only)\n",
              t5[["model", "window", "dAUC all later", "dAUC adjacent", "dAUC_lo", "dAUC_hi",
                  "dAP all later", "dAP adjacent", "dAP_lo", "dAP_hi"]].round(3).to_string(index=False))

    # Figure 1
    plt = apply_style()
    models = [m for m in ORDER if m in set(t4.model)]
    wins = sorted(t4.window.unique(), key=lambda s: int(s.split("=")[1].split("-")[0]))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    width = 0.8 / len(models)
    for j, m in enumerate(models):
        g = t4[t4.model == m].set_index("window").reindex(wins)
        x = np.arange(len(wins)) + (j - (len(models) - 1) / 2) * width
        ax.bar(x, g.dAUC, width * 0.9, color=MODEL_COLOURS[m], label=m)
        ax.errorbar(x, g.dAUC, yerr=[g.dAUC - g.dAUC_lo, g.dAUC_hi - g.dAUC], fmt="none",
                    ecolor="#0b0b0b", elinewidth=0.9, capsize=2)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xticks(np.arange(len(wins)), [w.replace("t=", "t = ").replace("-", " to ") for w in wins])
    ax.set_ylabel("Change in AUC (future-informed minus past-only)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "figure1_e1_future_information.png", dpi=300)
    print(f"\nsaved {FIGURES / 'figure1_e1_future_information.png'}")


if __name__ == "__main__":
    main()
