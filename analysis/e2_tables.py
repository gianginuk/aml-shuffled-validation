"""Tables 6, 7 and S1 and Figures 2 and 3 (Section 4.3) from results/e2_rows.csv.

Ranges in Table 6: losses are computed per control repeat, averaged over the
windows t = 20-49, then summarised by the 2.5th and 97.5th percentiles across
repeats for the classical models, and by the minimum and maximum across the
10 repeats for TabPFN. Window t = 10-19 is excluded from the averages because
k = 6 cannot be run there.

  python analysis/e2_tables.py                                 # own run
  python analysis/e2_tables.py --rows results/paper/e2_rows.csv  # paper outputs
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from _path import FIGURES, RESULTS  # noqa: E402
from plot_style import AXIS, GRID, INK, ORDER, apply_style  # noqa: E402

AVG_WINDOWS = ["t=20-29", "t=30-39", "t=40-49"]
wnum = lambda s: int(s.split("=")[1].split("-")[0])  # noqa: E731


def main():
    ap = argparse.ArgumentParser()
    default = RESULTS / "e2_rows.csv"
    ap.add_argument("--rows", default=str(default if default.exists() else RESULTS / "paper" / "e2_rows.csv"))
    args = ap.parse_args()
    e2 = pd.read_csv(args.rows)
    print(f"reading {args.rows} ({len(e2)} rows)")
    models = [m for m in ORDER if m in set(e2.model)]

    full = e2[e2.arm == "full"].groupby(["model", "window"])[["auc", "ap"]].mean()
    dly = e2[e2.arm == "delay"].groupby(["model", "window", "k"])[["auc", "ap"]].mean()
    ctl = e2[e2.arm == "control"]

    # ---- long table per model, window and k (e2_table.csv; Table S1 is the k = 6 slice)
    rows = []
    for (m, w, k), g in ctl.groupby(["model", "window", "k"]):
        a0, p0 = full.loc[(m, w)]
        ad, pd_ = dly.loc[(m, w, k)]
        ac, pc = g.auc.mean(), g.ap.mean()
        rows.append({"Model": m, "Window": w, "k": k, "AUC no delay": a0, "AUC delay": ad, "AUC control": ac,
                     "Data loss": a0 - ac, "Recency loss": ac - ad, "Total loss": a0 - ad,
                     "AP data loss": p0 - pc, "AP recency loss": pc - pd_, "Control repeats": len(g)})
    t = pd.DataFrame(rows)
    t["Model"] = pd.Categorical(t.Model, models, ordered=True)
    t = t.sort_values(["Model", "Window", "k"], key=lambda s: s.map(wnum) if s.name == "Window" else s)
    t.round(4).to_csv(RESULTS / "e2_table.csv", index=False)

    # ---- coverage check
    for m in models:
        have = sorted(set(t[(t.Model == m) & (t.k == 6)].Window), key=wnum)
        missing = sorted(set(AVG_WINDOWS) - set(have), key=wnum)
        if missing:
            print(f"WARNING: {m} has no k = 6 results for {missing}; its k = 6 averages cover fewer windows.")

    # ---- Table 6: mean over t = 20-49 with ranges across control repeats
    t6 = []
    for (m, k), g in ctl[ctl.window.isin(AVG_WINDOWS)].groupby(["model", "k"]):
        wins = sorted(g.window.unique(), key=wnum)
        per_rep = []
        for _, gr in g.groupby("rep"):
            gr = gr.set_index("window")
            per_rep.append([np.mean([full.loc[(m, w), "auc"] - gr.loc[w, "auc"] for w in wins]),
                            np.mean([gr.loc[w, "auc"] - dly.loc[(m, w, k), "auc"] for w in wins])])
        per_rep = np.array(per_rep)
        lo, hi = (0, 100) if m == "TabPFN" else (2.5, 97.5)
        sub = t[(t.Model == m) & (t.k == k) & (t.Window.isin(wins))]
        t6.append({"Model": m, "k": k, "Windows": len(wins),
                   "AUC data loss": per_rep[:, 0].mean(),
                   "Data loss range": f"[{np.percentile(per_rep[:, 0], lo):+.4f}, {np.percentile(per_rep[:, 0], hi):+.4f}]",
                   "AUC recency loss": per_rep[:, 1].mean(),
                   "Recency loss range": f"[{np.percentile(per_rep[:, 1], lo):+.4f}, {np.percentile(per_rep[:, 1], hi):+.4f}]",
                   "AUC total loss": sub["Total loss"].mean(),
                   "AP data loss": sub["AP data loss"].mean(), "AP recency loss": sub["AP recency loss"].mean()})
    t6 = pd.DataFrame(t6)
    t6["Model"] = pd.Categorical(t6.Model, models, ordered=True)
    t6 = t6.sort_values(["Model", "k"])
    t6.round(4).to_csv(RESULTS / "e2_summary.csv", index=False)
    print("\nTable 6\n", t6.round(4).to_string(index=False))

    # ---- Table 7: total loss by window
    t7 = (full["auc"] - dly["auc"]).unstack("k")
    t7.insert(0, "AUC no delay", full["auc"])
    t7 = t7.reset_index()
    t7["model"] = pd.Categorical(t7.model, models, ordered=True)
    t7 = t7.sort_values(["model", "window"], key=lambda s: s.map(wnum) if s.name == "window" else s)
    t7.round(4).to_csv(RESULTS / "e2_total_loss_by_window.csv", index=False)
    print("\nTable 7\n", t7.round(4).to_string(index=False))

    # ---- Table S1
    s1 = t[(t.k == 6)][["Model", "Window", "Data loss", "Recency loss", "Total loss"]]
    s1.round(4).to_csv(RESULTS / "e2_table_S1.csv", index=False)
    print("\nTable S1 (k = 6)\n", s1.round(4).to_string(index=False))

    # ---- Figures
    plt = apply_style()
    FIGURES.mkdir(exist_ok=True)

    # Figure 2: decomposition, mean over t = 20-49
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        s = t6[t6.Model == m]
        ks = [str(k) for k in s.k]
        dl, rc = s["AUC data loss"].values, s["AUC recency loss"].values
        pos_b, neg_b = np.zeros(len(ks)), np.zeros(len(ks))
        for vals, c, lab in [(dl, "#2a78d6", "Data loss"), (rc, "#eb6834", "Recency loss")]:
            bottom = np.where(vals >= 0, pos_b, neg_b)
            ax.bar(ks, vals, 0.6, bottom=bottom, color=c, label=lab, edgecolor="white", linewidth=1.5)
            pos_b += np.clip(vals, 0, None); neg_b += np.clip(vals, None, 0)
        ax.plot(ks, s["AUC total loss"].values, "D", color=INK, ms=4, label="Total")
        ax.axhline(0, color=AXIS, lw=1); ax.set_title(m, fontsize=10, color=INK)
        ax.set_xlabel("Label delay k (periods)"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("AUC lost vs no delay"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "figure2_e2_decomposition.png", dpi=300); plt.close(fig)

    # Figure 3: AUC by delay, one line per window (delay solid, control dashed); own y-axis per panel
    wins = sorted(e2.window.unique(), key=wnum)
    WCOL = dict(zip(wins, ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]))
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.8))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        for w in wins:
            g = t[(t.Model == m) & (t.Window == w)]
            ks = [0] + list(g.k)
            ax.plot(ks, [full.loc[(m, w), "auc"]] + list(g["AUC delay"]), "-o", color=WCOL[w], lw=2, ms=5, label=w)
            ax.plot(ks, [full.loc[(m, w), "auc"]] + list(g["AUC control"]), "--", color=WCOL[w], lw=1.2, alpha=0.8)
        ax.set_title(m, fontsize=10, color=INK); ax.set_xlabel("Label delay k (periods, 2 weeks each)")
        ax.set_xticks([0, 1, 2, 4, 6]); ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("Walk-forward AUC")
    axes[0].legend(frameon=False, fontsize=8, title="Test window", title_fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "figure3_e2_auc_by_delay.png", dpi=300); plt.close(fig)
    print(f"\nsaved figures to {FIGURES}")


if __name__ == "__main__":
    main()
