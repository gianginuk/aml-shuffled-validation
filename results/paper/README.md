# Archived results behind the published tables

These files are the outputs of the runs reported in the paper (working paper, version 1). The analysis scripts read them with `--rows`, so the tables can be rebuilt without rerunning any model.

| File | Contents | Used for |
|---|---|---|
| `e0_table.csv` | Shuffled vs walk-forward AUC and AP, gaps, error ratios and paired-bootstrap 95% intervals, per model | Table 1 |
| `e0_by_window.csv` | Walk-forward AUC by test window (mean and SD across seeds) | Table 2 |
| `e1_rows.csv` | One row per model, design (`all_later`, `adjacent`), test window and random draw `b`: past-only and future-informed AUC and AP, share of the training sample drawn from later periods, pool sizes | Tables 4 and 5, Figure 1 |
| `e2_rows.csv` | One row per model, test window, arm (`full`, `delay`, `control`), delay `k`, repeat and seed: training size, illicit count, AUC and AP | Tables 6, 7 and S1, Figures 2 and 3 |

Column definitions follow the experiment scripts in `experiments/`.
