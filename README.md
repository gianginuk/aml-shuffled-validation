# How Much Does Shuffled Validation Hide? Look-Ahead Bias and Label Delay in Cryptocurrency Anti-Money-Laundering Models

Code and results for the working paper by **Giang Ha** (Ulster University Business School).

The paper compares shuffled five-fold cross-validation with walk-forward validation on the 46,564 labelled transactions of the Elliptic Bitcoin dataset (time steps 1 to 49; evaluation on t = 10 to 49) for logistic regression, Random Forest, XGBoost and the tabular foundation model TabPFN-3.5. It then isolates the effect of future information with matched-size training sets (E1), measures the cost of label delay with a random-removal control (E2) and tests graph-degree features (E3).

> **Paper:** SSRN link to be added · **Archived code:** Zenodo DOI to be added

## Repository layout

```
src/amlval/            shared code: data loading, model configurations, metrics
experiments/           one script per experiment (E0 to E3)
analysis/              scripts that turn experiment outputs into the paper's tables and figures
results/paper/         the result files behind the published tables (archived outputs)
results/               outputs of your own runs are written here
figures/               figures produced by the analysis scripts
notebooks/original/    the original Kaggle and Colab notebooks that produced the results
environments/          exact library versions of the environments used in the paper
data/                  the data are downloaded here on first use (not committed)
```

## Where each table and figure comes from

| Paper | Experiment script | Analysis script | Archived result file |
|---|---|---|---|
| Table 1, Table 2 (E0) | `experiments/e0_baseline.py` | `analysis/e0_tables.py` | `results/paper/e0_table.csv`, `results/paper/e0_by_window.csv` |
| Table 3 (screening, top 1% and 5%) | `experiments/e0_baseline.py` | `analysis/e0_tables.py` | written to `results/e0_screening.csv` |
| Table 4, Table 5, Figure 1 (E1) | `experiments/e1_future_information.py` | `analysis/e1_tables.py` | `results/paper/e1_rows.csv` |
| Table 6, Table 7, Table S1, Figures 2 and 3 (E2) | `experiments/e2_label_delay.py` | `analysis/e2_tables.py` | `results/paper/e2_rows.csv` |
| Table 8, Table 9 (E3) | `experiments/e3_graph_degree.py` | (same script) | written to `results/e3_*.csv` |

The tables and figures can be rebuilt from the archived results without rerunning any model:

```bash
python analysis/e1_tables.py --rows results/paper/e1_rows.csv
python analysis/e2_tables.py --rows results/paper/e2_rows.csv
```

## Data

The analysis uses the prepared Elliptic data files from the FIN510 teaching repository ([quinfer/fin510-colab-notebooks](https://github.com/quinfer/fin510-colab-notebooks)), which the code downloads into `data/` on first use:

- `elliptic_labelled_v2.parquet`: 46,564 labelled transactions (4,545 illicit), 165 features, time step
- `elliptic_edges_labelled_v2.parquet`: 36,624 edges between labelled transactions

Construction B of E3 also needs the full edge list (234,355 edges), `elliptic_txs_edgelist.csv`, from the [Elliptic dataset on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). Place it in `data/`; without it, E3 runs constructions A, C and C2 only. The original dataset is described by Weber et al. (2019). The data are not redistributed here.

## Installation

Python 3.13 was used. For the classical models (CPU):

```bash
pip install -r requirements.txt
```

For TabPFN (GPU; the paper used a Kaggle Tesla T4):

```bash
pip install -r requirements-tabpfn.txt
export TABPFN_TOKEN=...      # TabPFN-3.5 weights require a Prior Labs account token
```

## Running the experiments

Run every command from the repository root.

```bash
# E0: shuffled vs walk-forward (classical models: about 10 minutes on 2 CPU cores)
python experiments/e0_baseline.py --models Logistic "Random Forest" XGBoost
python experiments/e0_baseline.py --models TabPFN                 # GPU, five seeds
python experiments/e0_baseline.py --models "TabPFN (10k cap)"     # GPU, one seed
python analysis/e0_tables.py                                      # Tables 1 to 3, paired bootstrap

# E1: effect of future information (about 2 hours on 2 CPU cores, mostly Random Forest)
python experiments/e1_future_information.py --models Logistic "Random Forest" XGBoost
python experiments/e1_future_information.py --models TabPFN --draws 25 --designs all_later
python analysis/e1_tables.py

# E2: label delay (TabPFN takes about 8 to 12 hours on a T4; the script resumes after interruption)
python experiments/e2_label_delay.py --models Logistic "Random Forest" XGBoost
python experiments/e2_label_delay.py --models TabPFN
python analysis/e2_tables.py

# E3: graph degree features (Random Forest, about 10 minutes on 2 CPU cores without B)
python experiments/e3_graph_degree.py
```

Scripts save progress as they go and skip finished steps on a rerun. Models that need a GPU (TabPFN) can be run on Kaggle with the same scripts: copy the repository into a notebook, add `TABPFN_TOKEN` as a secret, and copy the files written to `results/` back into `results/` here before running the analysis scripts.

## Settings used in the paper

| | Logistic regression | Random Forest | XGBoost | TabPFN |
|---|---|---|---|---|
| Configuration | standardised features, L2 (C = 1), 1,000 iterations, balanced class weight | 200 trees, depth 10, balanced class weight | 200 trees, depth 5, learning rate 0.1, no subsampling, `scale_pos_weight` from each training set | TabPFN-3.5 default, no rebalancing |
| Seeds (E0, E1 past-only, E2 full and delay) | one run (deterministic) | 42, 1, 7, 21, 99 | 42, 1, 7, 21, 99 (identical results) | 42, 1, 7, 21, 99 in E0; 42 in E2 |
| E1 draws | 100 | 100 | 100 | 25 (all later periods only) |
| E2 control repeats | 50 | 50 | 50 | 10 |

Shuffled cross-validation uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. The TabPFN 10,000-row variant trains on a stratified random sample of 10,000 transactions from each training set. The E0 paired bootstrap uses 1,000 resamples (random generator 2026). In E1, draw *b* uses random generator 1000 + *b* (illicit transactions drawn first, then licit) and model seed *b*; in E2, control repeat *r* uses random generator *r* and model seed *r*.

## Reproducibility notes

- **Library versions matter.** The paper's runs used two notebook environments (see `environments/`). A single-seed result can change sign across library versions (paper Section 4.4), so compare multi-seed means.
- **Checks against the archived results** (scikit-learn 1.6.1, XGBoost 3.4.1, CPU):
  - The analysis scripts rebuild Tables 4 to 7 and S1 exactly from the archived result files.
  - E0: rerunning the scripts reproduces the Random Forest and XGBoost results of Tables 1 and 2 exactly, including the bootstrap intervals. Logistic regression agrees to within 0.0002 in mean AUC (0.0007 in one window); the small differences come from the L-BFGS solver in newer SciPy versions.
  - Table 3: logistic regression and XGBoost are reproduced exactly; Random Forest differs by up to 0.003 because the paper computed this measure in a local environment (scikit-learn 1.9.1).
  - E1: the scripts draw exactly the same training samples as the original runs, and Random Forest and XGBoost reproduce the archived AUC and AP of each draw exactly; logistic regression agrees to within 0.001.
  - E2: the delay and control arms select exactly the same training transactions as the archived runs, and logistic regression AUCs agree to within 0.001.
  - E3: constructions A, C and C2 reproduce Table 8 exactly, and construction A reproduces Table 9.
- **Features contain test-period information.** As discussed in the paper (Šafář et al., 2026), the supplied features are address-level aggregates and were standardised on the whole sample. The walk-forward estimates therefore remove look-ahead through the validation design but not through the features, and should be read as upper bounds on deployable performance.

## Licence

The code is released under the MIT licence (see `LICENSE`). The TabPFN-3.5 model weights are distributed by Prior Labs under a non-commercial licence; they were used for research only and are not included here. The Elliptic data are subject to their own terms.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`):

> Ha, G. (2026) 'How much does shuffled validation hide? Look-ahead bias and label delay in cryptocurrency anti-money-laundering models', Working paper, Ulster University Business School.

## Acknowledgements

The prepared data file and the notebook framework come from the FIN510 teaching repository by Professor Barry Quinn, Ulster University Business School, who supervised the MSc dissertation (FIN741) on which this work builds.
