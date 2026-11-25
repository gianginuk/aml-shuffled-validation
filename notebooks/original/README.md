# Original notebooks

The notebooks that produced the paper's results, kept unchanged for the record. The scripts in `experiments/` and `analysis/` implement the same procedures, settings and random seeds, in a form that runs outside Kaggle and Colab.

| Notebook | Environment | Produced |
|---|---|---|
| `E0_shuffled_vs_walkforward_kaggle.ipynb` | Kaggle, Tesla T4 | E0 for all models, Tables 1 and 2 (`results/paper/e0_table.csv`, `e0_by_window.csv`) |
| `E1_future_information_TabPFN_kaggle.ipynb` | Kaggle, Tesla T4 | The TabPFN rows of E1 (appended to `results/paper/e1_rows.csv`); the classical rows came from the same procedure run earlier in Colab |
| `E2_label_delay_classical_kaggle.ipynb` | Kaggle, CPU | E2 for logistic regression, Random Forest and XGBoost |
| `E2_label_delay_TabPFN_kaggle.ipynb` | Kaggle, Tesla T4 | E2 for TabPFN (appended to `results/paper/e2_rows.csv`) |
| `E3_graph_leakage_colab.ipynb` | Colab, CPU | E3, Tables 8 and 9 |
