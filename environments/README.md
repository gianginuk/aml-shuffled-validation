# Environments used in the paper

The experiments were run in cloud notebooks. The two environments below are listed in Section 3.3 of the paper.

- `kaggle_gpu.txt`: E0 (all models), the TabPFN windows of E1 and all of E2.
- `notebook_cpu.txt`: the classical models of E1 and the graph experiment E3.

The two environments produce identical Random Forest results. The top-share screening measure (Table 3) was computed in a third, local environment (scikit-learn 1.9.1) for the three classical models only.
