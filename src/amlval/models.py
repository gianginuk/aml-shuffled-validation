"""Model configurations (paper Table A). No tuning is performed."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["Logistic", "Random Forest", "XGBoost", "TabPFN"]
CLASSICAL = ["Logistic", "Random Forest", "XGBoost"]

# Seeds used for the stochastic models throughout the paper.
SEEDS = [42, 1, 7, 21, 99]


def seeds_for(name, seeds=SEEDS):
    """Logistic regression is deterministic and is run once (seed 42)."""
    return [42] if name == "Logistic" else list(seeds)


def make_model(name, seed=42):
    if name == "Logistic":
        return Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ])
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                                      random_state=seed, n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        # No row or column subsampling, so the five seeds give identical predictions.
        return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                             random_state=seed, eval_metric="logloss", n_jobs=-1)
    if name == "TabPFN":
        from tabpfn import TabPFNClassifier
        # Default configuration of the TabPFN-3.5 checkpoint, GPU, no rebalancing.
        return TabPFNClassifier(device="cuda", random_state=seed)
    raise ValueError(f"unknown model: {name}")


def fit_proba(name, X, y, train_idx, test_idx, seed=42, batch=5000):
    """Fit on ``train_idx`` and return illicit probabilities for ``test_idx``.

    XGBoost's scale_pos_weight is computed from this training set only, so no
    information on the future illicit share enters training. Predictions are
    made in batches of 5,000 rows (needed for TabPFN on a T4 GPU).
    """
    m = make_model(name, seed)
    ytr = y[train_idx]
    if name == "XGBoost":
        m.set_params(scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum())
    m.fit(X[train_idx], ytr)
    Xte = X[test_idx]
    return np.concatenate([m.predict_proba(Xte[k:k + batch])[:, 1]
                           for k in range(0, len(Xte), batch)])
