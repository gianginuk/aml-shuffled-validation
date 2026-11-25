"""Performance measures used in the paper."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auc_ap(y, p):
    return roc_auc_score(y, p), average_precision_score(y, p)


def error_ratio(auc_walk_forward, auc_shuffled):
    """(1 - walk-forward AUC) / (1 - shuffled AUC): how many times larger the
    out-of-time error is than shuffled validation suggests."""
    return (1 - auc_walk_forward) / (1 - auc_shuffled)


def top_share(y, p, share):
    """Screening measure (paper Table 3).

    Returns (found, precision) when only the top ``share`` of transactions by
    score are checked: found = share of all illicit transactions among them,
    precision = share of the checked transactions that are illicit.
    """
    n = max(1, int(round(share * len(y))))
    top = np.argsort(-p, kind="stable")[:n]
    hits = y[top].sum()
    return hits / y.sum(), hits / n


def max_found(y, share):
    """Highest attainable 'found' value at a given share (limited by prevalence)."""
    n = max(1, int(round(share * len(y))))
    return min(1.0, n / y.sum())
