"""Data loading for the labelled Elliptic Bitcoin transactions.

The prepared data files come from the FIN510 teaching repository
(github.com/quinfer/fin510-colab-notebooks). They are downloaded on first use
into ``data/`` and are not redistributed in this repository.
"""

from pathlib import Path
import urllib.request

import pandas as pd

BASE_URL = ("https://raw.githubusercontent.com/quinfer/fin510-colab-notebooks/"
            "main/resources/elliptic/")
FILES = {
    "labelled": "elliptic_labelled_v2.parquet",
    "edges": "elliptic_edges_labelled_v2.parquet",
}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Test windows: t = 10-19, 20-29, 30-39, 40-49 (upper bound exclusive).
BOUNDARIES = [1, 10, 20, 30, 40, 50]
WINDOWS = [(BOUNDARIES[i + 1], BOUNDARIES[i + 2]) for i in range(len(BOUNDARIES) - 2)]

N_TRANSACTIONS = 46_564
N_ILLICIT = 4_545


def window_label(lo, hi):
    return f"t={lo}-{hi - 1}"


def _fetch(key, data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / FILES[key]
    if not path.exists():
        print(f"Downloading {FILES[key]} from the FIN510 repository ...")
        urllib.request.urlretrieve(BASE_URL + FILES[key], path)
    return path


def load_elliptic(data_dir=DATA_DIR):
    """Return (ell, X, y, ts, feat_cols) for the 46,564 labelled transactions.

    y = 1 for illicit (class 1), 0 for licit (class 2). Features are the 165
    supplied features, used as supplied (no further preprocessing here).
    """
    ell = pd.read_parquet(_fetch("labelled", data_dir))
    feat_cols = [c for c in ell.columns if c.startswith("feat_")]
    X = ell[feat_cols].values
    y = (ell["class"] == 1).astype(int).values
    ts = ell["time_step"].values
    assert len(y) == N_TRANSACTIONS and y.sum() == N_ILLICIT, \
        "sample does not match the paper (46,564 transactions, 4,545 illicit)"
    assert len(feat_cols) == 165
    return ell, X, y, ts, feat_cols


def load_labelled_edges(data_dir=DATA_DIR):
    """Edges between two labelled transactions (36,624 rows: txId1 -> txId2)."""
    return pd.read_parquet(_fetch("edges", data_dir))


def window_masks(ts, lo, hi):
    """Boolean masks for the walk-forward training set (t < lo) and test window."""
    return ts < lo, (ts >= lo) & (ts < hi)


def illicit_share_by_window(y, ts):
    return {window_label(lo, hi): y[(ts >= lo) & (ts < hi)].mean() for lo, hi in WINDOWS}
