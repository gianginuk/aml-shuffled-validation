"""Shared chart style for the figures."""

ORDER = ["Logistic", "Random Forest", "XGBoost", "TabPFN"]
MODEL_COLOURS = {"Logistic": "#2a78d6", "Random Forest": "#eb6834",
                 "XGBoost": "#1baf7a", "TabPFN": "#6250d6"}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#6b6a63", "#e1e0d9", "#c3c2b7"


def apply_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    return plt
