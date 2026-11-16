"""Reference-vs-current distribution plots (matplotlib lives here and nowhere else)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from credit_mlops.schema import CATEGORICAL_FEATURES  # noqa: E402


def plot_feature_drift(reference: pd.DataFrame, current: pd.DataFrame, feature: str) -> Figure:
    """Overlay the reference and current distribution of one feature."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if feature in CATEGORICAL_FEATURES:
        categories = sorted(set(reference[feature].unique()) | set(current[feature].unique()))
        x = np.arange(len(categories))
        ref = reference[feature].value_counts(normalize=True).reindex(categories, fill_value=0)
        cur = current[feature].value_counts(normalize=True).reindex(categories, fill_value=0)
        ax.bar(x - 0.2, ref.to_numpy(), width=0.4, label="reference")
        ax.bar(x + 0.2, cur.to_numpy(), width=0.4, label="current")
        ax.set_xticks(x, categories)
        ax.set_ylabel("share")
    else:
        both = np.concatenate([reference[feature].to_numpy(), current[feature].to_numpy()])
        edges = np.histogram_bin_edges(both.astype(float), bins=30)
        ax.hist(reference[feature], bins=edges, density=True, alpha=0.5, label="reference")
        ax.hist(current[feature], bins=edges, density=True, alpha=0.5, label="current")
        ax.set_ylabel("density")
    ax.set_title(feature)
    ax.legend()
    fig.tight_layout()
    return fig


def log_drift_figure(feature: str, reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """Log the drift plot of `feature` to the active MLflow run and always close the figure."""
    import mlflow

    fig = plot_feature_drift(reference, current, feature)
    try:
        mlflow.log_figure(fig, f"drift/{feature}.png")
    finally:
        plt.close(fig)
