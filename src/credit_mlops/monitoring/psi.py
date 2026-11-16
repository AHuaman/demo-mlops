"""Population Stability Index, written in-house (reference-quantile bins, epsilon floor)."""

import numpy as np
import pandas as pd


def _psi(reference_props: np.ndarray, current_props: np.ndarray, eps: float) -> float:
    r = np.clip(reference_props, eps, None)
    c = np.clip(current_props, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_numeric(
    reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4
) -> float:
    """PSI with bins cut at the reference quantiles; the outer edges are open-ended."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    inner = edges[1:-1]
    n_bins = len(inner) + 1
    ref_counts = np.bincount(np.digitize(reference, inner, right=False), minlength=n_bins)
    cur_counts = np.bincount(np.digitize(current, inner, right=False), minlength=n_bins)
    return _psi(ref_counts / len(reference), cur_counts / len(current), eps)


def psi_categorical(reference: pd.Series, current: pd.Series, eps: float = 1e-4) -> float:
    """PSI over the union of categories; a category absent on one side counts as `eps`."""
    categories = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))
    ref_props = reference.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    cur_props = current.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    return _psi(ref_props.to_numpy(), cur_props.to_numpy(), eps)
