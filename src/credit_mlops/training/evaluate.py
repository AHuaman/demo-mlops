"""Evaluation metrics (overall and per segment) and recency weights."""

from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from credit_mlops.months import months_between


class SegmentMetrics(BaseModel):
    n: int
    default_rate: float
    roc_auc: float | None
    calibration_ratio: float | None
    calibration_z: float | None = None


class EvalMetrics(BaseModel):
    n: int
    default_rate: float
    roc_auc: float | None
    pr_auc: float | None
    brier: float
    calibration_ratio: float | None
    calibration_z: float | None = None
    by_segment: dict[str, SegmentMetrics]

    def flat(self, prefix: str) -> dict[str, float]:
        """Flat metric dict for logging; `None` values are skipped."""
        out: dict[str, float | None] = {
            f"{prefix}_n": self.n,
            f"{prefix}_default_rate": self.default_rate,
            f"{prefix}_roc_auc": self.roc_auc,
            f"{prefix}_pr_auc": self.pr_auc,
            f"{prefix}_brier": self.brier,
            f"{prefix}_calibration_ratio": self.calibration_ratio,
            f"{prefix}_calibration_z": self.calibration_z,
        }
        for name, seg in self.by_segment.items():
            out[f"{prefix}_n__{name}"] = seg.n
            out[f"{prefix}_default_rate__{name}"] = seg.default_rate
            out[f"{prefix}_roc_auc__{name}"] = seg.roc_auc
            out[f"{prefix}_calibration_ratio__{name}"] = seg.calibration_ratio
            out[f"{prefix}_calibration_z__{name}"] = seg.calibration_z
        return {k: float(v) for k, v in out.items() if v is not None}


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _calibration(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    observed = float(np.mean(y_true))
    return float(np.mean(y_prob)) / observed if observed > 0 else None


def _calibration_z(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Binomial z-score: (observed defaults - expected defaults) / sqrt(sum p(1-p))."""
    variance = float(np.sum(y_prob * (1 - y_prob)))
    if variance <= 0:
        return None
    return float((np.sum(y_true) - np.sum(y_prob)) / np.sqrt(variance))


def evaluate(
    y_true: np.ndarray, y_prob: np.ndarray, segments: pd.Series | None = None
) -> EvalMetrics:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    by_segment: dict[str, SegmentMetrics] = {}
    if segments is not None:
        seg_values = np.asarray(segments)
        for name in sorted(set(seg_values)):
            mask = seg_values == name
            by_segment[str(name)] = SegmentMetrics(
                n=int(mask.sum()),
                default_rate=float(y_true[mask].mean()),
                roc_auc=_auc(y_true[mask], y_prob[mask]),
                calibration_ratio=_calibration(y_true[mask], y_prob[mask]),
                calibration_z=_calibration_z(y_true[mask], y_prob[mask]),
            )
    one_class = len(np.unique(y_true)) < 2
    return EvalMetrics(
        n=len(y_true),
        default_rate=float(y_true.mean()),
        roc_auc=_auc(y_true, y_prob),
        pr_auc=None if one_class else float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        calibration_ratio=_calibration(y_true, y_prob),
        calibration_z=_calibration_z(y_true, y_prob),
        by_segment=by_segment,
    )


def recency_weights(months: pd.Series, reference: date, half_life_months: float) -> np.ndarray:
    """Weight 0.5 ** (age / half_life), where age is the months before `reference`."""
    ages = np.array([months_between(m, reference) for m in months], dtype=float)
    return 0.5 ** (ages / half_life_months)
