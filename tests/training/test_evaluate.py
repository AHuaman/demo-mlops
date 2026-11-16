from datetime import date

import numpy as np
import pandas as pd
import pytest

from credit_mlops.training.evaluate import evaluate, recency_weights


def test_calibration_ratio():
    m = evaluate(np.array([0, 0, 1, 1]), np.array([0.25, 0.25, 0.25, 0.25]))
    assert m.calibration_ratio == pytest.approx(0.5)


def test_segment_with_one_class_has_no_auc():
    m = evaluate(np.array([0, 1, 0]), np.array([0.1, 0.9, 0.2]), pd.Series(["a", "a", "b"]))
    assert m.by_segment["b"].roc_auc is None
    assert m.by_segment["a"].roc_auc == pytest.approx(1.0)


def test_recency_weights_halve_every_half_life():
    w = recency_weights(pd.Series([date(2023, 9, 1), date(2023, 6, 1)]), date(2023, 9, 1), 3.0)
    assert w.tolist() == pytest.approx([1.0, 0.5])


def test_flat_prefixes_metrics_and_skips_none():
    m = evaluate(np.array([0, 1, 0]), np.array([0.1, 0.9, 0.2]), pd.Series(["a", "a", "b"]))
    flat = m.flat("oot")
    assert flat["oot_roc_auc"] == pytest.approx(1.0)
    assert "oot_calibration_ratio__a" in flat
    assert "oot_calibration_ratio__b" not in flat
    assert "oot_roc_auc__b" not in flat
    assert "oot_roc_auc__a" in flat
