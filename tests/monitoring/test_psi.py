import numpy as np
import pandas as pd
import pytest

from credit_mlops.monitoring.psi import psi_categorical, psi_numeric


def test_psi_identical_is_zero():
    x = np.arange(1000.0)
    assert psi_numeric(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_numeric_hand_computed():
    ref = np.arange(1000.0)
    cur = np.concatenate([np.arange(500.0), np.arange(500.0)])
    assert psi_numeric(ref, cur) == pytest.approx(3.7970, abs=1e-3)


def test_psi_categorical_hand_computed():
    ref = pd.Series(["A"] * 50 + ["B"] * 50)
    cur = pd.Series(["A"] * 80 + ["B"] * 20)
    assert psi_categorical(ref, cur) == pytest.approx(0.4159, abs=1e-3)


def test_psi_handles_new_category():
    assert psi_categorical(pd.Series(["A"] * 10), pd.Series(["A"] * 5 + ["C"] * 5)) > 0.25
