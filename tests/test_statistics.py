import numpy as np
import pandas as pd

from catabra_lib.statistics import (
    chi_square,
    delong_test,
    mann_whitney_u,
    roc_auc_confidence_interval,
)


def test_mann_whitney_u():
    rng = np.random.RandomState(0)

    x = rng.uniform(-1, 1, size=50)
    assert mann_whitney_u(x, rng.randn(60) * 5) > 1e-1
    assert mann_whitney_u(x, rng.uniform(0, 2, 40)) < 1e-2

    x[[15, 17, 43]] = np.nan
    y = rng.uniform(-2, 2, 100)
    assert mann_whitney_u(x, y, nan_policy="omit") > 1e-1
    assert np.isnan(mann_whitney_u(x, y, nan_policy="propagate"))

    try:
        mann_whitney_u(x, y, nan_policy="raise")
    except ValueError:
        pass
    else:
        assert False


def test_chi_square():
    rng = np.random.RandomState(1)

    x = rng.binomial(3, 0.4, size=1000)
    assert chi_square(x, rng.binomial(3, 0.4, size=900)) > 1e-1
    assert chi_square(x, rng.binomial(3, 0.7, size=1010)) < 1e-2
    assert (
        chi_square(pd.Series(pd.Categorical.from_codes(x, categories=list(range(5)))), rng.binomial(3, 0.4, size=1500))
        > 1e-1
    )

    x = x.astype(np.float32)
    x[[15, 17, 43, 987]] = np.nan
    y = rng.binomial(3, 0.4, size=700)
    assert chi_square(x, y, nan_policy="omit") > 1e-1
    assert np.isnan(chi_square(x, y, nan_policy="propagate"))

    try:
        chi_square(x, y, nan_policy="raise")
    except ValueError:
        pass
    else:
        assert False


def test_delong_test():
    rng = np.random.RandomState(2)

    y_true = rng.uniform(0, 1, size=150) < 0.1
    y_hat_1 = rng.uniform(-1, 1, size=y_true.shape) + y_true.astype(np.float32)
    y_hat_2 = rng.uniform(-1, 1, size=y_true.shape) + y_true.astype(np.float32)
    assert delong_test(y_true, y_hat_1, y_hat_2) > 1e-1
    assert delong_test(y_true, y_hat_1, rng.uniform(-0.1, 0.1, size=y_true.shape) + y_true.astype(np.float32)) < 1e-2

    y_hat_1[[1, 87]] = np.nan
    y_hat_2[[71, 87, 143]] = np.nan
    assert delong_test(y_true, y_hat_1, y_hat_2, sample_weight=rng.randn(*y_true.shape) + 2.0) > 1e-1
    assert np.isnan(delong_test(y_true, y_hat_1, y_hat_2, nan_policy="propagate"))

    try:
        delong_test(y_true, y_hat_1, y_hat_2, nan_policy="raise")
    except ValueError:
        pass
    else:
        assert False


def test_roc_auc_confidence_interval():
    rng = np.random.RandomState(3)

    y_true = rng.uniform(0, 1, size=150) < 0.1
    y_hat = rng.uniform(-0.75, 0.75, size=y_true.shape) + y_true.astype(np.float32)

    auc, ci1, ci2 = roc_auc_confidence_interval(y_true, y_hat)
    assert 0.9 <= ci1 < auc < ci2 <= 1.0

    auc, ci1, ci2 = roc_auc_confidence_interval(y_true, y_hat, alpha=0.99)
    assert 0.85 <= ci1 < auc < ci2 <= 1.0

    y_hat[[64, 128]] = np.nan
    auc, ci1, ci2 = roc_auc_confidence_interval(y_true, y_hat, sample_weight=rng.randn(*y_true.shape) + 2.0)
    assert 0.88 <= ci1 < auc < ci2 <= 1.0

    auc, ci1, ci2 = roc_auc_confidence_interval(y_true, y_hat, nan_policy="propagate")
    assert np.isnan(auc)
    assert np.isnan(ci1)
    assert np.isnan(ci2)

    try:
        roc_auc_confidence_interval(y_true, y_hat, nan_policy="raise")
    except ValueError:
        pass
    else:
        assert False
