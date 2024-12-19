from functools import partial

import numpy as np
import pandas as pd

from catabra_lib import metrics
from catabra_lib.bootstrapping import Bootstrapping


def test_fn():
    rng = np.random.RandomState(23)
    y_true = (rng.uniform(0, 1, size=(247, 3)) < np.array([[0.5, 0.4, 0.1]])).astype(np.float32)
    y_pred = y_true + np.array([[0.6, 0.8, 1.0]]) * rng.randn(*y_true.shape) > 0.5

    def _test_fn(a, b):
        return dict(zip(["lbl1", "lbl2", "lbl3"], metrics.sensitivity(a, b, average=None)))

    # `fn` returns scalar
    bs = Bootstrapping(y_true, y_pred, fn=metrics.balanced_accuracy_macro).run(n_repetitions=10)
    assert len(bs.results) == 10
    df = bs.dataframe()
    assert df.shape == (10, 1)
    assert np.isclose(df.describe(), df.describe()).all()

    # `fn` returns dict of scalars
    bs = Bootstrapping(y_true, y_pred, fn=_test_fn).run(n_repetitions=10)
    assert isinstance(bs.results, dict)
    assert set(bs.results) == {"lbl1", "lbl2", "lbl3"}
    df = bs.dataframe()
    assert df.shape == (10, 3)
    assert (df.columns == ["lbl1", "lbl2", "lbl3"]).all()

    # `fn` is tuple of dict-returning functions
    bs = Bootstrapping(y_true, y_pred, fn=(_test_fn, _test_fn)).run(n_repetitions=10)
    assert isinstance(bs.results, tuple)
    assert len(bs.results) == 2
    assert all(isinstance(r, dict) for r in bs.results)
    assert all(set(r) == {"lbl1", "lbl2", "lbl3"} for r in bs.results)
    assert bs.dataframe() is None

    # `fn` is dict of scalar-returning functions
    bs = Bootstrapping(y_true, y_pred, fn=dict(f1=metrics.f1_micro, informedness=metrics.informedness_weighted)).run(
        n_repetitions=10
    )
    df = bs.dataframe()
    assert df.shape == (10, 2)
    assert (df.columns == ["f1", "informedness"]).all()
    s = bs.describe()
    assert (s.columns == ["f1", "informedness"]).all()
    assert np.isclose(df.describe(), s).all()

    # `fn` is tuple of scalar-returning functions
    bs = Bootstrapping(y_true, y_pred, fn=(metrics.f1_micro, metrics.informedness_weighted)).run(n_repetitions=10)
    df = bs.dataframe()
    assert df.shape == (10, 2)
    assert np.isclose(df.describe(), bs.describe()).all()

    # `fn` returns tuple of scalars
    bs = Bootstrapping(y_true, y_pred, fn=partial(metrics.precision_recall_fscore_support, average="macro")).run(
        n_repetitions=10
    )
    df = bs.dataframe()
    assert df.shape == (10, 4)
    assert df.iloc[:, 3].isna().all()
    assert np.isclose(df.describe(), bs.describe()).all()

    # `fn` returns tuple of arrays
    res = Bootstrapping(y_true, y_pred, fn=metrics.precision_recall_fscore_support).run(n_repetitions=10).results
    assert len(res) == 4
    assert all(len(r) == 10 for r in res)
    assert all(r0.shape == (3,) for r in res for r0 in r)


def test_seed():
    arg = np.empty((1634,), dtype=np.float32)

    for replace in (True, False):
        for size in (0.1, 0.5, 0.9, 1.0):
            idx = Bootstrapping(arg, seed=0, replace=replace, size=size).run().get_sample_indices()

            assert all(
                (idx == Bootstrapping(arg, seed=0, replace=replace, size=size).run().get_sample_indices()).all()
                for _ in range(10)
            )


def test_sample_indices():
    rng = np.random.RandomState(seed=73)
    arg = np.empty((309,), dtype=np.float32)
    sample_indices = rng.randint(0, len(arg), size=(50, len(arg)))

    assert (Bootstrapping(arg).run(sample_indices=sample_indices).get_sample_indices() == sample_indices).all()


def test_reset():
    rng = np.random.RandomState(0)
    seed = rng.randint(2**31)
    bs = Bootstrapping(rng.randn(1000) * 1.5 + 5, fn=np.mean, seed=seed).run(n_repetitions=50)
    res = np.array(bs.results)
    bs.reset(seed=seed)
    res2 = np.array(bs.run(n_repetitions=50).results)
    assert np.isclose(res, res2).all()


def test_aggregate():
    rng = np.random.RandomState(0)
    bs = Bootstrapping(rng.randn(1000) * 1.5 + 5, fn=lambda x: x[500], seed=rng.randint(2**31)).run(n_repetitions=500)

    assert 4.8 < bs.agg("mean") < 5.2
    assert 1.4 < bs.std() < 1.6

    q1, q3 = bs.agg("iqr")
    assert 3.5 < q1 < 5.0 < q3 < 6.5

    ci1, ci2 = bs.agg("ci.95")
    assert 1.7 < ci1 < 5.0 < ci2 < 8.3

    ci12, ci22 = bs.agg("ci95")
    assert np.isclose(ci12, ci1)
    assert np.isclose(ci22, ci2)

    ci13, ci23 = bs.confidence_interval(0.95)
    assert np.isclose(ci13, ci1)
    assert np.isclose(ci23, ci2)

    try:
        bs.agg("cixy")
    except AttributeError:
        pass
    else:
        assert False

    assert 4.8 < bs.quantile(0.5) < 5.2

    bs = Bootstrapping(
        rng.randn(1000) - 5, fn=lambda x: pd.DataFrame(data=dict(x=x, x_sq=np.square(x))), seed=rng.randint(2**31)
    ).run(n_repetitions=10)
    df = bs.mean()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1000, 2)
    assert (df.columns == ["x", "x_sq"]).all()
    assert df["x"].between(-7, -3).all()
    assert df["x_sq"].between(17, 36).all()

    bs = Bootstrapping(rng.randn(1000) - 5, fn=lambda x: pd.Series(np.abs(x)), seed=rng.randint(2**31)).run(
        n_repetitions=10
    )
    s = bs.var()
    assert isinstance(s, pd.Series)
    assert len(s) == 1000
    assert s.between(0.0, 2.5).all()
