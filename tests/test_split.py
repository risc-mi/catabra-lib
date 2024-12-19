import numpy as np
import pandas as pd
import pytest

from catabra_lib.split import (
    CustomPredefinedSplit,
    StratifiedGroupKFold,
    StratifiedGroupShuffleSplit,
)


@pytest.mark.parametrize(
    argnames=["method", "seed"],
    argvalues=[
        ("brute_force", 42),
        ("exact", 9999),
        ("automatic", 1),
    ],
)
def test_stratified_group_shuffle_split(method: str, seed: int):
    rng = np.random.RandomState(seed)
    groups = np.abs(rng.normal(scale=100, size=10000)).astype(np.int32)
    y = rng.uniform(0, 1, size=groups.shape) < 0.1

    splitter = StratifiedGroupShuffleSplit(n_splits=10, test_size=0.2, method=method, random_state=seed)
    i = 0
    for train, test in splitter.split(None, y=y, groups=groups):
        assert np.abs(len(test) - 0.2 * len(y)) < 100, len(test)
        assert len(np.intersect1d(groups[train], groups[test])) == 0
        assert np.abs(y[train].mean() - y[test].mean()) < 3e-2
        i += 1

    assert i == 10


@pytest.mark.parametrize(
    argnames=["method", "seed"],
    argvalues=[
        ("brute_force", 937105),
        ("exact", 14638),
        ("automatic", 8003),
    ],
)
def test_stratified_group_k_fold(method: str, seed: int):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, method=method, random_state=seed)
    rng = np.random.RandomState(seed)
    groups = np.abs(rng.normal(scale=100, size=10000)).astype(np.int32)
    y = rng.uniform(0, 1, size=groups.shape) < 0.1

    all_test = []
    for train, test in splitter.split(y[..., np.newaxis], y=y, groups=groups):
        assert np.abs(len(test) - 0.2 * len(y)) < 50
        assert len(np.intersect1d(groups[train], groups[test])) == 0
        assert np.abs(y[train].mean() - y[test].mean()) < 3e-2
        all_test.append(test)

    assert len(all_test) == 5
    assert sum(len(t) for t in all_test) == len(y)
    assert len(np.unique(np.concatenate(all_test))) == len(y)


def test_custom_predefined_split():
    x = np.random.uniform(0, 1, size=(100, 4)) < np.array([[0.1, 0.3, 0.5, 0.9]])
    splitter = CustomPredefinedSplit.from_data(x)
    assert splitter.get_n_splits() == x.shape[1]
    for i, (train, test) in enumerate(splitter.split(x)):
        assert (train == np.flatnonzero(~x[:, i])).all()
        assert (test == np.flatnonzero(x[:, i])).all()

    df = pd.DataFrame(data=dict(A=[True, True, False, True, False], B=[False, True, True, False, False], C=["?"] * 5))
    splitter = CustomPredefinedSplit.from_data(df, columns=["A", "B"])
    assert splitter.get_n_splits() == 2
    (train_1, test_1), (train_2, test_2) = list(splitter.split(df))
    assert (train_1 == np.flatnonzero(~df["A"])).all()
    assert (test_1 == np.flatnonzero(df["A"])).all()
    assert (train_2 == np.flatnonzero(~df["B"])).all()
    assert (test_2 == np.flatnonzero(df["B"])).all()
