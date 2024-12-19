import numpy as np
import pandas as pd

from catabra_lib import preprocessing


def test_min_max_scaler():
    scaler = preprocessing.MinMaxScaler(fit_bool=False)

    # array
    scaler.fit(np.ones((17, 3), dtype=bool))
    assert (scaler.data_range_ == 1).all()
    assert (scaler.data_min_ == 0).all()
    assert (scaler.data_max_ == 1).all()
    x = np.random.uniform(0, 1, size=(12, 3)) < 0.5
    assert (scaler.transform(x) == x).all()

    # DataFrame
    scaler.fit(pd.DataFrame(data=dict(A=[-4, 2, 8], B=[True, True, True])))
    assert (scaler.data_range_ == np.array([12, 1])).all()
    assert (scaler.data_min_ == np.array([-4, 0])).all()
    assert (scaler.data_max_ == np.array([8, 1])).all()
    assert (
        scaler.transform(pd.DataFrame(data=dict(A=[5, -1], B=[True, False])))
        == pd.DataFrame(data=dict(A=[0.75, 0.25], B=[1.0, 0.0]))
    ).all(axis=None)


def test_feature_filter():
    filt = preprocessing.FeatureFilter()

    # array
    try:
        filt.fit(np.empty((5, 2)))
    except ValueError:
        pass
    else:
        assert False

    filt.fit(pd.DataFrame(columns=list("ABC"), index=pd.RangeIndex(0)))
    df = filt.transform(pd.DataFrame(data=dict(C=[5, -9, 4], A=["x", "y", "z"], X=[False, False, True])))
    assert (df.columns == list("ABC")).all()
    assert (df["A"] == ["x", "y", "z"]).all()
    assert df["B"].isna().all()
    assert (df["C"] == [5, -9, 4]).all()

    filt.add_missing = False
    try:
        filt.transform(pd.DataFrame(data=dict(C=[5, -9, 4], A=["x", "y", "z"])))
    except ValueError:
        pass
    else:
        assert False

    filt.remove_unknown = False
    try:
        filt.transform(pd.DataFrame(data=dict(C=[5, -9, 4], A=["x", "y", "z"], B=[0, 0, 0], X=[False, False, True])))
    except ValueError:
        pass
    else:
        assert False


def test_ordinal_encode():
    df = preprocessing.ordinal_encode(
        pd.DataFrame(
            data=dict(A=["X", None, "Y"], B=pd.Categorical.from_codes([-1, 2, 0], categories=list("ABC")), C=[5, 0, 1])
        ),
        obj="cat",
        output="pandas",
    )
    assert (df.columns == ["A", "B", "C"]).all()
    assert (df["A"] == [0, 2, 1]).all()
    assert np.isnan(df.loc[0, "B"])
    assert (df.loc[[1, 2], "B"] == [1, 0]).all()
    assert (df["C"] == [5, 0, 1]).all()


def test_one_hot_encode():
    df = preprocessing.one_hot_encode(
        pd.DataFrame(
            data=dict(
                A=["X", None, "Y"],
                B=pd.Categorical.from_codes([-1, 2, 0], categories=list("ABC")),
                C=[5, 0, 1],
                D=pd.Categorical.from_codes([-1] * 3, categories=[]),
            )
        ),
        drop_na=True,
        output="pandas",
    )
    assert (df.columns == ["B_A", "B_C", "C", "A"]).all()
    assert df.loc[0, "A"] == "X"
    assert df.loc[1, "A"] is None
    assert df.loc[2, "A"] == "Y"
    assert (df["B_A"] == [0, 0, 1]).all()
    assert (df["B_C"] == [0, 1, 0]).all()
    assert (df["C"] == [5, 0, 1]).all()


def test_k_bins_discretize():
    a = np.random.uniform(0, 10, size=100)
    b = pd.to_datetime(np.arange(75, 175), unit="m")
    df = preprocessing.k_bins_discretize(pd.DataFrame(data=dict(A=a, B=b)), n_bins=5, output="pandas")
    assert (df.columns == ["A_0.0", "A_1.0", "A_2.0", "A_3.0", "A_4.0", "B"]).all()
    assert (df["B"] == b).all()


def test_binarize():
    b = pd.to_datetime(np.arange(75, 175), unit="m")
    df = preprocessing.binarize(
        pd.DataFrame(data=dict(A=["<unknown>"] * len(b), B=b)),
        threshold=2,
        timestamp="[h]",
        obj="drop",
    )
    assert (df.columns == ["B"]).all()
    assert (df["B"] == (b > pd.Timestamp(2, unit="h"))).all()


def test_scale():
    x = np.linspace(0, np.pi, 100)
    df = preprocessing.scale(
        pd.DataFrame(data=dict(S=pd.to_datetime(np.sin(x), unit="m"), C=pd.to_timedelta(np.cos(x), unit="d"))),
        strategy="standard",
        timedelta="num",
        timestamp="num",
    )
    assert (df.columns == ["S", "C"]).all()
    for c in ("S", "C"):
        assert df[c].dtype.kind == "f"
        assert np.abs(df[c].mean()) < 1e-5
        assert np.abs(df[c].std() - 1.0) < 1e-2
