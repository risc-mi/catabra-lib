#  Copyright (c) 2024. RISC Software GmbH.
#  All rights reserved.

import sys
from typing import Optional, Tuple, Union

import numpy as np
from sklearn import __version__ as skl_version
from sklearn import preprocessing as skl_preprocessing
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

_SKL_MAJOR, _SKL_MINOR = [int(i) for i in skl_version.split(".", maxsplit=2)[:2]]


class MinMaxScaler(skl_preprocessing.MinMaxScaler):
    """Transform data by scaling each feature to a given range. The only difference to
    `sklearn.preprocessing.MinMaxScaler` is parameter `fit_bool` that, when set to False, does not fit this scaler on
    boolean features but rather uses 0 and 1 as fixed minimum and maximum values. This ensures that False is always
    mapped to `feature_range[0]` and True is always mapped to `feature_range[1]`. Otherwise, if the training data only
    contains True values, True would be mapped to `feature_range[0]` and False to `feature_range[0] - feature_range[1]`.
    The behavior on other numerical data types is not affected by this.

    Parameters
    ----------
    fit_bool : bool, default=True
        Whether to fit this scaler on boolean features. If True, the behavior is identical to
        `sklearn.preprocessing.MinMaxScaler`.
    **kwargs
        Additional keyword arguments, passed to `sklearn.preprocessing.MinMaxScaler`.

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler

    Notes
    -----
    Note that `sklearn.preprocessing.MaxAbsScaler` always maps False to 0 and True to 1, so there is no need for an
    analogous subclass.
    """

    def __init__(self, fit_bool: bool = True, **kwargs):
        super(MinMaxScaler, self).__init__(**kwargs)
        self.fit_bool = fit_bool

    def partial_fit(self, X, y=None) -> "MinMaxScaler":
        if isinstance(X, np.ndarray):
            if self.fit_bool or X.dtype.kind != "b":
                super(MinMaxScaler, self).partial_fit(X, y=y)
            else:
                if hasattr(self, "n_samples_seen_"):
                    self.data_min_ = np.minimum(self.data_min_, 0.0)
                    self.data_max_ = np.maximum(self.data_max_, 1.0)
                    self.n_samples_seen_ += X.shape[0]
                else:
                    self.data_min_ = np.zeros(X.shape[1], dtype=np.float64)
                    self.data_max_ = np.ones(X.shape[1], dtype=np.float64)
                    self.n_samples_seen_ = X.shape[0]

                self.data_range_ = self.data_max_ - self.data_min_
                self.scale_ = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
                self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        else:
            _ensure_dataframe(X, "MinMaxScaler.partial_fit")
            updates = []
            if not self.fit_bool:
                for i, c in enumerate(X.columns):
                    if X[c].dtype.kind == "b":
                        data_min = 0
                        data_max = 1
                        if hasattr(self, "data_min_"):
                            data_min = np.minimum(self.data_min_[i], data_min)
                            data_max = np.maximum(self.data_max_[i], data_max)
                        updates.append((i, data_min, data_max))

            super(MinMaxScaler, self).partial_fit(X, y=y)

            for i, data_min, data_max in updates:
                self.data_min_[i] = data_min
                self.data_max_[i] = data_max
                self.data_range_[i] = data_max - data_min
                self.scale_[i] = (self.feature_range[1] - self.feature_range[0]) / self.data_range_[i]
                self.min_[i] = self.feature_range[0] - data_min * self.scale_[i]

        return self

    @classmethod
    def _get_param_names(cls):
        # necessary, since otherwise `self.get_params()` only returns the params explicitly listed in `self.__init__()`
        # this negatively impacts `clone()`, for instance
        return ["fit_bool"] + skl_preprocessing.MinMaxScaler._get_param_names()


class OneHotEncoder(skl_preprocessing.OneHotEncoder):
    """Encode categorical features as a one-hot numeric array. The only difference to
    `sklearn.preprocessing.OneHotEncoder` is parameter `drop_na` that, when set to True, allows to drop NaN categories.
    More precisely, no separate columns representing NaN categories are added upon transformation, resembling the
    behavior of `pandas.get_dummies()`.

    Parameters
    ----------
    drop_na : bool, default=False
        Drop NaN categories. If False, the behavior is identical to `sklearn.preprocessing.OneHotEncode`.
    drop : iterable, optional
        Categories to drop. If `drop_na` is True, this parameter must be None.
    handle_unknown : str, optional
        How to handle unknown categories. If `drop_na` is True, this parameter must be "ignore". None defaults to
        "ignore" if `drop_na` is True and to "error" otherwise.
    min_frequency : int | float, optional
        Specifies the minimum frequency below which a category will be considered infrequent. If `drop_na` is True,
        this parameter must be None.
    max_categories : int, optional
        Specifies an upper limit to the number of output features for each input feature when considering infrequent
        categories. If `drop_na` is True, this parameter must be None.

    See Also
    --------
    sklearn.preprocessing.OneHotEncoder

    Notes
    -----
    If `drop_na` is True, all features containing only NaN values during `fit()` are removed entirely.
    """

    def __init__(
        self,
        drop_na: bool = False,
        drop=None,
        handle_unknown: Optional[str] = None,
        min_frequency: Union[int, float, None] = None,
        max_categories: Optional[int] = None,
        **kwargs,
    ):
        # Dropping NaN categories is very tricky to implement on top of scikit-learn's OneHotEncoder class.
        # Eventually, scikit-learn might natively support dropping NaN categories;
        # see https://github.com/scikit-learn/scikit-learn/issues/26543 for a discussion.

        self.drop_na = drop_na
        if self.drop_na:
            if drop is not None:
                raise ValueError("If `drop_na` is True, `drop` must be None.")
            if handle_unknown is None:
                handle_unknown = "ignore"
            elif handle_unknown != "ignore":
                raise ValueError('If `drop_na` is True, `handle_unknown` must be "ignore".')
            if min_frequency not in (1, None):
                raise ValueError("If `drop_na` is True, `min_frequency` must be None.")
            if max_categories is not None:
                raise ValueError("If `drop_na` is True, `max_categories` must be None.")
        elif handle_unknown is None:
            handle_unknown = "error"

        if min_frequency is not None:
            kwargs.update(min_frequency=min_frequency)
        if max_categories is not None:
            kwargs.update(max_categories=max_categories)

        super(OneHotEncoder, self).__init__(drop=drop, handle_unknown=handle_unknown, **kwargs)

    def fit(self, X, y=None) -> "OneHotEncoder":
        if self.drop_na:
            # check again
            if self.drop is not None:
                raise ValueError("If `drop_na` is True, `drop` must be None.")
            if self.handle_unknown != "ignore":
                raise ValueError('If `drop_na` is True, `handle_unknown` must be "ignore".')
            if getattr(self, "min_frequency", None) not in (None, 1):
                raise ValueError("If `drop_na` is True, `min_frequency` must be None.")
            if getattr(self, "max_categories", None) is not None:
                raise ValueError("If `drop_na` is True, `max_categories` must be None.")
            assert not getattr(self, "_infrequent_enabled", False)

        super(OneHotEncoder, self).fit(X, y=y)

        if self.drop_na:
            try:
                pd = sys.modules["pandas"]
            except KeyError:
                pd = None
            assert self.drop_idx_ is None
            drop_idx = []
            for categories in self.categories_:
                mask = np.isnan(categories) if pd is None else pd.isna(categories)
                if mask.any():
                    drop_idx.append(np.flatnonzero(mask)[-1])
                else:
                    drop_idx.append(None)
            if not all(i is None for i in drop_idx):
                self.drop_idx_ = np.asarray(drop_idx, dtype=object)
                if hasattr(self, "_drop_idx_after_grouping"):
                    self._drop_idx_after_grouping = self.drop_idx_
                if hasattr(self, "_compute_n_features_outs"):
                    self._n_features_outs = self._compute_n_features_outs()

        return self

    @classmethod
    def _get_param_names(cls):
        # necessary, since otherwise `self.get_params()` only returns the params explicitly listed in `self.__init__()`,
        # which would negatively impact `clone()`, for instance
        return ["drop_na"] + skl_preprocessing.OneHotEncoder._get_param_names()


class NumCatTransformer(BaseEstimator, TransformerMixin):
    """Transform numerical and categorical columns of a pandas DataFrame separately.

    The order of columns may change compared to the input: numerical columns come first, followed by categorical
    columns, followed by passed-through columns.

    Parameters
    ----------
    num_transformer : str | callable, optional
        The transformer to apply to numerical columns, or "passthrough" or "drop". Must implement `fit()` and
        `transform()`. Class instances are cloned before being fit to data, to ensure that the given instances are left
        unchanged.
    cat_transformer : str | callable, optional
        The transformer to apply to categorical columns, or "passthrough" or "drop". Must implement `fit()` and
        `transform()`. Class instances are cloned before being fit to data, to ensure that the given instances are left
        unchanged.
    bool : str, default="passthrough"
        How to treat boolean columns. One of "num", "cat", "passthrough" or "drop".
    obj : str, default="drop"
        How to treat columns with object data type. One of "num", "cat", "passthrough" or "drop".
    timedelta : str, default="num"
        How to treat timedelta columns. One of "num", "cat", "passthrough", "drop", "[ns]", "[us]", "[ms]", "[s]",
        "[m]", "[h]", "[d]", "[w]" or "[y]". A string representing a temporal resolution means that timedelta columns
        are first converted into floats by dividing by the given resolution, and then treating the result as numeric.
        This is useful if `num_transformer` does not natively support timedelta values.
    timestamp : str, default="num"
        How to treat timestamp/datetime columns. Same possibilities as for `timedelta`.

    See Also
    --------
    sklearn.compose.ColumnTransformer

    Notes
    -----
    This preprocessing transformation is only applicable to pandas DataFrames.
    """

    def __init__(
        self,
        num_transformer=None,
        cat_transformer=None,
        bool: str = "passthrough",
        obj: str = "drop",
        timedelta: str = "num",
        timestamp: str = "num",
    ):
        try:
            import pandas as pd
        except ImportError:
            raise ValueError("Class NumCatTransformer can only be instantiated if pandas is installed.")

        self._pd = pd
        self.num_transformer = num_transformer or "passthrough"
        self.cat_transformer = cat_transformer or "passthrough"
        self.bool = bool
        self.obj = obj
        self._timedelta = timedelta
        self._timestamp = timestamp
        self._timedelta_resolution = self._get_resolution(self._timedelta)
        self._timestamp_resolution = self._get_resolution(self._timestamp)

    @property
    def timedelta(self) -> str:
        return self._timedelta

    @property
    def timestamp(self) -> str:
        return self._timestamp

    @property
    def timedelta_resolution(self) -> Optional["pandas.Timedelta"]:  # noqa F821 # type: ignore
        return self._timedelta_resolution

    @property
    def timestamp_resolution(self) -> Optional["pandas.Timedelta"]:  # noqa F821 # type: ignore
        return self._timestamp_resolution

    def fit(self, X: "pandas.DataFrame", y=None) -> "NumCatTransformer":  # noqa F821 # type: ignore
        _ensure_dataframe(X, "NumCatTransformer.fit")

        self.num_cols_: list = []
        self.cat_cols_: list = []
        self.passthrough_cols_: list = []

        def _add_to_list(_c, _spec: str, _allow_temporal: bool):
            if _spec == "num":
                self.num_cols_.append(_c)
            elif _spec == "cat":
                self.cat_cols_.append(_c)
            elif _spec == "passthrough":
                self.passthrough_cols_.append(_c)
            elif _allow_temporal and _spec in (
                "[ns]",
                "[us]",
                "[ms]",
                "[s]",
                "[m]",
                "[h]",
                "[d]",
                "[w]",
                "[y]",
            ):
                self.num_cols_.append(_c)

        for c in X.columns:
            if X[c].dtype.name == "category":
                self.cat_cols_.append(c)
            elif X[c].dtype.kind == "b":
                _add_to_list(c, self.bool, False)
            elif X[c].dtype.kind == "O":
                _add_to_list(c, self.obj, False)
            elif X[c].dtype.kind == "M":
                _add_to_list(c, self._timestamp, True)
            elif X[c].dtype.kind == "m":
                _add_to_list(c, self._timedelta, True)
            elif X[c].dtype.kind in "uif":
                self.num_cols_.append(c)

        if self.cat_cols_ and not isinstance(self.cat_transformer, str):
            self.cat_transformer_ = (
                self.cat_transformer() if type(self.cat_transformer) is type else clone(self.cat_transformer)
            )
            self.cat_transformer_.fit(X[self.cat_cols_])
        else:
            self.cat_transformer_ = None
            if self.cat_cols_ and self.cat_transformer == "passthrough":
                self.passthrough_cols_ = self.cat_cols_ + self.passthrough_cols_
                self.cat_cols_ = []

        if self.num_cols_ and not isinstance(self.num_transformer, str):
            self.num_transformer_ = (
                self.num_transformer() if type(self.num_transformer) is type else clone(self.num_transformer)
            )
            X_num, _ = self._prepare_num(X[self.num_cols_])
            self.num_transformer_.fit(X_num)
        else:
            self.num_transformer_ = None
            if self.num_cols_ and self.num_transformer == "passthrough":
                self.passthrough_cols_ = self.num_cols_ + self.passthrough_cols_
                self.num_cols_ = []

        return self

    def transform(self, X: "pandas.DataFrame") -> Union["pandas.DataFrame", np.ndarray]:  # noqa F821 # type: ignore
        _ensure_dataframe(X, "NumCatTransformer.transform")

        check_is_fitted(self)
        self._validate_input(X)

        if self.num_transformer_ is not None:
            X_num, _ = self._prepare_num(X[self.num_cols_])
            num = self.num_transformer_.transform(X_num)
            num, num_df = self._postproc_num(num, X.index)
        else:
            num = num_df = None

        if self.cat_transformer_ is not None:
            cat = self.cat_transformer_.transform(X[self.cat_cols_])
            cat, cat_df = self._postproc_cat(cat, X.index)
        else:
            cat = cat_df = None

        return self._combine_results(num, num_df, cat, cat_df, X[self.passthrough_cols_])

    def set_output(self, *args, **kwargs):
        # like `ColumnTransformer`, we set the output of both `num_transformer` and `num_transformer_`
        for t in (
            self.num_transformer,
            self.cat_transformer,
            getattr(self, "num_transformer_", None),
            getattr(self, "cat_transformer_", None),
        ):
            if type(t) is not type and t is not None and not isinstance(t, str):
                t.set_output(*args, **kwargs)
        return self

    def _validate_input(self, df):
        diff = [c for c in self.num_cols_ + self.cat_cols_ + self.passthrough_cols_ if c not in df.columns]
        if diff:
            raise ValueError("X lacks columns " + str(diff))

    def _prepare_num(self, df) -> Tuple["pandas.DataFrame", bool]:  # noqa F821 # type: ignore
        copied = False
        if self._timedelta_resolution is not None:
            for c in df.columns:
                if df[c].dtype.kind == "m":
                    if not copied:
                        df = df.copy()
                        copied = True
                    df[c] = df[c] / self._timedelta_resolution
        if self._timestamp_resolution is not None:
            for c in df.columns:
                if df[c].dtype.kind == "M":
                    if not copied:
                        df = df.copy()
                        copied = True
                    df[c] = (df[c] - self._pd.Timestamp(0)) / self._timestamp_resolution

        return df, copied

    def _postproc_num(self, num, index: "pandas.Index"):  # noqa F821 # type: ignore
        num_df = None
        if hasattr(num, "toarray"):
            num = num.toarray()
        if isinstance(num, self._pd.DataFrame):
            assert (num.index == index).all()
            num_df = num
        elif num.shape[1] == len(self.num_cols_):
            num_df = self._pd.DataFrame(index=index, columns=self.num_cols_, data=num)
        elif isinstance(self.num_transformer_, skl_preprocessing.KBinsDiscretizer):
            if (
                self.num_transformer_.encode in ("onehot", "onehot-dense")
                and self.num_transformer_.n_bins_.sum() == num.shape[1]
                and len(self.num_transformer_.n_bins_) == len(self.num_cols_)
            ):
                num_df = self._pd.DataFrame(
                    index=index,
                    columns=[
                        f"{col}_{b}" for col, n in zip(self.num_cols_, self.num_transformer_.n_bins_) for b in range(n)
                    ],
                    data=num,
                )
        return num, num_df

    def _postproc_cat(self, cat, index: "pandas.Index"):  # noqa F821 # type: ignore
        cat_df = None
        if hasattr(cat, "toarray"):
            cat = cat.toarray()
        if isinstance(cat, self._pd.DataFrame):
            assert (cat.index == index).all()
            cat_df = cat
        elif isinstance(self.cat_transformer_, skl_preprocessing.OneHotEncoder):
            if self.cat_transformer_.drop_idx_ is not None:
                features_out = []
                for i, categories in enumerate(self.cat_transformer_.categories_):
                    j = self.cat_transformer_.drop_idx_[i]
                    if j is None:
                        features_out.append(categories)
                    elif j == 0:
                        features_out.append(categories[1:])
                    elif j + 1 == len(categories):
                        features_out.append(categories[:-1])
                    else:
                        features_out.append(np.concatenate([categories[:j], categories[j + 1 :]]))
            else:
                features_out = self.cat_transformer_.categories_
            if len(self.cat_cols_) == len(features_out) and cat.shape[1] == sum(len(f) for f in features_out):
                cat_df = self._pd.DataFrame(
                    index=index,
                    columns=[f"{col}_{f}" for col, feats in zip(self.cat_cols_, features_out) for f in feats],
                    data=cat.toarray() if hasattr(cat, "toarray") else cat,
                )
        elif cat.shape[1] == len(self.cat_cols_):
            cat_df = self._pd.DataFrame(index=index, columns=self.cat_cols_, data=cat)
        return cat, cat_df

    def _combine_results(self, num, num_df, cat, cat_df, passthrough_df):
        if (num is not None and num_df is None) or (cat is not None and cat_df is None):
            # return array
            arrs = []
            if num is not None:
                arrs.append(num)
            elif num_df is not None:
                arrs.append(num_df.values)
            if cat is not None:
                arrs.append(cat)
            elif cat_df is not None:
                arrs.append(cat_df.values)
            if not passthrough_df.empty:
                arrs.append(passthrough_df.values)
            return np.hstack(arrs)
        else:
            # return DataFrame
            dfs = []
            if num_df is not None:
                dfs.append(num_df)
            if cat_df is not None:
                dfs.append(cat_df)
            if not passthrough_df.empty:
                dfs.append(passthrough_df)
            return self._pd.concat(dfs, axis=1, sort=False)

    def _get_resolution(self, spec: str) -> Optional["pandas.Timedelta"]:  # noqa F821 # type: ignore
        if spec == "[ns]":
            return self._pd.Timedelta(1, unit="ns")
        elif spec == "[us]":
            return self._pd.Timedelta(1, unit="us")
        elif spec == "[ms]":
            return self._pd.Timedelta(1, unit="ms")
        elif spec == "[s]":
            return self._pd.Timedelta(1, unit="s")
        elif spec == "[m]":
            return self._pd.Timedelta(1, unit="m")
        elif spec == "[h]":
            return self._pd.Timedelta(1, unit="h")
        elif spec == "[d]":
            return self._pd.Timedelta(1, unit="d")
        elif spec == "[w]":
            return self._pd.Timedelta(7, unit="d")
        elif spec == "[y]":
            return self._pd.Timedelta(365.2525, unit="d")
        return None


class FeatureFilter(BaseEstimator, TransformerMixin):
    """Simple transformer that ensures that list of features is identical to features seen during fit.

    Parameters
    ----------
    add_missing : bool, default=True
        Add missing columns in `transform()`, filling them with NaN. If False, an error is raised instead.
    remove_unknown : bool, default=True
        Remove unknown columns in `transform()`. If False, an error is raised instead.

    Notes
    -----
    Data types are ignored, i.e., the output of `transform()` has the same data types as the input, which may differ
    from the data types seen during fit.

    This preprocessing transformation is only applicable to pandas DataFrames.
    """

    def __init__(self, add_missing: bool = True, remove_unknown: bool = True):
        super(FeatureFilter, self).__init__()
        self.add_missing = add_missing
        self.remove_unknown = remove_unknown

    def fit(self, X: "pandas.DataFrame", y=None) -> "FeatureFilter":  # noqa F821 # type: ignore
        _ensure_dataframe(X, "FeatureFilter.fit")
        self.columns_ = X.columns
        return self

    def transform(self, X: "pandas.DataFrame") -> "pandas.DataFrame":  # noqa F821 # type: ignore
        check_is_fitted(self)
        _ensure_dataframe(X, "FeatureFilter.transform")
        if X.shape[1] == len(self.columns_) and (X.columns == self.columns_).all():
            # fast track
            return X
        else:
            if not self.add_missing:
                missing = [c for c in self.columns_ if c not in X.columns]
                if missing:
                    raise ValueError("`X` is missing the following column(s): {}".format(missing))
            if not self.remove_unknown:
                unknown = [c for c in X.columns if c not in self.columns_]
                if unknown:
                    raise ValueError("`X` contains the following unknown column(s): {}".format(unknown))
            return X.reindex(self.columns_, axis=1)

    def set_output(self, transform: str = "default"):
        if transform not in ("default", "pandas"):
            raise ValueError("`{}` does not support output transform {}".format(self.__class__.__name__, transform))


# convenience functions


def ordinal_encoder(
    dtype=np.float64,
    num: str = "passthrough",
    bool: str = "passthrough",
    obj: str = "passthrough",
    timedelta: str = "num",
    timestamp: str = "num",
) -> NumCatTransformer:
    """Create a transformation for ordinal-encoding categorical features, while keeping other features unchanged.

    Parameters
    ----------
    dtype
        Data type of ordinal encoding. Passed to `sklearn.preprocessing.OrdinalEncoder`.
    num : str, default="passthrough"
        How to handle numerical features.
    bool : str, default="passthrough"
        How to handle boolean features.
    obj : str, default="passthrough"
        How to handle object features.
    timedelta : str, default="num"
        How to handle timedelta features.
    timestamp : str, default="num"
        How to handle datetime features.

    Returns
    -------
    NumCatTransformer instance that can be used for ordinal-encoding categorical columns in pandas DataFrames.

    See Also
    --------
    ordinal_encode
    sklearn.preprocessing.OrdinalEncoder
    """
    return NumCatTransformer(
        num_transformer=num,
        cat_transformer=skl_preprocessing.OrdinalEncoder(dtype=dtype),
        bool=bool,
        obj=obj,
        timedelta=timedelta,
        timestamp=timestamp,
    )


def ordinal_encode(
    X: "pandas.DataFrame",  # noqa F821 # type: ignore
    dtype=np.float64,
    num: str = "passthrough",
    bool: str = "passthrough",
    obj: str = "passthrough",
    timedelta: str = "num",
    timestamp: str = "num",
    output: str = "default",
) -> Union["pandas.DataFrame", np.ndarray]:  # noqa F821 # type: ignore
    """Ordinal-encode categorical features in a pandas DataFrame, while keeping other features unchanged.

    Internally, this function creates a suitable transformation using `ordinal_encoder()` and applies its
    `fit_transform()` method to the given DataFrame.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame to process.
    dtype
        Data type of ordinal encoding. Passed to `sklearn.preprocessing.OrdinalEncoder`.
    num : str, default="passthrough"
        How to handle numerical features.
    bool : str, default="passthrough"
        How to handle boolean features.
    obj : str, default="passthrough"
        How to handle object features.
    timedelta : str, default="num"
        How to handle timedelta features.
    timestamp : str, default="num"
        How to handle datetime features.
    output : str, default="default"
        Desired output type, either "default" (Numpy array) or "pandas" (pandas DataFrame).

    Returns
    -------
    Transformed input, either a DataFrame or an array, depending on `output`.

    See Also
    --------
    ordinal_encoder
    sklearn.preprocessing.OrdinalEncoder
    """
    transformer = ordinal_encoder(dtype=dtype, num=num, bool=bool, obj=obj, timedelta=timedelta, timestamp=timestamp)
    _try_set_output(transformer, output)
    return transformer.fit_transform(X)


def one_hot_encoder(
    drop_na: bool = False,
    drop=None,
    dtype=np.float64,
    handle_unknown: Optional[str] = None,
    num: str = "passthrough",
    bool: str = "passthrough",
    obj: str = "passthrough",
    timedelta: str = "num",
    timestamp: str = "num",
) -> NumCatTransformer:
    """Create a transformation for one-hot-encoding categorical features, while keeping other features unchanged.

    Parameters
    ----------
    drop_na : bool, default=False
        Drop NaN categories. Passed to `OneHotEncoder`.
    drop : iterable, optional
        Categories to drop. If `drop_na` is True, this parameter must be None. Passed to `OneHotEncoder`.
    dtype
        Data type of one-hot encoding. Passed to `OneHotEncoder`.
    handle_unknown : str, optional
        How to handle unknown categories. Passed to `OneHotEncoder`.
    num : str, default="passthrough"
        How to handle numerical features.
    bool : str, default="passthrough"
        How to handle boolean features.
    obj : str, default="passthrough"
        How to handle object features.
    timedelta : str, default="num"
        How to handle timedelta features.
    timestamp : str, default="num"
        How to handle datetime features.

    Returns
    -------
    NumCatTransformer instance that can be used for one-hot-encoding categorical columns in pandas DataFrames.

    See Also
    --------
    one_hot_encode
    OneHotEncoder

    Notes
    -----
    The resulting transformation always returns dense output in a pandas DataFrame.
    """
    if _SKL_MAJOR < 1 or (_SKL_MAJOR == 1 and _SKL_MINOR < 2):
        # <1.2
        kwargs = dict(sparse=False)
    else:
        # >=1.2
        kwargs = dict(sparse_output=False)
    return NumCatTransformer(
        num_transformer=num,
        cat_transformer=OneHotEncoder(
            drop_na=drop_na,
            drop=drop,
            dtype=dtype,
            handle_unknown=handle_unknown,
            **kwargs,
        ),
        bool=bool,
        obj=obj,
        timedelta=timedelta,
        timestamp=timestamp,
    )


def one_hot_encode(
    X: "pandas.DataFrame",  # noqa F821 # type: ignore
    drop_na: bool = False,
    drop=None,
    dtype=np.float64,
    handle_unknown: Optional[str] = None,
    num: str = "passthrough",
    bool: str = "passthrough",
    obj: str = "passthrough",
    output: str = "default",
    timedelta: str = "num",
    timestamp: str = "num",
) -> Union["pandas.DataFrame", np.ndarray]:  # noqa F821 # type: ignore
    """One-hot encode categorical features in a pandas DataFrame, while keeping other features unchanged.

    Internally, this function creates a suitable transformation using `one_hot_encoder()` and applies its
    `fit_transform()` method to the given DataFrame.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame to process.
    drop_na : bool, default=False
        Drop NaN categories. Passed to `OneHotEncoder`.
    drop : iterable, optional
        Categories to drop. If `drop_na` is True, this parameter must be None. Passed to `OneHotEncoder`.
    dtype
        Data type of one-hot encoding. Passed to `OneHotEncoder`.
    num : str, default="passthrough"
        How to handle numerical features.
    bool : str, default="passthrough"
        How to handle boolean features.
    obj : str, default="passthrough"
        How to handle object features.
    timedelta : str, default="num"
        How to handle timedelta features.
    timestamp : str, default="num"
        How to handle datetime features.
    output : str, default="default"
        Desired output type, either "default" (Numpy array) or "pandas" (pandas DataFrame).

    Returns
    -------
    Transformed input, either a DataFrame or an array, depending on `output`.

    See Also
    --------
    one_hot_encoder
    OneHotEncoder
    """
    transformer = one_hot_encoder(
        drop_na=drop_na,
        drop=drop,
        dtype=dtype,
        handle_unknown=handle_unknown,
        num=num,
        bool=bool,
        obj=obj,
        timedelta=timedelta,
        timestamp=timestamp,
    )
    _try_set_output(transformer, output)
    return transformer.fit_transform(X)


def k_bins_discretizer(
    n_bins: int = 5,
    encode: str = "onehot",
    strategy: str = "quantile",
    cat: str = "passthrough",
    bool: str = "passthrough",
    obj: str = "passthrough",
    timedelta: str = "num",
    timestamp: str = "passthrough",
) -> NumCatTransformer:
    """Create a transformation for k-bins-discretizing numerical features, while keeping other features unchanged.

    Parameters
    ----------
    n_bins : int, default=5
        Number of bins to produce. Passed to `sklearn.preprocessing.KBinsDiscretizer`.
    encode : str, default="onehot"
        Method used to encode the transformed result. Passed to `sklearn.preprocessing.KBinsDiscretizer`.
    strategy : str, default="quantile"
        Strategy used to define the widths of the bins. Passed to `sklearn.preprocessing.KBinsDiscretizer`.
    cat : str, default="passthrough"
        How to handle categorical features.
    bool : str, default="passthrough"
        How to handle boolean features.
    obj : str, default="passthrough"
        How to handle object features.
    timedelta : str, default="num"
        How to handle timedelta features.
    timestamp : str, default="num"
        How to handle datetime features.

    Returns
    -------
    NumCatTransformer instance that can be used for k-bins-discretizing numerical columns in pandas DataFrames.

    See Also
    --------
    k_bins_discretize
    sklearn.preprocessing.KBinsDiscretizer
    """
    return NumCatTransformer(
        num_transformer=skl_preprocessing.KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy),
        cat_transformer=cat,
        bool=bool,
        obj=obj,
        timedelta=timedelta,
        timestamp=timestamp,
    )


def k_bins_discretize(
    X: "pandas.DataFrame",  # noqa F821 # type: ignore
    n_bins: int = 5,
    encode: str = "onehot",
    strategy: str = "quantile",
    cat: str = "passthrough",
    bool: str = "passthrough",
    obj: str = "passthrough",
    timedelta: str = "num",
    timestamp: str = "passthrough",
    output: str = "default",
) -> Union["pandas.DataFrame", np.ndarray]:  # noqa F821 # type: ignore
    """K-bins discretize numerical features in a pandas DataFrame, while keeping other features unchanged.

    Internally, this function creates a suitable transformation using `k_bins_discretizer()` and applies its
    `fit_transform()` method to the given DataFrame.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame to process.
    n_bins : int, default=5
        Number of bins to produce. Passed to `sklearn.preprocessing.KBinsDiscretizer`.
    encode : str, default="onehot"
        Method used to encode the transformed result. Passed to `sklearn.preprocessing.KBinsDiscretizer`.
    strategy : str, default="quantile"
        Strategy used to define the widths of the bins. Passed to `sklearn.preprocessing.KBinsDiscretizer`.
    num : str, default="passthrough"
        How to handle numerical features.
    bool : str, default="passthrough"
        How to handle boolean features.
    obj : str, default="passthrough"
        How to handle object features.
    timedelta : str, default="num"
        How to handle timedelta features.
    timestamp : str, default="num"
        How to handle datetime features.
    output : str, default="default"
        Desired output type, either "default" (Numpy array) or "pandas" (pandas DataFrame).

    Returns
    -------
    Transformed input, either a DataFrame or an array, depending on `output`.

    See Also
    --------
    k_bins_discretizer
    sklearn.preprocessing.KBinsDiscretizer
    """
    if output == "pandas" and encode == "onehot":
        encode = "onehot-dense"
    transformer = k_bins_discretizer(
        n_bins=n_bins,
        encode=encode,
        strategy=strategy,
        cat=cat,
        bool=bool,
        obj=obj,
        timedelta=timedelta,
        timestamp=timestamp,
    )
    _try_set_output(transformer, output)
    return transformer.fit_transform(X)


def binarizer(
    threshold: float = 0,
    cat: str = "passthrough",
    bool: str = "passthrough",
    obj: str = "passthrough",
    timedelta: str = "passthrough",
    timestamp: str = "passthrough",
) -> NumCatTransformer:
    """Create a transformation for binarizing numerical features, while keeping other features unchanged.

    Parameters
    ----------
    threshold : float, default=0
        Feature values below or equal to this are replaced by 0, above it by 1.
        Passed to `sklearn.preprocessing.Binarizer`.
    cat : str, default="passthrough"
        How to handle categorical features.
    bool : str, default="passthrough"
        How to handle boolean features.
    obj : str, default="passthrough"
        How to handle object features.
    timedelta : str, default="num"
        How to handle timedelta features.
    timestamp : str, default="num"
        How to handle datetime features.

    Returns
    -------
    NumCatTransformer instance that can be used for binarizing numerical columns in pandas DataFrames.

    See Also
    --------
    binarize
    sklearn.preprocessing.Binarizer
    """
    return NumCatTransformer(
        num_transformer=skl_preprocessing.Binarizer(threshold=threshold),
        cat_transformer=cat,
        bool=bool,
        obj=obj,
        timedelta=timedelta,
        timestamp=timestamp,
    )


def binarize(
    X: "pandas.DataFrame",  # noqa F821 # type: ignore
    threshold: float = 0,
    cat: str = "passthrough",
    bool: str = "passthrough",
    obj: str = "passthrough",
    timedelta: str = "passthrough",
    timestamp: str = "passthrough",
    output: str = "default",
) -> Union["pandas.DataFrame", np.ndarray]:  # noqa F821 # type: ignore
    """Binarize numerical features in a pandas DataFrame, while keeping other features unchanged.

    Internally, this function creates a suitable transformation using `binarizer()` and applies its `fit_transform()`
    method to the given DataFrame.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame to process.
    threshold : float, default=0
        Feature values below or equal to this are replaced by 0, above it by 1.
        Passed to `sklearn.preprocessing.Binarizer`.
    num : str, default="passthrough"
        How to handle numerical features.
    bool : str, default="passthrough"
        How to handle boolean features.
    obj : str, default="passthrough"
        How to handle object features.
    timedelta : str, default="num"
        How to handle timedelta features.
    timestamp : str, default="num"
        How to handle datetime features.
    output : str, default="default"
        Desired output type, either "default" (Numpy array) or "pandas" (pandas DataFrame).

    Returns
    -------
    Transformed input, either a DataFrame or an array, depending on `output`.

    See Also
    --------
    binarizer
    sklearn.preprocessing.Binarizer
    """
    transformer = binarizer(threshold=threshold, cat=cat, bool=bool, obj=obj, timedelta=timedelta, timestamp=timestamp)
    _try_set_output(transformer, output)
    return transformer.fit_transform(X)


def scaler(
    strategy: str = "standard",
    cat: str = "passthrough",
    bool: str = "passthrough",
    obj: str = "passthrough",
    timedelta: str = "num",
    timestamp: str = "passthrough",
    fit_bool=None,
    **kwargs,
) -> NumCatTransformer:
    """Create a transformation for scaling numerical features, while keeping other features unchanged.

    Parameters
    ----------
    strategy : str, default="standard"
        Strategy used to scale numerical data:
        * "standard": Scale data to have zero mean and unit variance, using `sklearn.preprocessing.StandardScaler`.
        * "robust": Scale data using statistics that are robust to outliers, using `sklearn.preprocessing.RobustScaler`.
        * "minmax": Scale data to have zero minimum and unit maximum, using `sklearn.preprocessing.MinMaxScaler`.
        * "maxabs": Scale data to have a maximum absolute value of 1, using `sklearn.preprocessing.MaxAbsScaler`.

    cat : str, default="passthrough"
        How to handle categorical features.
    bool : str, default="passthrough"
        How to handle boolean features.
    obj : str, default="passthrough"
        How to handle object features.
    timedelta : str, default="num"
        How to handle timedelta features.
    timestamp : str, default="num"
        How to handle datetime features.
    **kwargs
        Additional keyword arguments passed to the underlying scikit-learn scaler.

    Returns
    -------
    NumCatTransformer instance that can be used for scaling numerical columns in pandas DataFrames.

    See Also
    --------
    scale
    sklearn.preprocessing.StandardScaler
    sklearn.preprocessing.RobustScaler
    sklearn.preprocessing.MinMaxScaler
    sklearn.preprocessing.MaxAbsScaler
    """
    if fit_bool is not None:
        print(
            "Passing `fit_bool` to function `scale` has no effect."
            " Use parameter `bool` to control the scaling behavior for boolean data."
        )
    if strategy == "standard":
        num_transformer = skl_preprocessing.StandardScaler(**kwargs)
    elif strategy == "robust":
        num_transformer = skl_preprocessing.RobustScaler(**kwargs)
    elif strategy == "minmax":
        num_transformer = skl_preprocessing.MinMaxScaler(**kwargs)
    elif strategy == "maxabs":
        num_transformer = skl_preprocessing.MaxAbsScaler(**kwargs)
    else:
        raise ValueError(
            'Scaling strategy must be one of "standard", "robust", "minmax" or "maxabs", but got "{}".'.format(strategy)
        )
    if timedelta == "num":
        # we have to pass some resolution, but it does not matter which
        timedelta = "[s]"
    if timestamp == "num":
        timestamp = "[s]"
    return NumCatTransformer(
        num_transformer=num_transformer,
        cat_transformer=cat,
        bool=bool,
        obj=obj,
        timedelta=timedelta,
        timestamp=timestamp,
    )


def scale(
    X: "pandas.DataFrame",  # noqa F821 # type: ignore
    strategy: str = "standard",
    cat: str = "passthrough",
    bool: str = "passthrough",
    obj: str = "passthrough",
    timedelta: str = "num",
    timestamp: str = "passthrough",
    output: str = "default",
    **kwargs,
) -> Union["pandas.DataFrame", np.ndarray]:  # noqa F821 # type: ignore
    """Scale numerical features in a pandas DataFrame, while keeping other features unchanged.

    Internally, this function creates a suitable transformation using `scaler()` and applies its `fit_transform()`
    method to the given DataFrame.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame to process.
    strategy : str, default="standard"
        Strategy used to scale numerical data:
        * "standard": Scale data to have zero mean and unit variance, using `sklearn.preprocessing.StandardScaler`.
        * "robust": Scale data using statistics that are robust to outliers, using `sklearn.preprocessing.RobustScaler`.
        * "minmax": Scale data to have zero minimum and unit maximum, using `sklearn.preprocessing.MinMaxScaler`.
        * "maxabs": Scale data to have a maximum absolute value of 1, using `sklearn.preprocessing.MaxAbsScaler`.

    num : str, default="passthrough"
        How to handle numerical features.
    bool : str, default="passthrough"
        How to handle boolean features.
    obj : str, default="passthrough"
        How to handle object features.
    timedelta : str, default="num"
        How to handle timedelta features.
    timestamp : str, default="num"
        How to handle datetime features.
    output : str, default="default"
        Desired output type, either "default" (Numpy array) or "pandas" (pandas DataFrame).
    **kwargs
        Additional keyword arguments passed to the underlying scikit-learn scaler.

    Returns
    -------
    Transformed input, either a DataFrame or an array, depending on `output`.

    See Also
    --------
    scaler
    sklearn.preprocessing.StandardScaler
    sklearn.preprocessing.RobustScaler
    sklearn.preprocessing.MinMaxScaler
    sklearn.preprocessing.MaxAbsScaler
    """
    transformer = scaler(
        strategy=strategy,
        cat=cat,
        bool=bool,
        obj=obj,
        timedelta=timedelta,
        timestamp=timestamp,
        **kwargs,
    )
    _try_set_output(transformer, output)
    return transformer.fit_transform(X)


def _ensure_dataframe(df, method_name: str):
    if not hasattr(df, "columns"):
        raise ValueError("{} does not support input of type {}".format(method_name, type(df)))


def _try_set_output(transformer, output: str):
    if output != "default":
        try:
            transformer.set_output(transform=output)
        except AttributeError:
            import warnings

            msg = "Transformer of type {} does not implement the `set_output` method.".format(type(transformer))
            if _SKL_MAJOR < 1 or (_SKL_MAJOR == 1 and _SKL_MINOR < 2):
                msg += " Upgrading scikit-learn to version 1.2.0 might solve the problem."
            warnings.warn(msg)

    return transformer
