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


class DTypeTransformer(BaseEstimator, TransformerMixin):
    """Transform columns of a pandas DataFrame depending on their data types.

    The order of columns may change compared to the input.

    Parameters
    ----------
    num : str | BaseEstimator, optional
        The transformation to apply to numerical columns, or "passthrough", "drop", "num", "cat", "bool", "timedelta",
        "datetime", "obj" or None/"default":
        * BaseEstimator: Apply the BaseEstimator to all columns with numerical data type. The BaseEstimator must
            implement `fit()` and `transform()`. Class instances are cloned before being fit to data, to ensure that
            the given instances are left unchanged.
        * "passthrough": Pass numerical columns through unchanged.
        * "drop": Drop numerical columns.
        * "num": Prohibited here, but allowed with `cat`, `bool`, `timedelta`, `datetime`, `obj` and `default`: Treat
            columns of the respective data type as numerical and apply the transformation specified by `num`.
        * "cat": Treat numerical columns like categorical columns, and apply the transformation specified by `cat`.
        * "bool": Treat numerical columns like boolean columns, and apply the transformation specified by `bool`.
        * "timedelta": Treat numerical columns like timedelta columns, and apply the transformation specified by
            `timedelta`.
        * "datetime": Treat numerical columns like datetime columns, and apply the transformation specified by
            `datetime`.
        * "obj": Treat numerical columns like columns with object data type, and apply the transformation specified by
            `obj`.
        * None or "default": Apply the default transformation, specified by `default`.

    cat : str | BaseEstimator, optional
        The transformation to apply to categorical columns. Same options as for `num`.
    bool : str | BaseEstimator, optional
        The transformation to apply to boolean columns. Same options as for `num`.
    timedelta : str | BaseEstimator, optional
        The transformation to apply to timedelta columns. Same options as for `num`.
    timestamp : str | BaseEstimator, optional
        The transformation to apply to datetime columns. Same options as for `num`.
    obj : str | BaseEstimator, optional
        The transformation to apply to columns with object data type. Same options as for `num`.
    default : str | BaseEstimator, default="passthrough"
        Default behavior for columns with unspecified transformation. Same options as for `num`, but cannot be None.
    timedelta_resolution : str | pandas.Timedelta, optional
        Convert timedelta columns to float by diving through the given temporal resolution. This transformation is
        applied before any other transformation, and regardless of the value of `timedelta`.
        None keeps the data type of timedelta columns.
    datetime_resolution : str | pandas.Timedelta, optional
        Convert datetime columns to float by diving through the given temporal resolution. This transformation is
        applied before any other transformation, and regardless of the value of `datetime`.
        None keeps the data type of timedelta columns.

    See Also
    --------
    sklearn.compose.ColumnTransformer

    Notes
    -----
    This preprocessing transformation is only applicable to pandas DataFrames.

    If the transformation specification is recursive, `fit()` raises a ValueError. Recursive specifications arise
    when some data type A shall be treated like B, B shall be treated like C, C shall be treated like ... A.
    """

    def __init__(
        self,
        num=None,
        cat=None,
        bool=None,
        timedelta=None,
        datetime=None,
        obj=None,
        default="passthrough",
        timedelta_resolution: Union[str, "pandas.Timedelta", None] = None,  # noqa F821 # type: ignore
        datetime_resolution: Union[str, "pandas.Timedelta", None] = None,  # noqa F821 # type: ignore
    ):
        try:
            import pandas as pd
        except ImportError:
            raise ValueError("Class DTypeTransformer can only be instantiated if pandas is installed.")

        self._pd = pd
        self.num = num
        self.cat = cat
        self.bool = bool
        self.obj = obj
        self.timedelta = timedelta
        self.datetime = datetime
        self.default = default
        self.timedelta_resolution = timedelta_resolution
        self.datetime_resolution = datetime_resolution

    @property
    def timedelta_resolution(self) -> Optional["pandas.Timedelta"]:  # noqa F821 # type: ignore
        return self._timedelta_resolution

    @timedelta_resolution.setter
    def timedelta_resolution(self, value: Union[str, "pandas.Timedelta", None]):  # noqa F821 # type: ignore
        self._timedelta_resolution = self._get_resolution(value)

    @property
    def datetime_resolution(self) -> Optional["pandas.Timedelta"]:  # noqa F821 # type: ignore
        return self._datetime_resolution

    @datetime_resolution.setter
    def datetime_resolution(self, value: Union[str, "pandas.Timedelta", None]):  # noqa F821 # type: ignore
        self._datetime_resolution = self._get_resolution(value)

    def fit(self, X: "pandas.DataFrame", y=None) -> "DTypeTransformer":  # noqa F821 # type: ignore
        _ensure_dataframe(X, "DTypeTransformer.fit")

        transformers = {}  # maps names to pairs `(transformer: Optional[BaseEstimator], columns: list)`
        for c in X.columns:
            if X[c].dtype.name == "category":
                name = "cat"
            elif X[c].dtype.kind == "b":
                name = "bool"
            elif X[c].dtype.kind == "O":
                name = "obj"
            elif X[c].dtype.kind == "M":
                name = "datetime"
            elif X[c].dtype.kind == "m":
                name = "timedelta"
            elif X[c].dtype.kind in "uif":
                name = "num"
            else:
                name = "default"
            trans, name = self.get_transformer(name)

            if trans == "drop":
                continue
            elif trans == "passthrough":
                trans = None
            try:
                _, cols = transformers[name]
                cols.append(c)
            except KeyError:
                transformers[name] = (trans, [c])

        X, _ = self._convert_temporal(X)

        # convert to list of triples for consistency with ColumnTransformer
        self.transformers_ = [
            (
                name,
                trans if trans is None else (trans() if type(trans) is type else clone(trans)).fit(X[cols]),
                cols,
            )
            for name, (trans, cols) in transformers.items()
        ]

        return self

    def transform(self, X: "pandas.DataFrame") -> Union["pandas.DataFrame", np.ndarray]:  # noqa F821 # type: ignore
        _ensure_dataframe(X, "DTypeTransformer.transform")

        check_is_fitted(self)
        self._validate_input(X)

        X, _ = self._convert_temporal(X)

        out = []
        all_df = True
        for _, trans, cols in self.transformers_:
            if trans is None:
                # passthrough
                X_trans = X[cols]
            else:
                X_trans = trans.transform(X[cols])
                X_trans, X_trans_df = self._postproc(X_trans, trans, cols, X.index)
                if X_trans_df is not None:
                    X_trans = X_trans_df
            all_df = all_df and isinstance(X_trans, self._pd.DataFrame)
            out.append(X_trans)

        if all_df:
            # return DataFrame
            return self._pd.concat(out, axis=1, sort=False)
        else:
            # return array
            arrs = [X_trans.values if isinstance(X_trans, self._pd.DataFrame) else X_trans for X_trans in out]
            return np.hstack(arrs)

    def set_output(self, *args, **kwargs):
        # like ColumnTransformer, we set the output of both fitted and unfitted transformers
        for t in [
            self.num,
            self.cat,
            self.bool,
            self.timedelta,
            self.datetime,
            self.obj,
            self.default,
        ] + [t for _, t, _ in getattr(self, "transformers_", [])]:
            if type(t) is not type and t is not None and not isinstance(t, str):
                t.set_output(*args, **kwargs)
        return self

    def _validate_input(self, df):
        diff = [c for _, _, cols in self.transformers_ for c in cols if c not in df.columns]
        if diff:
            raise ValueError("X lacks columns " + str(diff))

    def get_transformer(self, name: str):
        visited = [name]
        spec = getattr(self, name)
        while True:
            if spec is None:
                spec = "default"

            if isinstance(spec, str):
                if spec in ("drop", "passthrough"):
                    return (spec, name)

                name = spec
                try:
                    spec = getattr(self, spec)
                except AttributeError:
                    raise ValueError(
                        'Transformation must be BaseEstimator, None, "default", "num", "cat", "bool",'
                        ' "timedelta", "datetime", "obj", "passthrough" or "drop", but got "{}"'.format(name)
                    )

                try:
                    i = visited.index(name)
                except ValueError:
                    visited.append(name)
                else:
                    raise ValueError("Recursive transformation specification: " + " -> ".join(visited[i:] + [name]))
            else:
                return (spec, name)

    def _convert_temporal(self, df) -> Tuple["pandas.DataFrame", bool]:  # noqa F821 # type: ignore
        copied = False
        if self._timedelta_resolution is not None:
            for c in df.columns:
                if df[c].dtype.kind == "m":
                    if not copied:
                        df = df.copy()
                        copied = True
                    df[c] = df[c] / self._timedelta_resolution
        if self._datetime_resolution is not None:
            for c in df.columns:
                if df[c].dtype.kind == "M":
                    if not copied:
                        df = df.copy()
                        copied = True
                    df[c] = (df[c] - self._pd.Timestamp(0)) / self._datetime_resolution

        return df, copied

    def _postproc(self, X, trans, cols: list, index: "pandas.Index"):  # noqa F821 # type: ignore
        df = None
        if hasattr(X, "toarray"):
            X = X.toarray()
        if isinstance(X, self._pd.DataFrame):
            assert (X.index == index).all()
            df = X
        else:
            if isinstance(trans, skl_preprocessing.KBinsDiscretizer):
                if (
                    trans.encode in ("onehot", "onehot-dense")
                    and self.num_transformer_.n_bins_.sum() == X.shape[1]
                    and len(trans.n_bins_) == len(cols)
                ):
                    columns = [f"{col}_{b}" for col, n in zip(cols, trans.n_bins_) for b in range(n)]
                else:
                    columns = []
            elif isinstance(trans, skl_preprocessing.OneHotEncoder):
                if trans.drop_idx_ is None:
                    features_out = trans.categories_
                else:
                    features_out = []
                    for i, categories in enumerate(trans.categories_):
                        j = trans.drop_idx_[i]
                        if j is None:
                            features_out.append(categories)
                        elif j == 0:
                            features_out.append(categories[1:])
                        elif j + 1 == len(categories):
                            features_out.append(categories[:-1])
                        else:
                            features_out.append(np.concatenate([categories[:j], categories[j + 1 :]]))
                if len(cols) == len(features_out) and X.shape[1] == sum(len(f) for f in features_out):
                    columns = [f"{col}_{f}" for col, feats in zip(cols, features_out) for f in feats]
                else:
                    columns = []
            else:
                columns = cols

            if len(columns) == X.shape[1]:
                df = self._pd.DataFrame(index=index, columns=columns, data=X)

        return X, df

    def _get_resolution(self, spec) -> Optional["pandas.Timedelta"]:  # noqa F821 # type: ignore
        if isinstance(spec, str):
            if spec in ("ns", "us", "ms", "s", "m", "h", "d", "w"):
                return self._pd.Timedelta(1, unit=spec)
            elif spec in ("y", "Y"):
                return self._pd.Timedelta(365.2525, unit="d")
            elif len(spec) > 1 and spec[-1] in ("y", "Y"):
                try:
                    y = float(spec[:-1])
                except ValueError:
                    return self._pd.to_timedelta(spec)
                else:
                    return self._pd.Timedelta(y * 365.2525, unit="d")
            else:
                return self._pd.to_timedelta(spec)
        return spec


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


def ordinal_encoder(dtype=np.float64, **kwargs) -> DTypeTransformer:
    """Create a transformation for ordinal-encoding categorical features, while keeping other features unchanged.

    Parameters
    ----------
    dtype
        Data type of ordinal encoding. Passed to `sklearn.preprocessing.OrdinalEncoder`.
    **kwargs
        Keyword arguments passed to DTypeTransformer, most notably `num`, `bool` etc. for specifying how to treat
        non-categorical columns. `cat` cannot be specified.

    Returns
    -------
    DTypeTransformer instance that can be used for ordinal-encoding categorical columns in pandas DataFrames.

    See Also
    --------
    ordinal_encode
    sklearn.preprocessing.OrdinalEncoder
    """
    return DTypeTransformer(cat=skl_preprocessing.OrdinalEncoder(dtype=dtype), **kwargs)


def ordinal_encode(
    X: "pandas.DataFrame",  # noqa F821 # type: ignore
    dtype=np.float64,
    output: str = "default",
    **kwargs,
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
    output : str, default="default"
        Desired output type, either "default" (Numpy array) or "pandas" (pandas DataFrame).
    **kwargs
        Keyword arguments passed to DTypeTransformer, most notably `num`, `bool` etc. for specifying how to treat
        non-categorical columns. `cat` cannot be specified.

    Returns
    -------
    Transformed input, either a DataFrame or an array, depending on `output`.

    See Also
    --------
    ordinal_encoder
    sklearn.preprocessing.OrdinalEncoder
    """
    transformer = ordinal_encoder(dtype=dtype, **kwargs)
    _try_set_output(transformer, output)
    return transformer.fit_transform(X)


def one_hot_encoder(
    drop_na: bool = False, drop=None, dtype=np.float64, handle_unknown: Optional[str] = None, **kwargs
) -> DTypeTransformer:
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
    **kwargs
        Keyword arguments passed to DTypeTransformer, most notably `num`, `bool` etc. for specifying how to treat
        non-categorical columns. `cat` cannot be specified.

    Returns
    -------
    DTypeTransformer instance that can be used for one-hot-encoding categorical columns in pandas DataFrames.

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
        ohe_kwargs = dict(sparse=False)
    else:
        # >=1.2
        ohe_kwargs = dict(sparse_output=False)
    return DTypeTransformer(
        cat=OneHotEncoder(drop_na=drop_na, drop=drop, dtype=dtype, handle_unknown=handle_unknown, **ohe_kwargs),
        **kwargs,
    )


def one_hot_encode(
    X: "pandas.DataFrame",  # noqa F821 # type: ignore
    drop_na: bool = False,
    drop=None,
    dtype=np.float64,
    handle_unknown: Optional[str] = None,
    output: str = "default",
    **kwargs,
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
    output : str, default="default"
        Desired output type, either "default" (Numpy array) or "pandas" (pandas DataFrame).
    **kwargs
        Keyword arguments passed to DTypeTransformer, most notably `num`, `bool` etc. for specifying how to treat
        non-categorical columns. `cat` cannot be specified.

    Returns
    -------
    Transformed input, either a DataFrame or an array, depending on `output`.

    See Also
    --------
    one_hot_encoder
    OneHotEncoder
    """
    transformer = one_hot_encoder(drop_na=drop_na, drop=drop, dtype=dtype, handle_unknown=handle_unknown, **kwargs)
    _try_set_output(transformer, output)
    return transformer.fit_transform(X)


def k_bins_discretizer(
    n_bins: int = 5,
    encode: str = "onehot",
    strategy: str = "quantile",
    timedelta: str = "num",
    **kwargs,
) -> DTypeTransformer:
    """Create a transformation for k-bins-discretizing numerical features, while keeping other features unchanged.

    Parameters
    ----------
    n_bins : int, default=5
        Number of bins to produce. Passed to `sklearn.preprocessing.KBinsDiscretizer`.
    encode : str, default="onehot"
        Method used to encode the transformed result. Passed to `sklearn.preprocessing.KBinsDiscretizer`.
    strategy : str, default="quantile"
        Strategy used to define the widths of the bins. Passed to `sklearn.preprocessing.KBinsDiscretizer`.
    timedelta : str, default="num"
        How to treat timedelta features.
    **kwargs
        Keyword arguments passed to DTypeTransformer, most notably `cat`, `bool` etc. for specifying how to treat
        non-numerical columns. `num` cannot be specified.

    Returns
    -------
    DTypeTransformer instance that can be used for k-bins-discretizing numerical columns in pandas DataFrames.

    See Also
    --------
    k_bins_discretize
    sklearn.preprocessing.KBinsDiscretizer
    """
    return DTypeTransformer(
        num=skl_preprocessing.KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy),
        timedelta=timedelta,
        **kwargs,
    )


def k_bins_discretize(
    X: "pandas.DataFrame",  # noqa F821 # type: ignore
    n_bins: int = 5,
    encode: str = "onehot",
    strategy: str = "quantile",
    output: str = "default",
    timedelta: str = "num",
    **kwargs,
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
    output : str, default="default"
        Desired output type, either "default" (Numpy array) or "pandas" (pandas DataFrame).
    timedelta : str, default="num"
        How to handle timedelta features.
    **kwargs
        Keyword arguments passed to DTypeTransformer, most notably `cat`, `bool` etc. for specifying how to treat
        non-numerical columns. `num` cannot be specified.

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
    transformer = k_bins_discretizer(n_bins=n_bins, encode=encode, strategy=strategy, timedelta=timedelta, **kwargs)
    _try_set_output(transformer, output)
    return transformer.fit_transform(X)


def binarizer(threshold: float = 0, **kwargs) -> DTypeTransformer:
    """Create a transformation for binarizing numerical features, while keeping other features unchanged.

    Parameters
    ----------
    threshold : float, default=0
        Feature values below or equal to this are replaced by 0, above it by 1.
        Passed to `sklearn.preprocessing.Binarizer`.
    **kwargs
        Keyword arguments passed to DTypeTransformer, most notably `cat`, `bool` etc. for specifying how to treat
        non-numerical columns. `num` cannot be specified.

    Returns
    -------
    DTypeTransformer instance that can be used for binarizing numerical columns in pandas DataFrames.

    See Also
    --------
    binarize
    sklearn.preprocessing.Binarizer
    """
    return DTypeTransformer(num=skl_preprocessing.Binarizer(threshold=threshold), **kwargs)


def binarize(
    X: "pandas.DataFrame",  # noqa F821 # type: ignore
    threshold: float = 0,
    output: str = "default",
    **kwargs,
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
    output : str, default="default"
        Desired output type, either "default" (Numpy array) or "pandas" (pandas DataFrame).
    **kwargs
        Keyword arguments passed to DTypeTransformer, most notably `cat`, `bool` etc. for specifying how to treat
        non-numerical columns. `num` cannot be specified.

    Returns
    -------
    Transformed input, either a DataFrame or an array, depending on `output`.

    See Also
    --------
    binarizer
    sklearn.preprocessing.Binarizer
    """
    transformer = binarizer(threshold=threshold, **kwargs)
    _try_set_output(transformer, output)
    return transformer.fit_transform(X)


def scaler(
    strategy: str = "standard",
    cat=None,
    bool=None,
    timedelta="num",
    datetime=None,
    obj=None,
    default="passthrough",
    timedelta_resolution=None,
    datetime_resolution=None,
    fit_bool=None,
    **kwargs,
) -> DTypeTransformer:
    """Create a transformation for scaling numerical features, while keeping other features unchanged.

    Parameters
    ----------
    strategy : str, default="standard"
        Strategy used to scale numerical data:
        * "standard": Scale data to have zero mean and unit variance, using `sklearn.preprocessing.StandardScaler`.
        * "robust": Scale data using statistics that are robust to outliers, using `sklearn.preprocessing.RobustScaler`.
        * "minmax": Scale data to have zero minimum and unit maximum, using `sklearn.preprocessing.MinMaxScaler`.
        * "maxabs": Scale data to have a maximum absolute value of 1, using `sklearn.preprocessing.MaxAbsScaler`.

    cat : optional
        How to handle categorical features.
    bool : optional
        How to handle boolean features.
    timedelta : default="num"
        How to handle timedelta features.
    datetime : optional
        How to handle datetime features.
    obj : optional
        How to handle object features.
    default : default="passthrough"
        How to handle features for which no transformation is specified elsewhere.
    timedelta_resolution : str | pandas.Timedelta, optional
        Timedelta resolution. If None and `timedelta` is set to "num" (either explicitly or implicitly), the resolution
        is automatically set to "s".
    datetime_resolution : str | pandas.Timedelta, optional
        Datetime resolution. If None and `datetime` is set to "num" (either explicitly or implicitly), the resolution
        is automatically set to "s".
    **kwargs
        Additional keyword arguments passed to the underlying scikit-learn scaler.

    Returns
    -------
    DTypeTransformer instance that can be used for scaling numerical columns in pandas DataFrames.

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
            'Scaling strategy must be one of "standard", "robust", "minmax" or "maxabs", but got "{}"'.format(strategy)
        )
    trans = DTypeTransformer(
        num=num_transformer,
        cat=cat,
        bool=bool,
        timedelta=timedelta,
        datetime=datetime,
        obj=obj,
        default=default,
        timedelta_resolution=timedelta_resolution,
        datetime_resolution=datetime_resolution,
    )
    if timedelta_resolution is None and trans.get_transformer("timedelta")[1] == "num":
        # we have to pass some resolution, but it does not matter which
        trans.timedelta_resolution = "s"
    if datetime_resolution is None and trans.get_transformer("datetime")[1] == "num":
        # we have to pass some resolution, but it does not matter which
        trans.datetime_resolution = "s"
    return trans


def scale(
    X: "pandas.DataFrame",  # noqa F821 # type: ignore
    strategy: str = "standard",
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

    output : str, default="default"
        Desired output type, either "default" (Numpy array) or "pandas" (pandas DataFrame).
    **kwargs
        Additional keyword arguments passed to `scaler()`.

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
    transformer = scaler(strategy=strategy, **kwargs)
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
