#  Copyright (c) 2025. RISC Software GmbH.
#  All rights reserved.

import sys
import warnings
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
from sklearn import metrics as skl_metrics
from sklearn.utils.multiclass import type_of_target

from .bootstrapping import Bootstrapping
from .util import to_num_array, to_num_arrays


class _OperatorBase:
    @classmethod
    def op_name(cls) -> str:
        return cls.__name__.lstrip("_")

    @classmethod
    def get_function_name(cls, func) -> str:
        return getattr(func, "__name__", None) or repr(func)

    def __init__(self, func):
        if not callable(func):
            raise ValueError(f"Function must be callable, but found {func}")
        self._func = func

    @property
    def base_function(self):
        return self._func

    @property
    def __name__(self) -> str:
        return self.op_name() + "(" + self.get_function_name(self._func) + ")"

    def __repr__(self) -> str:
        return self.__name__

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class averageable(_OperatorBase):  # noqa
    """Return an averageable variant of a given metric, i.e., a function that accepts parameter `average` with possible
    values None, "binary", "micro", "macro", "weighted", "samples" and , optionally, "global". Apart from "binary" and
    "global", averaging is taken care of by the new metric; the original metric only needs to handle binary
    classification tasks.

    Parameters
    ----------
    func : callable
        Metric to make averageable, callable that accepts `y_true` and `y_pred` and returns a scalar value.
    accepts_global : bool, optional
        What to do if `average` is set to "global": if true, `func` is simply called on the provided arguments;
        otherwise, a ValueError is raised.

    See Also
    --------
    no_average
    micro_average
    macro_average
    weighted_average
    samples_average
    """

    def __init__(self, func, accepts_global: bool = False):
        super(averageable, self).__init__(func)
        self._accepts_global = accepts_global

    @property
    def accepts_global(self) -> bool:
        return self._accepts_global

    def __call__(self, y_true, y_pred, **kwargs):
        try:
            average = kwargs.pop("average")
        except KeyError:
            # simply call function
            return self._func(y_true, y_pred, **kwargs)
        else:
            if average is None:
                return no_average.compute(self._func, y_true, y_pred, **kwargs)
            elif average == "binary":
                # pass `average="binary"` to function
                return self._func(y_true, y_pred, average="binary", **kwargs)
            elif average == "micro":
                return micro_average.compute(self._func, y_true, y_pred, **kwargs)
            elif average == "macro":
                return macro_average.compute(self._func, y_true, y_pred, **kwargs)
            elif average == "weighted":
                return weighted_average.compute(self._func, y_true, y_pred, **kwargs)
            elif average == "samples":
                return samples_average.compute(self._func, y_true, y_pred, **kwargs)
            elif self._accepts_global:
                if average == "global":
                    # simply call function
                    return self._func(y_true, y_pred, **kwargs)
                else:
                    raise ValueError(
                        'average has to be one of None, "global", "micro", "macro", "weighted" or "samples",'
                        ' but got "{}"'.format(average)
                    )
            else:
                raise ValueError(
                    'average has to be one of None, "micro", "macro", "weighted" or "samples", but got "{}"'.format(
                        average
                    )
                )


class no_average(_OperatorBase):  # noqa
    """Return the "no-average" variant of a given classification metric, i.e., a new metric that returns class- or
    label-wise results, without any averaging.

    Parameters
    ----------
    func : callable
        Base metric, callable that accepts `y_true` and `y_pred` and returns a scalar value.

    Notes
    -----
    This corresponds to `metric(..., average=None)`.
    """

    # in case of multilabel tasks, this implementation also works with class probabilities

    def __call__(self, *args, **kwargs):
        return self.compute(self._func, *args, **kwargs)

    @classmethod
    def compute(cls, func, y_true, y_pred, **kwargs):
        y_true, y_pred = to_num_arrays(y_true, y_pred, arg_names=("y_true", "y_pred"), rank=(1, 2))
        if y_true.ndim == 1:
            if any(type_of_target(x) not in ("binary", "multiclass") for x in (y_true, y_pred)):
                raise ValueError("Averaging is not defined for binary/multiclass tasks with class probabilities.")
            classes = np.unique(y_true)
            return np.array([func(y_true == c, y_pred == c, **kwargs) for c in classes])
        else:
            return np.array([func(y_true[:, i], y_pred[:, i], **kwargs) for i in range(y_true.shape[1])])


class micro_average(_OperatorBase):  # noqa
    """Return the micro-averaged variant of a given classification metric, i.e., a new metric that returns
    micro-averaged results.

    Micro-averaging amounts to counting the total number of true and false positives and negatives across all classes,
    and computing the metric value wrt. these numbers.

    Parameters
    ----------
    func : callable
        Base metric, callable that accepts `y_true` and `y_pred` and returns a scalar value.

    Notes
    -----
    This corresponds to `metric(..., average="micro")`.
    """

    # in case of multilabel tasks, this implementation also works with class probabilities

    def __call__(self, *args, **kwargs):
        return self.compute(self._func, *args, **kwargs)

    @classmethod
    def compute(cls, func, y_true, y_pred, sample_weight: Optional[np.ndarray] = None, **kwargs):
        y_true, y_pred = to_num_arrays(y_true, y_pred, arg_names=("y_true", "y_pred"), rank=(1, 2))
        if y_true.ndim == 1:
            if any(type_of_target(x) not in ("binary", "multiclass") for x in (y_true, y_pred)):
                raise ValueError("Micro-averaging is not defined for binary/multiclass tasks with class probabilities.")
            classes = np.unique(y_true)
            y_true = y_true[..., np.newaxis] == classes
            y_pred = y_pred[..., np.newaxis] == classes

        if sample_weight is not None:
            sample_weight = np.repeat(sample_weight, y_true.shape[1])

        return func(y_true.ravel(), y_pred.ravel(), sample_weight=sample_weight, **kwargs)


class macro_average(_OperatorBase):  # noqa
    """Return the macro-averaged variant of a given classification metric, i.e., a new metric that returns
    macro-averaged results.

    Macro-averaging amounts to computing the metric value for each class/label individually, and then returning the
    unweighted mean of these values.

    Parameters
    ----------
    func : callable
        Base metric, callable that accepts `y_true` and `y_pred` and returns a scalar value.

    Notes
    -----
    This corresponds to `metric(..., average="macro")`.
    """

    # in case of multilabel tasks, this implementation also works with class probabilities

    def __call__(self, *args, **kwargs):
        return self.compute(self._func, *args, **kwargs)

    @classmethod
    def compute(cls, func, y_true, y_pred, **kwargs):
        return no_average.compute(func, y_true, y_pred, **kwargs).mean()


class weighted_average(_OperatorBase):  # noqa
    """Return the weighted-averaged variant of a given classification metric, i.e., a new metric that returns
    weighted-averaged results.

    Weighted-averaging amounts to computing the metric value for each class/label individually, and then returning the
    weighted mean of these values. Weights correspond to class/label support.

    Parameters
    ----------
    func : callable
        Base metric, callable that accepts `y_true` and `y_pred` and returns a scalar value.

    Notes
    -----
    This corresponds to `metric(..., average="weighted")`.
    """

    # in case of multilabel tasks, this implementation also works with class probabilities

    def __call__(self, *args, **kwargs):
        return self.compute(self._func, *args, **kwargs)

    @classmethod
    def compute(cls, func, y_true, y_pred, sample_weight: Optional[np.ndarray] = None, **kwargs):
        y_true, y_pred = to_num_arrays(y_true, y_pred, arg_names=("y_true", "y_pred"), rank=(1, 2))
        if y_true.ndim == 1:
            if any(type_of_target(x) not in ("binary", "multiclass") for x in (y_true, y_pred)):
                raise ValueError(
                    "Weighted-averaging is not defined for binary/multiclass tasks with class probabilities."
                )
            if sample_weight is None:
                classes, weights = np.unique(y_true, return_counts=True)
            else:
                classes = np.unique(y_true)
                weights = np.zeros((len(classes),), dtype=np.float32)
                for i, c in enumerate(classes):
                    weights[i] = ((y_true == c) * sample_weight).sum()
            return (
                sum(
                    w * func(y_true == c, y_pred == c, sample_weight=sample_weight, **kwargs)
                    for c, w in zip(classes, weights)
                )
                / weights.sum()
            )
        else:
            if sample_weight is None:
                weights = (y_true > 0).sum(axis=0)
            else:
                weights = ((y_true > 0) * sample_weight[..., np.newaxis]).sum(axis=0)
            return (
                sum(
                    w * func(y_true[:, i], y_pred[:, i], sample_weight=sample_weight, **kwargs)
                    for i, w in enumerate(weights)
                )
                / weights.sum()
            )


class samples_average(_OperatorBase):  # noqa
    """Return the samples-averaged variant of a given classification metric, i.e., a new metric that returns
    samples-averaged results.

    Samples-averaging amounts to computing the metric value for each sample individually, and then returning the
    (weighted) mean of these values.

    Parameters
    ----------
    func : callable
        Base metric, callable that accepts `y_true` and `y_pred` and returns a scalar value.

    Notes
    -----
    This corresponds to `metric(..., average="samples")`, and is only defined for multilabel tasks.
    """

    # only defined for multilabel tasks, and works for class probabilities and class indicators alike

    def __call__(self, *args, **kwargs):
        return self.compute(self._func, *args, **kwargs)

    @classmethod
    def compute(cls, func, y_true, y_pred, sample_weight: Optional[np.ndarray] = None, **kwargs):
        y_true, y_pred = to_num_arrays(y_true, y_pred, arg_names=("y_true", "y_pred"), rank=(2,))
        return np.average([func(y_true[i], y_pred[i], **kwargs) for i in range(y_true.shape[0])], weights=sample_weight)


class _NegativeFunction(_OperatorBase):
    def __call__(self, *args, **kwargs):
        return -self._func(*args, **kwargs)

    @property
    def __name__(self) -> str:
        return "-" + self.get_function_name(self._func)

    @classmethod
    def make(cls, func):
        if isinstance(func, _NegativeFunction):
            return func.base_function
        else:
            return cls(func)


def to_score(func, errors: str = "ignore"):
    """Convenience function for converting a metric into a (possibly different) metric that returns scores (i.e.,
    higher values correspond to better results). That means, if the given metric returns scores already, it is returned
    unchanged. Otherwise, it is negated.

    Parameters
    ----------
    func : callable
        The metric to convert, e.g., `accuracy`, `balanced_accuracy`, etc. Note that in case of classification metrics,
        both thresholded and non-thresholded metrics are accepted.
    errors : str, default="ignore"
        What to do if the polarity of `func` cannot be determined:

        * "ignore": return `func`.
        * "negate": return `-func`.
        * "raise": raise a ValueError.

    Returns
    -------
    Either `func` itself or `-func`.
    """

    func_th = maybe_thresholded(func, threshold=0.5)
    y = np.array([0, 0, 1], dtype=np.float32)
    y_hat = y * 0.8 + 0.1  # some metrics (e.g., log_loss) don't accept exact 0. and 1.
    try:
        perfect = func_th(y, y_hat)
        assert not np.isnan(perfect)  # treat NaN like an exception
    except:  # noqa
        y = np.stack([y, 1 - y], axis=1)
        y_hat = np.stack([y_hat, 1 - y_hat], axis=1)
        try:
            perfect = func_th(y, y_hat)
            assert not np.isnan(perfect)  # treat NaN like an exception
        except:  # noqa
            y = y[:, 0]
            y[1] = 1.0
            y[2] = 2.0
            y_hat = np.eye(3, dtype=np.float32) * 0.8 + 0.1
            try:
                perfect = func_th(y, y_hat)
                assert not np.isnan(perfect)  # treat NaN like an exception
            except:  # noqa
                if errors == "ignore":
                    return func
                elif errors == "negate":
                    return _NegativeFunction.make(func)
                else:
                    raise ValueError("The polarity of {} cannot be determined.".format(func))

    worse = func_th(y, y_hat[::-1])
    if perfect < worse:
        return _NegativeFunction.make(func)
    else:
        return func


def get(name):
    """Retrieve a metric function given by its name.

    Parameters
    ----------
    name : str | callable
        The name of the requested metric function. It must be of the form

            "`name` [@ `threshold`] [(`agg` : `n_reps`)]"

        where `name` is the name of a recognized metric and the `threshold` and `agg`/`n_reps` parts are optional.

        If `threshold` is specified, `name` must be the name of a thresholded classification metric (e.g., "accuracy")
        and `threshold` must be either a specific numerical threshold or the name of a thresholding strategy; see
        function `thresholded()` for details.

        If `agg` and `n_reps` are specified, the bootstrapped metric with `n_reps` repetitions and aggregation `agg`
        is returned.

        If both a threshold and bootstrapping are specified, the threshold must be specified first.

        Note that some synonyms are recognized as well, most notably "precision" for "positive_predictive_value" and
        "recall" for "sensitivity".

    Returns
    -------
    Metric function (callable).
    """
    if isinstance(name, str):
        s = name.split("(", 1)
        if len(s) == 1:
            s = name.split("@", 1)
            if len(s) == 1:
                name = s[0].strip()
                if name == "precision" or name.startswith("precision_"):
                    name = "positive_predictive_value" + name[9:]
                elif name == "recall" or name.startswith("recall_"):
                    name = "sensitivity" + name[6:]
                elif name == "youden_index" or name.startswith("youden_index_"):
                    name = "informedness" + name[12:]
                elif name == "mean_average_precision":
                    name = "average_precision_macro"
                return getattr(sys.modules[__name__], name)
            else:
                fn = get(s[0])
                th = s[1].strip()
                try:
                    th = float(th)
                except ValueError:
                    # assume `th` is the name of a thresholding strategy => pass on to `thresholded()`
                    pass
                return thresholded(fn, threshold=th)
        elif s[1].endswith(")"):
            b = s[1][:-1].split(":", 1)
            if len(b) == 2:
                agg = b[0].strip()
                n_reps = int(b[1].strip())
            else:
                raise ValueError('Invalid bootstrapping specification: "({}"'.format(s[1]))
            fn = get(s[0])
            return bootstrapped(fn, n_repetitions=n_reps, agg=agg)
        else:
            raise ValueError('Invalid metric specification: "{}"'.format(s))
    return name


class _bootstrapped(_OperatorBase):  # noqa
    def __init__(
        self,
        func,
        n_repetitions: int = 100,
        agg="mean",
        seed=None,
        replace: bool = True,
        size: Union[int, float] = 1.0,
        **kwargs,
    ):
        super(_bootstrapped, self).__init__(func)
        self._n_repetitions = n_repetitions
        self._agg = agg
        self._seed = seed
        self._replace = replace
        self._size = size
        self._kwargs = kwargs

    @property
    def n_repetitions(self) -> int:
        return self._n_repetitions

    @property
    def agg(self):
        return self._agg

    @property
    def seed(self):
        return self._seed

    @property
    def replace(self) -> bool:
        return self._replace

    @property
    def size(self):
        return self._size

    @property
    def kwargs(self) -> dict:
        return self._kwargs

    def __call__(self, y_true, y_hat, sample_weight=None, **kwargs):
        kwargs.update(self._kwargs)
        if kwargs:
            _func = partial(self._func, **kwargs)
        else:
            _func = self._func
        bs = Bootstrapping(
            y_true,
            y_hat,
            kwargs=None if sample_weight is None else dict(sample_weight=sample_weight),
            fn=_func,
            seed=self._seed,
            replace=self._replace,
            size=self._size,
        )
        bs.run(n_repetitions=self._n_repetitions)
        return bs.agg(self._agg)

    @classmethod
    def make(
        cls,
        func,
        n_repetitions: int = 100,
        agg="mean",
        seed=None,
        replace: bool = True,
        size: Union[int, float] = 1.0,
        **kwargs,
    ):
        """Convenience function for converting a metric into its bootstrapped version.

        Parameters
        ----------
        func : callable
            The metric to convert, e.g., `roc_auc`, `accuracy`, `mean_squared_error`, etc.
        n_repetitions : int, default=100
            Number of bootstrapping repetitions to perform. If 0, `func` is returned unchanged.
        agg : str | callable, default='mean'
            Aggregation to compute of bootstrapping results.
        seed : int, optional
            Random seed.
        replace : bool, default=True
            Whether to resample with replacement. If False, this does not actually correspond to bootstrapping.
        size : int | float, default=1.
            The size of the resampled data. If <= 1, it is multiplied with the number of samples in the given data.
            Bootstrapping normally assumes that resampled data have the same number of samples as the original data, so
            this parameter should be set to 1.
        **kwargs
            Additional keyword arguments that are passed to `func` upon application. Note that only arguments that do
            not need to be resampled can be passed here; in particular, this excludes `sample_weight`.

        Returns
        -------
        New metric that, when applied to `y_true` and `y_hat`, resamples the data, evaluates the metric on each
        resample, and returns some aggregation (typically average) of the results thus obtained.
        """
        if n_repetitions <= 0:
            if kwargs:
                return partial(func, **kwargs)
            else:
                return func

        return cls(
            func,
            n_repetitions=n_repetitions,
            agg=agg,
            seed=seed,
            replace=replace,
            size=size,
            **kwargs,
        )


bootstrapped = _bootstrapped.make  # convenient alias


# regression
r2 = skl_metrics.r2_score
mean_absolute_error = skl_metrics.mean_absolute_error
mean_squared_error = skl_metrics.mean_squared_error
if hasattr(skl_metrics, "root_mean_squared_error"):
    root_mean_squared_error = skl_metrics.root_mean_squared_error
else:
    root_mean_squared_error = partial(skl_metrics.mean_squared_error, squared=False)
mean_squared_log_error = skl_metrics.mean_squared_log_error
median_absolute_error = skl_metrics.median_absolute_error
mean_absolute_percentage_error = skl_metrics.mean_absolute_percentage_error
explained_variance = skl_metrics.explained_variance_score
mean_tweedie_deviance = skl_metrics.mean_tweedie_deviance
mean_poisson_deviance = skl_metrics.mean_poisson_deviance
mean_gamma_deviance = skl_metrics.mean_gamma_deviance


def max_error(y_true, y_pred, sample_weight=None):
    return skl_metrics.max_error(y_true, y_pred)


# classification with probabilities
roc_auc = skl_metrics.roc_auc_score
roc_auc_micro = partial(roc_auc, average="micro")
roc_auc_macro = partial(roc_auc, average="macro")
roc_auc_samples = partial(roc_auc, average="samples")
roc_auc_weighted = partial(roc_auc, average="weighted")
roc_auc_ovr = partial(roc_auc, multi_class="ovr")
roc_auc_ovo = partial(roc_auc, multi_class="ovo")
roc_auc_ovr_weighted = partial(roc_auc, multi_class="ovr", average="weighted")
roc_auc_ovo_weighted = partial(roc_auc, multi_class="ovo", average="weighted")
average_precision = skl_metrics.average_precision_score
average_precision_micro = partial(average_precision, average="micro")
average_precision_macro = partial(average_precision, average="macro")
average_precision_samples = partial(average_precision, average="samples")
average_precision_weighted = partial(average_precision, average="weighted")
brier_loss = averageable(skl_metrics.brier_score_loss)  # not defined for multiclass
brier_loss_micro = micro_average(skl_metrics.brier_score_loss)
brier_loss_macro = macro_average(skl_metrics.brier_score_loss)
brier_loss_samples = samples_average(skl_metrics.brier_score_loss)
brier_loss_weighted = weighted_average(skl_metrics.brier_score_loss)
hinge_loss = averageable(skl_metrics.hinge_loss, accepts_global=True)
hinge_loss_micro = micro_average(skl_metrics.hinge_loss)
hinge_loss_macro = macro_average(skl_metrics.hinge_loss)
hinge_loss_samples = samples_average(skl_metrics.hinge_loss)
hinge_loss_weighted = weighted_average(skl_metrics.hinge_loss)
log_loss = averageable(skl_metrics.log_loss, accepts_global=True)
log_loss_micro = micro_average(skl_metrics.log_loss)
log_loss_macro = macro_average(skl_metrics.log_loss)
log_loss_samples = samples_average(skl_metrics.log_loss)
log_loss_weighted = weighted_average(skl_metrics.log_loss)


def _pr_auc(y_true, y_score, **kwargs) -> float:
    precision, recall, _ = skl_metrics.precision_recall_curve(y_true, y_score, **kwargs)
    return skl_metrics.auc(recall, precision)


pr_auc = averageable(_pr_auc)
pr_auc_micro = micro_average(_pr_auc)
pr_auc_macro = macro_average(_pr_auc)
pr_auc_samples = samples_average(_pr_auc)
pr_auc_weighted = weighted_average(_pr_auc)


def balance_score_threshold(y_true, y_score, sample_weight: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Compute the balance score and -threshold of a binary classification problem.

    Parameters
    ----------
    y_true : array-like
        Ground truth, with 0 representing the negative class and 1 representing the positive class. Must not contain
        NaN.
    y_score : array-like
        Predicted scores, i.e., the higher a score the more confident the model is that the sample belongs to the
        positive class. Range is arbitrary.
    sample_weight : array-like, optional
        Sample weights.

    Returns
    -------
    balance_score : float
        Sensitivity at `balance_threshold`, which by definition is approximately equal to specificity and can
        furthermore be shown to be approximately equal to accuracy and balanced accuracy, too.
    balance_threshold: float
        Decision threshold that minimizes the difference between sensitivity and specificity, i.e., it is defined as

        .. math::
            min_t |sensitivity(y_true, y_score >= t) - specificity(y_true, y_score >= t)|
    """
    y_true, y_score = to_num_arrays(y_true, y_score, arg_names=("y_true", "y_score"), rank=(1,))
    if sample_weight is not None:
        sample_weight = to_num_array(sample_weight, arg_name="sample_weight", rank=(1,))
    if sample_weight is not None and len(y_true) != len(sample_weight):
        raise ValueError(
            "Length of `sample_weight` differs from length of `y_true`: %i, %i" % (len(sample_weight), len(y_true))
        )
    elif len(y_true) == 0:
        return 0.0, 0.0

    pn = np.stack([(y_true > 0).astype(np.float32), (y_true < 1).astype(np.float32)], axis=1)  # (N, 2)
    if sample_weight is not None:
        pn *= sample_weight[..., np.newaxis]
    y_score, pn = _sort_cumsum_groupby_last(y_score, pn, ascending=False)
    # `y_score` is strictly decreasing with length `N0`, `pn` has shape `(N0, 2)`
    n_pos, n_neg = pn[-1]
    tp = pn[:, 0]
    tn = n_neg - pn[:, 1]
    diff = tp * n_neg - tn * n_pos  # monotonically increasing; constant 0 if `n_pos == 0` or `n_neg == 0`
    i = np.argmin(np.abs(diff))
    th = y_score[i]
    if i + 1 < len(y_score):  # take midpoint of `i`-th and nearest *smaller* threshold
        th += y_score[i + 1]
        th *= 0.5

    return ((tn[i] / n_neg) if n_pos == 0 else (tp[i] / n_pos)), th


def balance_score(y_true, y_score, sample_weight: Optional[np.ndarray] = None) -> float:
    return balance_score_threshold(y_true, y_score, sample_weight=sample_weight)[0]


def balance_threshold(y_true, y_score, sample_weight: Optional[np.ndarray] = None) -> float:
    return balance_score_threshold(y_true, y_score, sample_weight=sample_weight)[1]


def prevalence_score_threshold(y_true, y_score, sample_weight: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Compute the prevalence score and -threshold of a binary classification problem.

    Parameters
    ----------
    y_true : array-like
        Ground truth, with 0 representing the negative class and 1 representing the positive class. Must not contain
        NaN.
    y_score : array-like
        Predicted scores, i.e., the higher a score the more confident the model is that the sample belongs to the
        positive class. Range is arbitrary.
    sample_weight : array-like, optional
        Sample weights.

    Returns
    -------
    prevalence_score : float
        Sensitivity at `prevalence_threshold`, which can be shown to be approximately equal to positive predictive
        value and F1-score.
    prevalence_threshold : float
        Decision threshold that minimizes the difference between the number of positive samples in `y_true` (`m`) and
        the number of predicted positives. In other words, the threshold is set to the `m`-th largest value in
        `y_score`. If `sample_weight` is given, the threshold minimizes the difference between the total weight of all
        positive samples and the total weight of all samples predicted positive.
    """
    th = prevalence_threshold(y_true, y_score, sample_weight=sample_weight)
    return sensitivity(y_true, y_score >= th, sample_weight=sample_weight), th


def prevalence_score(y_true, y_score, sample_weight: Optional[np.ndarray] = None) -> float:
    return prevalence_score_threshold(y_true, y_score, sample_weight=sample_weight)[0]


def prevalence_threshold(y_true, y_score, sample_weight: Optional[np.ndarray] = None) -> float:
    y_true, y_score = to_num_arrays(y_true, y_score, arg_names=("y_true", "y_score"), rank=(1,))
    if sample_weight is not None:
        sample_weight = to_num_array(sample_weight, arg_name="sample_weight", rank=(1,))
    if sample_weight is not None and len(y_true) != len(sample_weight):
        raise ValueError(
            "Length of `sample_weight` differs from length of `y_true`: %i, %i" % (len(sample_weight), len(y_true))
        )
    elif len(y_true) == 0:
        return 0.0

    pw = np.stack(
        [(y_true > 0).astype(np.float32), np.ones(len(y_true), dtype=np.float32)],
        axis=1,
    )  # (N, 2)
    if sample_weight is not None:
        pw *= sample_weight[..., np.newaxis]
    y_score, pw = _sort_cumsum_groupby_last(y_score, pw, ascending=False)
    # `y_score` is strictly decreasing with length `N0`, `pn` has shape `(N0, 2)`
    i = np.argmin(np.abs(pw[:, 1] - pw[-1, 0]))
    th = y_score[i]
    if i + 1 < len(y_score):  # take midpoint of `i`-th and nearest *smaller* threshold
        th += y_score[i + 1]
        th *= 0.5
    return th


def zero_one_threshold(
    y_true,
    y_score,
    sample_weight: Optional[np.ndarray] = None,
    specificity_weight: float = 1.0,
) -> float:
    """Compute the threshold corresponding to the (0,1)-criterion [1] of a binary classification problem.
    Although a popular strategy for selecting decision thresholds, [1] advocates maximizing informedness (aka Youden
    index) instead, which is equivalent to maximizing balanced accuracy.

    Parameters
    ----------
    y_true : array-like
        Ground truth, with 0 representing the negative class and 1 representing the positive class. Must not contain
        NaN.
    y_score : array-like
        Predicted scores, i.e., the higher a score the more confident the model is that the sample belongs to the
        positive class. Range is arbitrary.
    sample_weight : array-like, optional
        Sample weights.
    specificity_weight : float, default=1.
        The relative weight of specificity wrt. sensitivity. 1 means that sensitivity and specificity are weighted
        equally, a value < 1 means that sensitivity is weighted stronger than specificity, and a value > 1 means that
        specificity is weighted stronger than sensitivity. See the formula below for details.

    Returns
    -------
    threshold : float
        Decision threshold that minimizes the Euclidean distance between the point `(0, 1)` and the point
        `(1 - specificity, sensitivity)`, i.e., arg min_t (1 - sensitivity(y_true, y_score >= t)) ** 2 +
        specificity_weight * (1 - specificity(y_true, y_score >= t)) ** 2

    References
    ----------
    .. [1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1444894/
    """
    y_true, y_score = to_num_arrays(y_true, y_score, arg_names=("y_true", "y_score"), rank=(1,))
    if sample_weight is not None:
        sample_weight = to_num_array(sample_weight, arg_name="sample_weight", rank=(1,))
    if sample_weight is not None and len(y_true) != len(sample_weight):
        raise ValueError(
            "Length of `sample_weight` differs from length of `y_true`: %i, %i" % (len(sample_weight), len(y_true))
        )
    elif len(y_true) == 0:
        return 0.0

    pn = np.stack([(y_true > 0).astype(np.float32), (y_true < 1).astype(np.float32)], axis=1)  # (N, 2)
    if sample_weight is not None:
        pn *= sample_weight[..., np.newaxis]
    y_score, pn = _sort_cumsum_groupby_last(y_score, pn, ascending=False)
    # `y_score` is strictly decreasing with length `N0`, `pn` has shape `(N0, 2)`
    n_pos, n_neg = pn[-1]
    fn = n_pos - pn[:, 0]
    fp = pn[:, 1]
    obj = (fn * n_neg) ** 2 + specificity_weight * (fp * n_pos) ** 2
    i = np.argmin(obj)
    th = y_score[i]
    if i + 1 < len(y_score):  # take midpoint of `i`-th and nearest *smaller* threshold
        th += y_score[i + 1]
        th *= 0.5
    return th


def argmax_score_threshold(
    func,
    y_true,
    y_score,
    sample_weight: Optional[np.ndarray] = None,
    discretize=100,
    **kwargs,
) -> Tuple[float, float]:
    """Compute the decision threshold that maximizes a given binary classification metric or callable.
    Since in most built-in classification metrics larger values indicate better results, there is no analogous
    `argmin_score_threshold()`.

    Parameters
    ----------
    func : callable
        The metric or function ot maximize. If a string, function `get()` is called on it.
    y_true : array-like
        Ground truth, with 0 representing the negative class and 1 representing the positive class. Must not contain
        NaN.
    y_score : array-like
        Predicted scores, i.e., the higher a score the more confident the model is that the sample belongs to the
        positive class. Range is arbitrary.
    sample_weight : array-like, optional
        Sample weights.
    discretize : int, default=100
        Discretization steps for limiting the number of calls to `func`. If None, no discretization happens, i.e., all
        unique values in `y_score` are tried.
    **kwargs
        Additional keyword arguments passed to `func`.

    Returns
    -------
    score : float
        Value of `func` at `threshold`.
    threshold : float
        Decision threshold that maximizes `func`, i.e.,

        .. math::
            arg max_t func(y_true, y_score >= t).
    """
    y_score = to_num_array(y_score, arg_name="y_score", rank=(1,))
    if isinstance(func, str):
        func = get(func)
    opt = -np.inf
    th = 0.0
    if discretize is None:
        thresholds = np.unique(y_score)
    else:
        thresholds = get_thresholds(y_score, n_max=discretize, sample_weight=sample_weight)
    for t in thresholds:
        res = func(y_true, y_score >= t, sample_weight=sample_weight, **kwargs)
        if res > opt:
            opt = res
            th = t
    return opt, th


def argmax_score(func, y_true, y_score, **kwargs) -> float:
    return argmax_score_threshold(func, y_true, y_score, **kwargs)[0]


def argmax_threshold(func, y_true, y_score, **kwargs) -> float:
    return argmax_score_threshold(func, y_true, y_score, **kwargs)[1]


def get_thresholding_strategy(name: str):
    """Retrieve a thresholding strategy for binary classification, given by its name.

    Parameters
    ----------
    name : str
        The name of the thresholding strategy, like "balance", "prevalence" or "zero_one".

    Returns
    -------
    Thresholding strategy (callable) that can be applied to `y_true`, `y_score` and `sample_weight`, and that returns
    a single scalar threshold.
    """
    if name == "balance":
        return balance_threshold
    elif name == "prevalence":
        return prevalence_threshold
    elif name == "zero_one":
        return zero_one_threshold
    elif name.startswith("zero_one(") and name[-1] == ")":
        return partial(zero_one_threshold, specificity_weight=float(name[9:-1]))
    elif name.startswith("argmax "):
        return partial(argmax_threshold, name[7:])
    else:
        raise ValueError(
            f'Unknown thresholding strategy "{name}".'
            ' Choose one of "balance", "prevalence", "zero_one" or "argmax <metric>"'
        )


def calibration_curve(
    y_true,
    y_score,
    sample_weight: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the calibration curve of a binary classification problem. The predicated class probabilities are binned
    and, for each bin, the fraction of positive samples is determined. These fractions can then be plotted against the
    midpoints of the respective bins. Ideally, the resulting curve will be monotonic increasing.

    Parameters
    ----------
    y_true : array-like
        Ground truth, array of shape `(n,)` with values among 0 and 1. Values must not be NaN.
    y_score : array-like
        Predicated probabilities of the positive class, array of shape `(n,)` with arbitrary non-NaN values; in
        particular, the values do not necessarily need to correspond to probabilities or confidences.
    sample_weight : array-like, optional
        Sample weight.
    thresholds : array-like, optional
        The thresholds used for binning `y_score`. If None, suitable thresholds are determined automatically.

    Returns
    -------
    fractions : ndarray
        Fractions of positive samples in each bin defined by `thresholds`, array of shape `(m - 1,)`. Note that the
        `i`-th bin corresponds to the half-open interval `[thresholds[i], thresholds[i + 1])` if `i < m - 2`, and to
        the closed interval `[thresholds[i], thresholds[i + 1]]` otherwise (in other words: the last bin is closed).
    thresholds : ndarray
        Thresholds, array of shape `(m,)`.
    """
    y_true, y_score = to_num_arrays(y_true, y_score, arg_names=("y_true", "y_score"), rank=(1,))
    if thresholds is None:
        thresholds = get_thresholds(y_score, n_max=40, add_half_one=False, sample_weight=sample_weight)
    else:
        thresholds = to_num_array(thresholds, arg_name="thresholds", rank=(1,))
    if len(thresholds) < 2:
        return np.empty((0,), dtype=np.float32), thresholds
    fractions = np.array(
        [y_true[(t1 <= y_score) & (y_score < t2)].mean() for t1, t2 in zip(thresholds[:-1], thresholds[1:])]
    )
    fractions[-1] = y_true[(thresholds[-2] <= y_score) & (y_score <= thresholds[-1])].mean()
    return fractions, np.asarray(thresholds)


def roc_pr_curve(
    y_true,
    y_score,
    *,
    pos_label: Union[int, str, None] = None,
    sample_weight: Optional[np.ndarray] = None,
    drop_intermediate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience function for computing ROC- and precision-recall curves simultaneously, with only one call to
    function `_binary_clf_curve()`.

    Parameters
    ----------
    y_true : array-like
        Same as in `sklearn.metrics.roc_curve()` and `sklearn.metrics.precision_recall_curve()`.
    y_score: array-like
        Same as in `sklearn.metrics.roc_curve()` and `sklearn.metrics.precision_recall_curve()`.
    pos_label : int | str, optional
        Same as in `sklearn.metrics.roc_curve()` and `sklearn.metrics.precision_recall_curve()`.
    sample_weight : array-like, optional
        Same as in `sklearn.metrics.roc_curve()` and `sklearn.metrics.precision_recall_curve()`.
    drop_intermediate : bool, default=True
        Same as in `sklearn.metrics.roc_curve()`.

    Returns
    -------
    6-tuple `(fpr, tpr, thresholds_roc, precision, recall, thresholds_pr)`, i.e., the concatenation of the return
    values of functions `sklearn.metrics.roc_curve()` and `sklearn.metrics.precision_recall_curve()`.

    See Also
    --------
    sklearn.metrics.roc_curve
    sklearn.metrics.precision_recall_curve

    Notes
    -----
    The output of this function may differ from the output of `sklearn.metrics.roc_curve` and
    `sklearn.metrics.precision_recall_curve`, because the implementation of the latter changed over time. For instance,
    early versions of scikit-learn set the first threshold in the output of `roc_curve` to 1 + the second threshold,
    whereas later this was changed to +inf. Similarly, early versions of `precision_recall_curve` only returned
    precision and recall until full recall was attained, whereas more recent versions return precision and recall for
    all thresholds.
    """

    fps, tps, thresholds = skl_metrics._ranking._binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )

    # ROC
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps_roc = fps[optimal_idxs]
        tps_roc = tps[optimal_idxs]
        thresholds_roc = thresholds[optimal_idxs]
    else:
        fps_roc = fps
        tps_roc = tps
        thresholds_roc = thresholds

    # Add an extra threshold position to make sure that the curve starts at (0, 0)
    tps_roc = np.r_[0, tps_roc]
    fps_roc = np.r_[0, fps_roc]

    if fps[-1] <= 0:
        warnings.warn(
            "No negative samples in y_true, false positive value should be meaningless",
            skl_metrics._ranking.UndefinedMetricWarning,
        )
        fpr = np.repeat(np.nan, fps_roc.shape)
    else:
        fpr = fps_roc / fps[-1]

    if tps[-1] <= 0:
        warnings.warn(
            "No positive samples in y_true,"
            " true positive value should be meaningless and recall is set to 1 for all thresholds",
            skl_metrics._ranking.UndefinedMetricWarning,
        )
        tpr = np.repeat(np.nan, tps_roc.shape)
        recall = np.ones_like(tps)
    else:
        tpr = tps_roc / tps[-1]
        recall = tps / tps[-1]

    # PR
    ps = tps + fps
    precision = np.divide(tps, ps, where=(ps != 0))

    # reverse the outputs so recall is decreasing
    sl = slice(None, None, -1)

    return (
        fpr,
        tpr,
        np.r_[np.inf, thresholds_roc],
        np.hstack((precision[sl], 1)),
        np.hstack((recall[sl], 0)),
        thresholds[sl],
    )


def get_thresholds(
    y: np.ndarray,
    n_max: int = 100,
    add_half_one: Optional[bool] = None,
    ensure: Optional[list] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> list:
    """Return equally-spaced thresholds for a given array of classification scores or class probabilities.

    Parameters
    ----------
    y : array-like
        Values used for determining the thresholds, typically (but not necessarily) the scores or class probabilities
        returned by a binary classification model. Must be a 1D-array of floats; may contain NaN and infinite values,
        which are tacitly ignored.
    n_max : int, default=100
        Maximum number of thresholds to return. Note that `ensure` takes precedence over this parameter, i.e., if
        `ensure` is given, the output may contain more than `n_max` elements.
    add_half_one : bool, optional
        Ensure 0.5 and 1.0 in the resulting list of thresholds. If None, 0.5 and 1.0 are added iff all elements of `y`
        are in the [0, 1] interval, i.e., correspond to class probabilities.
    ensure : list, optional
        Thresholds to ensure. If given, all of its elements appear in the final list.
    sample_weight : array-like, optional
        Sample weights. Thresholds are chosen such that the total sample weights in each bin are roughly equal.

    Returns
    -------
    list of float
        Thresholds, ascending list of floats with length >= 2.
    """
    y = to_num_array(y, arg_name="y", rank=(1,))
    n = min(np.isfinite(y).sum(), n_max)
    if n == 0:
        thresholds = set()
    elif n == 1:
        thresholds = {np.nanmin(y)}
    elif sample_weight is None:
        thresholds = np.sort(y[np.isfinite(y)])
        thresholds = set(thresholds[np.linspace(0, len(thresholds) - 1, n).round().astype(np.int32)])
    else:
        sample_weight = to_num_array(sample_weight, arg_name="sample_weight", rank=(1,))
        mask = np.isfinite(y)
        y0 = y[mask]
        sample_weight = sample_weight[mask]
        indices = np.argsort(y0)
        y0 = y0[indices]
        sample_weight = sample_weight[indices]
        s = sample_weight.cumsum() / np.maximum(sample_weight.sum(), 1e-8)
        np.clip(s, 0.0, 1.0, out=s)
        s *= n
        s = np.ceil(s).astype(np.int32)
        indices = np.zeros((n,), dtype=np.int32)
        for i in range(1, n):
            aux = np.where(s == i)[0]
            if len(aux) > 0:
                indices[i] = aux[-1]
        thresholds = set(y0[indices])
    if add_half_one is True or (add_half_one is None and not (y < 0).any() and not (1 < y).any()):
        thresholds.update({0.5, 1.0})
    if ensure is not None:
        thresholds.update(ensure)
    thresholds = list(thresholds)
    if len(thresholds) == 0:
        return [0, 1]
    elif len(thresholds) == 1:
        if 0 <= thresholds[0] <= 1:
            thresholds = [0, 1]
        else:
            thresholds = [np.floor(thresholds[0]), np.floor(thresholds[0]) + 1]
    else:
        thresholds.sort()
    return thresholds


def multiclass_proba_to_pred(y) -> np.ndarray:
    """Translate multiclass class probabilities into actual predictions, by returning the class with the highest
    probability. If two or more classes have the same highest probabilities, the last one is returned. This behavior is
    consistent with binary classification problems, where the positive class is returned if both classes have equal
    probabilities and the default threshold of 0.5 is used.

    Parameters
    ----------
    y : array-like
        Class probabilities, of shape `(n_classes,)` or `(n_samples, n_classes)`. The values of `y` can be arbitrary,
        they don't need to be between 0 and 1. `n_classes` must be >= 1.

    Returns
    -------
    Predicted class indices, either single integer or array of shape `(n_samples,)`.
    """
    y = to_num_array(y, arg_name="y", rank=(1, 2))
    if y.ndim == 1:
        return len(y) - np.argmax(y[::-1]) - 1
    else:
        return y.shape[1] - np.argmax(y[:, ::-1], axis=1) - 1


class thresholded(_OperatorBase):
    """Convenience class for converting a classification metric that can only be applied to class predictions into a
    metric that can be applied to probabilities. This proceeds by specifying a fixed decision threshold.

    Parameters
    ----------
    func : callable
        The metric to convert, e.g., `accuracy`, `balanced_accuracy`, etc.
    threshold : float | str, default=0.5
        The decision threshold. In binary classification this can also be the name of a thresholding strategy that is
        accepted by function `get_thresholding_strategy()`.
    **kwargs
        Additional keyword arguments that are passed to `func` upon application.

    Returns
    -------
    New metric that, when applied to `y_true` and `y_score`, returns `func(y_true, y_score >= threshold)` in case of
    binary- or multilabel classification, and `func(y_true, multiclass_proba_to_pred(y_score))` in case of multiclass
    classification.
    """

    def __init__(self, func, threshold: Union[float, str] = 0.5, **kwargs):
        if isinstance(threshold, str):
            if threshold == "argmax":
                threshold = partial(argmax_threshold, func)
            else:
                threshold = get_thresholding_strategy(threshold)
        elif not isinstance(threshold, (int, float)):
            raise ValueError(f"Threshold must be a string or float, but found {threshold}")
        super(thresholded, self).__init__(func)
        self._threshold = threshold
        self._kwargs = kwargs

    @property
    def threshold(self):
        return self._threshold

    def __call__(self, y_true, y_score, **kwargs):
        kwargs.update(self._kwargs)
        if y_score.ndim == 2 and np.shape(y_score)[1] > 1 and (y_true.ndim == 1 or np.shape(y_true)[1] == 1):
            # multiclass classification => `threshold` is not needed
            return self._func(y_true, multiclass_proba_to_pred(y_score), **kwargs)
        elif isinstance(self._threshold, (int, float)):
            # binary- or multilabel classification
            return self._func(y_true, y_score >= self._threshold, **kwargs)
        else:
            t = self._threshold(y_true, y_score, sample_weight=kwargs.get("sample_weight"))
            return self._func(y_true, y_score >= t, **kwargs)

    @classmethod
    def make(cls, func, threshold: Union[float, str] = 0.5, **kwargs):
        """Convenience function for converting a classification metric into its "thresholded" version IF NECESSARY.
        That means, if the given metric can be applied to class probabilities, it is returned unchanged. Otherwise,
        `thresholded(func, threshold)` is returned.

        Parameters
        ----------
        func : callable
            The metric to convert, e.g., `accuracy`, `balanced_accuracy`, etc.
        threshold : float | str, default=0.5
            The decision threshold.
        **kwargs
            Additional keyword arguments that shall be passed to `func` upon application.

        Returns
        -------
        Either `func` itself or `thresholded(func, threshold)`.
        """

        # strip all bootstrapping wrappers
        test_func = func
        while isinstance(test_func, _bootstrapped):
            kwargs.update(test_func.kwargs)
            test_func = test_func.base_function

        if not isinstance(test_func, cls):
            try:
                # apply `func` to see whether it can be applied to probabilities
                # apply it to one positive and one negative sample, because some metrics like `roc_auc` raise an
                # exception if only one class is present
                test_func(
                    np.arange(2, dtype=np.float32),
                    0.3 * np.arange(1, 3, dtype=np.float32),
                    **kwargs,
                )
            except:  # noqa
                try:
                    # check whether `func` can be applied to multilabel tasks
                    # we need one positive and one negative label for each sample
                    test_func(
                        np.eye(2, dtype=np.float32),
                        0.2 * np.arange(1, 5, dtype=np.float32).reshape((2, 2)),
                        **kwargs,
                    )
                except:  # noqa
                    try:
                        # check whether `func` can be applied to multiclass tasks
                        test_func(
                            np.arange(3, dtype=np.float32),
                            np.array(
                                [[0.1, 0.3, 0.6], [0.9, 0.1, 0], [0.2, 0.7, 0.1]],
                                dtype=np.float32,
                            ),
                            **kwargs,
                        )
                    except:  # noqa
                        return cls(func, threshold=threshold, **kwargs)

        if kwargs:
            return partial(func, **kwargs)
        else:
            return func


maybe_thresholded = thresholded.make


# classification with thresholds
precision_recall_fscore_support = skl_metrics.precision_recall_fscore_support
accuracy = averageable(skl_metrics.accuracy_score, accepts_global=True)
accuracy_micro = micro_average(skl_metrics.accuracy_score)
accuracy_macro = macro_average(skl_metrics.accuracy_score)
accuracy_samples = samples_average(skl_metrics.accuracy_score)
accuracy_weighted = weighted_average(skl_metrics.accuracy_score)
balanced_accuracy = averageable(skl_metrics.balanced_accuracy_score, accepts_global=True)
balanced_accuracy_micro = micro_average(skl_metrics.balanced_accuracy_score)
balanced_accuracy_macro = macro_average(skl_metrics.balanced_accuracy_score)
balanced_accuracy_samples = samples_average(skl_metrics.balanced_accuracy_score)
balanced_accuracy_weighted = weighted_average(skl_metrics.balanced_accuracy_score)
f1 = skl_metrics.f1_score
f1_micro = partial(f1, average="micro")
f1_macro = partial(f1, average="macro")
f1_samples = partial(f1, average="samples")
f1_weighted = partial(f1, average="weighted")
sensitivity = partial(skl_metrics.recall_score, zero_division=0)
sensitivity_micro = partial(sensitivity, average="micro", pos_label=None)
sensitivity_macro = partial(sensitivity, average="macro", pos_label=None)
sensitivity_samples = partial(sensitivity, average="samples", pos_label=None)
sensitivity_weighted = partial(sensitivity, average="weighted", pos_label=None)
specificity = averageable(partial(skl_metrics.recall_score, pos_label=0, zero_division=0))
specificity_micro = micro_average(specificity)  # cannot use built-in version, because `pos_label=0` would be ignored
specificity_macro = macro_average(specificity)
specificity_samples = samples_average(specificity)
specificity_weighted = weighted_average(specificity)
positive_predictive_value = partial(skl_metrics.precision_score, zero_division=1)
positive_predictive_value_micro = partial(positive_predictive_value, average="micro")
positive_predictive_value_macro = partial(positive_predictive_value, average="macro")
positive_predictive_value_samples = partial(positive_predictive_value, average="samples")
positive_predictive_value_weighted = partial(positive_predictive_value, average="weighted")
negative_predictive_value = averageable(partial(skl_metrics.precision_score, pos_label=0, zero_division=1))
negative_predictive_value_micro = micro_average(negative_predictive_value)  # see comment at `specificity_micro`
negative_predictive_value_macro = macro_average(negative_predictive_value)
negative_predictive_value_samples = samples_average(negative_predictive_value)
negative_predictive_value_weighted = weighted_average(negative_predictive_value)
cohen_kappa = averageable(skl_metrics.cohen_kappa_score, accepts_global=True)
cohen_kappa_micro = micro_average(skl_metrics.cohen_kappa_score)
cohen_kappa_macro = macro_average(skl_metrics.cohen_kappa_score)
cohen_kappa_samples = samples_average(skl_metrics.cohen_kappa_score)
cohen_kappa_weighted = weighted_average(skl_metrics.cohen_kappa_score)
matthews_correlation_coefficient = averageable(skl_metrics.matthews_corrcoef, accepts_global=True)
matthews_correlation_coefficient_micro = micro_average(skl_metrics.matthews_corrcoef)
matthews_correlation_coefficient_macro = macro_average(skl_metrics.matthews_corrcoef)
matthews_correlation_coefficient_samples = samples_average(skl_metrics.matthews_corrcoef)
matthews_correlation_coefficient_weighted = weighted_average(skl_metrics.matthews_corrcoef)
jaccard = partial(skl_metrics.jaccard_score, zero_division=1)
jaccard_micro = partial(jaccard, average="micro")
jaccard_macro = partial(jaccard, average="macro")
jaccard_samples = partial(jaccard, average="samples")
jaccard_weighted = partial(jaccard, average="weighted")


def confusion_matrix(
    y_true, y_pred, multilabel="auto", normalize: Optional[str] = None, samplewise: bool = False, **kwargs
) -> np.ndarray:
    """Compute confusion matrix to evaluate the accuracy of a classification.

    In the binary and multiclass case, the result is an array of shape `(n_classes, n_classes)` whose `ij`-th entry is
    the number of samples belonging to class `i` and classified as class `j`.
    In short: rows = ground truth, columns = predictions.

    In the multilabel case, the result is an array of shape `(n_labels, 2, 2)`, with a binary confusion matrix for each
    label.

    Parameters
    ----------
    y_true : array-like
        Ground-truth (correct) target values, array-like of shape `(samples,)` or `(n_samples, n_labels)`.
    y_pred : array-like
        Predictions, array-like with the same shape as `y_true`.
    multilabel : str | bool, default="auto"
        Whether to return a binary/multiclass confusion matrix, or a multiplabel confusion matrix:

        * True: Return a multilabel confusion matrix, even if the input is binary/multiclass. Multiclass data will be
            treated as if binarized under a one-vs-rest transformation.
        * False: Return a binary/multiclass confusion matrix. Raises a ValueError if the input is multilabel.
        * "auto" (default): Automatically detect the confusion matrix type to return: multilabel if the input is
            multilabel, binary/multiclass otherwise.

    normalize : str, optional
        Normalize the confusion matrix over the rows ("true"), columns ("pred") conditions or the whole population
        ("all"). If None, the confusion matrix will not be normalized.
        For multilabel input, each of the 2x2 confusion matrices is normalized separately.
    samplewise : bool, default=False
        In the multilabel case, this calculates a confusion matrix per sample.

    Returns
    -------
    Confusion matrix, array of shape `(n_classes, n_classes)` if `multilabel` is False, or `(n_classe, 2, 2)` if
    `multilabel` is True.

    See Also
    --------
    sklearn.metrics.confusion_matrix
    sklearn.metrics.multilabel_confusion_matrix

    Notes
    -----
    This implementation combines both `sklearn.metrics.confusion_matrix` and
    `sklearn.metrics.multilabel_confusion_matrix`. Setting `multilabel` to False is equivalent to the former, setting
    it to True is equivalent to the latter.
    """
    if multilabel == "auto":
        y_true, y_pred = to_num_arrays(y_true, y_pred, arg_names=("y_true", "y_pred"), rank=(1, 2))
        return confusion_matrix(
            y_true, y_pred, multilabel=y_true.ndim == 2, normalize=normalize, samplewise=samplewise, **kwargs
        )
    elif multilabel is True:
        cm = skl_metrics.multilabel_confusion_matrix(y_true, y_pred, samplewise=samplewise, **kwargs)  # (N, 2, 2)
        if normalize == "true":
            cm = cm / cm.sum(axis=2, keepdims=True)
        elif normalize == "pred":
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "all":
            cm = cm / cm.sum(axis=(1, 2), keepdims=True)
        return cm
    elif multilabel is False:
        if samplewise:
            raise ValueError("Binary/multiclass confusion matrices cannot be constructed per sample.")
        return skl_metrics.confusion_matrix(y_true, y_pred, normalize=normalize, **kwargs)
    else:
        raise ValueError('multilabel must be one of True, False or "auto", but got {}'.format(multilabel))


def hamming_loss(y_true, y_pred, *, sample_weight=None, **kwargs):
    """Hamming loss is 1 - accuracy, but their multilabel default averaging policy differs:
    accuracy returns subset accuracy by default (i.e., all labels must match), whereas hamming loss returns label-wise
    macro average by default.
    """
    y_true, y_pred = to_num_arrays(y_true, y_pred, arg_names=("y_true", "y_pred"), rank=(1, 2))
    if y_true.ndim == 1 or "average" in kwargs:
        return 1.0 - accuracy(y_true, y_pred, sample_weight=sample_weight, **kwargs)
    else:
        # multilabel without explicitly specified averaging policy => macro averaging
        return 1.0 - accuracy_macro(y_true, y_pred, sample_weight=sample_weight, **kwargs)


def informedness(y_true, y_pred, *, average="binary", sample_weight=None):
    """Informedness (aka Youden index or Youden's J statistic) is the sum of sensitivity and specificity, minus 1."""
    # Don't define informedness in terms of balanced accuracy, because balanced accuracy is defined for multiclass
    # tasks but informedness is not (at least, not analogous to balanced accuracy).
    sens = sensitivity(y_true, y_pred, sample_weight=sample_weight, average=average)
    spec = specificity(y_true, y_pred, sample_weight=sample_weight, average=average)
    return sens + spec - 1


def markedness(y_true, y_pred, *, average="binary", sample_weight=None):
    """Markedness is the sum of positive- and negative predictive value, minus 1."""
    ppv = positive_predictive_value(y_true, y_pred, sample_weight=sample_weight, average=average)
    npv = negative_predictive_value(y_true, y_pred, sample_weight=sample_weight, average=average)
    return ppv + npv - 1


hamming_loss_micro = partial(hamming_loss, average="micro")
hamming_loss_macro = partial(hamming_loss, average="macro")
hamming_loss_samples = partial(hamming_loss, average="samples")
hamming_loss_weighted = partial(hamming_loss, average="weighted")
informedness_micro = partial(informedness, average="micro")
informedness_macro = partial(informedness, average="macro")
informedness_samples = partial(informedness, average="samples")
informedness_weighted = partial(informedness, average="weighted")
markedness_micro = partial(markedness, average="micro")
markedness_macro = partial(markedness, average="macro")
markedness_samples = partial(markedness, average="samples")
markedness_weighted = partial(markedness, average="weighted")


def precision_recall_fscore_support_cm(
    *,
    cm=None,
    tp=None,
    fp=None,
    tn=None,
    fn=None,
    beta: float = 1.0,
    average: Optional[str] = None,
    zero_division: Union[float, str] = "warn",
):
    """Compute precision, recall, F-measure and support for each class. This is the confusion-matrix based variant of
    `precision_recall_fscore_support`.

    Parameters
    ----------
    cm : array-like, optional
        Confusion matrix, array-like of shape `(n_classes, n_classes)` or `(n_labels, 2, 2)`.
        Mutually exclusive with `tp`, `fp`, `tn` and `fn`.
    tp : scalar | array-like, optional
        Number/fraction of true positives, scalar or array-like of shape `(n_labels,)`. If given, `fp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fp : scalar | array-like, optional
        Number/fraction of false positives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    tn : scalar | array-like, optional
        Number/fraction of true negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fn : scalar | array-like, optional
        Number/fraction of false negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `tn`
        must be given, too.
        Mutually exclusive with `cm`.
    beta : float, default=1
        The strength of recall versus precision in the F-score.
    average : str, optional
        Averaging to perform for multiclass and multilabel input.
    zero_division : float | str, default="warn"
        The value to return if there is a division by zero.

    Returns
    -------
    precision : float | ndarray
        Precision score (positive predictive value).
    recall : float | ndarray
        Recall score (sensitivity).
    f_score : float | ndarray
        F-beta score.
    support : float | ndarray
        The support of each class/label. None unless `average` is None.

    See Also
    --------
    precision_recall_fscore_support
    """
    res = _calc_single_cm_metric(
        _precision_recall_fscore_support_cm_aux,
        cm,
        tp,
        fp,
        tn,
        fn,
        beta=beta,
        average="list" if average is None else average,
        zero_division=zero_division,
    )
    if isinstance(res, list):
        res = [np.array([r[i] for r in res]) for i in range(4)]
        r3 = res[3].astype(np.int64)
        if (r3.astype(res[3].dtype) == res[3]).all():
            res[3] = r3
        return tuple(res)
    else:
        return res[0], res[1], res[2], None


def _precision_recall_fscore_support_cm_aux(
    tp, fp, tn, fn, beta: float = 1.0, zero_division: Union[float, str] = "warn"
):
    b2 = beta * beta
    pr = _precision_cm_aux(tp, fp, tn, fn, zero_division=zero_division)
    rc = _recall_cm_aux(tp, fp, tn, fn, zero_division=zero_division)
    fb = (1 + b2) * (pr * rc) / (b2 * pr + rc)
    return np.array([pr, rc, fb, tp + fn])


def precision_cm(
    *,
    cm=None,
    tp=None,
    fp=None,
    tn=None,
    fn=None,
    average: Optional[str] = "binary",
    swap_pos_neg: bool = False,
    zero_division: Union[float, str] = "warn",
) -> Union[float, int, np.ndarray]:
    """Compute the precision (positive predictive value) from a given confusion matrix.
    This is the confusion-matrix based variant of `positive_predictive_value`.

    Parameters
    ----------
    cm : array-like, optional
        Confusion matrix, array-like of shape `(n_classes, n_classes)` or `(n_labels, 2, 2)`.
        Mutually exclusive with `tp`, `fp`, `tn` and `fn`.
    tp : scalar | array-like, optional
        Number/fraction of true positives, scalar or array-like of shape `(n_labels,)`. If given, `fp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fp : scalar | array-like, optional
        Number/fraction of false positives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    tn : scalar | array-like, optional
        Number/fraction of true negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fn : scalar | array-like, optional
        Number/fraction of false negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `tn`
        must be given, too.
        Mutually exclusive with `cm`.
    swap_pos_neg : bool, default=False
        Swap positive and negative class. If True, negative predictive value is computed instead.
    average : str, optional
        Averaging to perform for multiclass and multilabel input.
    zero_division : float | str, default="warn"
        The value to return if there is a division by zero.

    Returns
    -------
    Precision score (positive predictive value).

    See Also
    --------
    positive_predictive_value
    negative_predictive_value
    positive_predictive_value_cm (alias)
    negative_predictive_value_cm
    """
    return _calc_single_cm_metric(
        _precision_cm_aux, cm, tp, fp, tn, fn, average=average, zero_division=zero_division, swap_pos_neg=swap_pos_neg
    )


def _precision_cm_aux(tp, fp, tn, fn, zero_division: Union[float, str] = "warn", swap_pos_neg: bool = False):
    if swap_pos_neg:
        return _safe_divide(tn, tn + fn, zero_division=zero_division)
    return _safe_divide(tp, tp + fp, zero_division=zero_division)


def recall_cm(
    *,
    cm=None,
    tp=None,
    fp=None,
    tn=None,
    fn=None,
    average: Optional[str] = "binary",
    zero_division: Union[float, str] = "warn",
    swap_pos_neg: bool = False,
) -> Union[float, int, np.ndarray]:
    """Compute the recall (sensitivity) from a given confusion matrix.
    This is the confusion-matrix based variant of `sensitivity`.

    Parameters
    ----------
    cm : array-like, optional
        Confusion matrix, array-like of shape `(n_classes, n_classes)` or `(n_labels, 2, 2)`.
        Mutually exclusive with `tp`, `fp`, `tn` and `fn`.
    tp : scalar | array-like, optional
        Number/fraction of true positives, scalar or array-like of shape `(n_labels,)`. If given, `fp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fp : scalar | array-like, optional
        Number/fraction of false positives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    tn : scalar | array-like, optional
        Number/fraction of true negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fn : scalar | array-like, optional
        Number/fraction of false negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `tn`
        must be given, too.
        Mutually exclusive with `cm`.
    swap_pos_neg : bool, default=False
        Swap positive and negative class. If True, specificity is computed instead.
    average : str, optional
        Averaging to perform for multiclass and multilabel input.
    zero_division : float | str, default="warn"
        The value to return if there is a division by zero.

    Returns
    -------
    Recall score (sensitivity).

    See Also
    --------
    sensitivity
    specificity
    sensitivity_cm (alias)
    specificity_cm
    """
    return _calc_single_cm_metric(
        _recall_cm_aux, cm, tp, fp, tn, fn, average=average, zero_division=zero_division, swap_pos_neg=swap_pos_neg
    )


def _recall_cm_aux(tp, fp, tn, fn, zero_division: Union[float, str] = "warn", swap_pos_neg: bool = False):
    if swap_pos_neg:
        return _safe_divide(tn, tn + fp, zero_division=zero_division)
    return _safe_divide(tp, tp + fn, zero_division=zero_division)


def accuracy_cm(
    *,
    cm=None,
    tp=None,
    fp=None,
    tn=None,
    fn=None,
    average: Optional[str] = "global",
    normalize: bool = True,
) -> Union[float, int, np.ndarray]:
    """Compute the accuracy from a given confusion matrix. This is the confusion-matrix based variant of `accuracy`.

    Parameters
    ----------
    cm : array-like, optional
        Confusion matrix, array-like of shape `(n_classes, n_classes)` or `(n_labels, 2, 2)`.
        Mutually exclusive with `tp`, `fp`, `tn` and `fn`.
    tp : scalar | array-like, optional
        Number/fraction of true positives, scalar or array-like of shape `(n_labels,)`. If given, `fp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fp : scalar | array-like, optional
        Number/fraction of false positives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    tn : scalar | array-like, optional
        Number/fraction of true negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fn : scalar | array-like, optional
        Number/fraction of false negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `tn`
        must be given, too.
        Mutually exclusive with `cm`.
    normalize : bool, default=True
        Return the fraction of correctly classified samples. Otherwise, return the number of correctly classified
        samples.
    average : str, optional
        Averaging to perform for multiclass and multilabel input.

    Returns
    -------
    Accuracy score.

    See Also
    --------
    accuracy
    """
    cm = _get_cm(cm, tp, fp, tn, fn)
    if average == "global":
        if cm.ndim == 3:
            raise ValueError('average="global" is not supported in multilabel confusion-matrix-based metrics.')
        elif normalize:
            return _safe_divide(cm.trace(), cm.sum(), zero_division=1, inplace=True)
        else:
            return cm.trace()
    elif average == "binary":
        # ensure consistency with `accuracy`
        raise ValueError(_BINARY_EXCEPTION)
    else:
        return _calc_single_cm_metric(
            _accuracy_cm_aux, cm, None, None, None, None, average=average, normalize=normalize
        )


def _accuracy_cm_aux(tp, fp, tn, fn, normalize: bool = True):
    correct = tp + tn
    if normalize:
        return _safe_divide(correct, correct + fp + fn, zero_division=1, inplace=True)
    else:
        return correct


def balanced_accuracy_cm(
    *,
    cm=None,
    tp=None,
    fp=None,
    tn=None,
    fn=None,
    average: Optional[str] = "global",
    adjusted: bool = False,
) -> Union[float, int, np.ndarray]:
    """Compute the balanced accuracy from a given confusion matrix. This is the confusion-matrix based variant of
    `balanced_accuracy`.

    Parameters
    ----------
    cm : array-like, optional
        Confusion matrix, array-like of shape `(n_classes, n_classes)` or `(n_labels, 2, 2)`.
        Mutually exclusive with `tp`, `fp`, `tn` and `fn`.
    tp : scalar | array-like, optional
        Number/fraction of true positives, scalar or array-like of shape `(n_labels,)`. If given, `fp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fp : scalar | array-like, optional
        Number/fraction of false positives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    tn : scalar | array-like, optional
        Number/fraction of true negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fn : scalar | array-like, optional
        Number/fraction of false negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `tn`
        must be given, too.
        Mutually exclusive with `cm`.
    adjusted : bool, default=False
        Adjust the result for chance, so that random performance would score 0, while keeping perfect performance at a
        score of 1.
    average : str, optional
        Averaging to perform for multiclass and multilabel input.

    Returns
    -------
    Balanced accuracy score.

    See Also
    --------
    balanced_accuracy
    """
    cm = _get_cm(cm, tp, fp, tn, fn)
    if average == "global":
        if cm.ndim == 3:
            raise ValueError("multilabel-indicator is not supported")
        elif cm.shape[0] == 2:
            return _balanced_accuracy_cm_aux(cm[1, 1], cm[0, 1], cm[0, 0], cm[1, 0], adjusted=adjusted)
        else:
            score = recall_cm(cm=cm, average="macro")
            if adjusted:
                score -= 0.5
                score *= 2
            return score
    elif average == "binary":
        # ensure consistency with `balanced_accuracy`
        raise ValueError(_BINARY_EXCEPTION)
    else:
        return _calc_single_cm_metric(
            _balanced_accuracy_cm_aux, cm, None, None, None, None, average=average, adjusted=adjusted
        )


def _balanced_accuracy_cm_aux(tp, fp, tn, fn, adjusted: bool = False):
    score = (tp / (tp + fn) + tn / (tn + fp)) * 0.5
    if adjusted:
        score -= 0.5
        score *= 2
    return score


def fbeta_cm(
    *,
    cm=None,
    tp=None,
    fp=None,
    tn=None,
    fn=None,
    average: Optional[str] = "binary",
    zero_division: Union[str, float] = "warn",
    beta: float = 1.0,
) -> Union[float, int, np.ndarray]:
    """Compute the F-beta score from a given confusion matrix.

    Parameters
    ----------
    cm : array-like, optional
        Confusion matrix, array-like of shape `(n_classes, n_classes)` or `(n_labels, 2, 2)`.
        Mutually exclusive with `tp`, `fp`, `tn` and `fn`.
    tp : scalar | array-like, optional
        Number/fraction of true positives, scalar or array-like of shape `(n_labels,)`. If given, `fp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fp : scalar | array-like, optional
        Number/fraction of false positives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    tn : scalar | array-like, optional
        Number/fraction of true negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fn : scalar | array-like, optional
        Number/fraction of false negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `tn`
        must be given, too.
        Mutually exclusive with `cm`.
    beta : float, default=1
        The strength of recall versus precision.
    average : str, optional
        Averaging to perform for multiclass and multilabel input.
    zero_division : float | str, default="warn"
        The value to return if there is a division by zero.

    Returns
    -------
    F-beta score.

    See Also
    --------
    precision_recall_fscore_support_cm
    f1_cm
    """
    return _calc_single_cm_metric(
        _fbeta_cm_aux, cm, tp, fp, tn, fn, average=average, zero_division=zero_division, beta=beta
    )


def _fbeta_cm_aux(tp, fp, tn, fn, zero_division="warn", beta=1.0):
    b2 = beta * beta
    return _safe_divide(
        (1 + b2) * tp,
        (1 + b2) * tp + b2 * fn + fp,
        zero_division=zero_division,
        inplace=True,
    )


def cohen_kappa_cm(
    *, cm=None, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = "global", weights: Optional[str] = None
) -> Union[float, int, np.ndarray]:
    """Compute Cohen's kappa from a given confusion matrix. This is the confusion-matrix based variant of `cohen_kappa`.

    Parameters
    ----------
    cm : array-like, optional
        Confusion matrix, array-like of shape `(n_classes, n_classes)` or `(n_labels, 2, 2)`.
        Mutually exclusive with `tp`, `fp`, `tn` and `fn`.
    tp : scalar | array-like, optional
        Number/fraction of true positives, scalar or array-like of shape `(n_labels,)`. If given, `fp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fp : scalar | array-like, optional
        Number/fraction of false positives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    tn : scalar | array-like, optional
        Number/fraction of true negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fn : scalar | array-like, optional
        Number/fraction of false negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `tn`
        must be given, too.
        Mutually exclusive with `cm`.
    weights : str, optional
        Weighting type to calculate the score. None means not weighted; "linear" means linear weighting; "quadratic"
        means quadratic weighting.
    average : str, optional
        Averaging to perform for multiclass and multilabel input.

    Returns
    -------
    Kappa statistic, float or array of floats between -1 and 1. The maximum value means complete agreement; zero or
    lower means chance agreement.

    See Also
    --------
    cohen_kappa
    """
    cm = _get_cm(cm, tp, fp, tn, fn)
    if average == "global":
        if cm.ndim == 3:
            raise ValueError("multilabel-indicator is not supported")
        elif cm.shape[0] == 2:
            return _cohen_kappa_cm_aux(cm[1, 1], cm[0, 1], cm[0, 0], cm[1, 0])
        else:
            # copied from sklearn.metrics._classification
            n_classes = cm.shape[0]
            sum0 = cm.sum(axis=0)  # number of predictions per class
            sum1 = cm.sum(axis=1)  # per-class support
            expected = np.outer(sum0, sum1) / sum0.sum()

            if weights is None:
                w_mat = np.ones([n_classes, n_classes], dtype=cm.dtype)
                w_mat.flat[:: n_classes + 1] = 0
            else:  # "linear" or "quadratic"
                w_mat = np.zeros([n_classes, n_classes], dtype=cm.dtype)
                w_mat += np.arange(n_classes)
                if weights == "linear":
                    w_mat = np.abs(w_mat - w_mat.T)
                else:
                    w_mat = np.square(w_mat - w_mat.T)

            return 1.0 - _safe_divide(np.sum(w_mat * cm), np.sum(w_mat * expected), zero_division=1, inplace=True)
    elif average == "binary":
        # ensure consistency with `cohen_kappa`
        raise ValueError(_BINARY_EXCEPTION)
    else:
        # `weights` have no effect on binary tasks
        return _calc_single_cm_metric(_cohen_kappa_cm_aux, cm, None, None, None, None, average=average)


def _cohen_kappa_cm_aux(tp, fp, tn, fn):
    return _safe_divide(
        2 * (tp * tn - fp * fn),
        (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn),
        zero_division=0,
        inplace=True,
    )


def matthews_correlation_coefficient_cm(
    *, cm=None, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = "global"
) -> Union[float, int, np.ndarray]:
    """Compute the Matthews correlation coefficient (MCC) from a given confusion matrix.
    This is the confusion-matrix based variant of `matthews_correlation_coefficient`.

    Parameters
    ----------
    cm : array-like, optional
        Confusion matrix, array-like of shape `(n_classes, n_classes)` or `(n_labels, 2, 2)`.
        Mutually exclusive with `tp`, `fp`, `tn` and `fn`.
    tp : scalar | array-like, optional
        Number/fraction of true positives, scalar or array-like of shape `(n_labels,)`. If given, `fp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fp : scalar | array-like, optional
        Number/fraction of false positives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    tn : scalar | array-like, optional
        Number/fraction of true negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fn : scalar | array-like, optional
        Number/fraction of false negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `tn`
        must be given, too.
        Mutually exclusive with `cm`.
    average : str, optional
        Averaging to perform for multiclass and multilabel input.

    Returns
    -------
    Matthews correlation coefficient (+1 represents a perfect prediction, 0 an average random prediction and -1 and
    inverse prediction).

    See Also
    --------
    matthews_correlation_coefficient
    """
    cm = _get_cm(cm, tp, fp, tn, fn)
    if average == "global":
        if cm.ndim == 3:
            raise ValueError("multilabel-indicator is not supported")
        elif cm.shape[0] == 2:
            return _mcc_cm_aux(cm[1, 1], cm[0, 1], cm[0, 0], cm[1, 0])
        else:
            # copied from sklearn.metrics._classification
            t_sum = cm.sum(axis=1)
            p_sum = cm.sum(axis=0)
            n_correct = np.trace(cm)
            n_samples = p_sum.sum()
            cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
            cov_ypyp = n_samples**2 - np.dot(p_sum, p_sum)
            cov_ytyt = n_samples**2 - np.dot(t_sum, t_sum)

            return _safe_divide(cov_ytyp, np.sqrt(cov_ytyt * cov_ypyp), zero_division=0, inplace=True)
    elif average == "binary":
        # ensure consistency with `matthews_correlation_coefficient`
        raise ValueError(_BINARY_EXCEPTION)
    else:
        return _calc_single_cm_metric(_mcc_cm_aux, cm, None, None, None, None, average=average)


def _mcc_cm_aux(tp, fp, tn, fn):
    return _safe_divide(
        tp * tn - fp * fn,
        np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
        zero_division=0,
        inplace=True,
    )


def jaccard_cm(
    *,
    cm=None,
    tp=None,
    fp=None,
    tn=None,
    fn=None,
    average: Optional[str] = "binary",
    zero_division: Union[str, float] = "warn",
) -> Union[float, int, np.ndarray]:
    """Compute the Jaccard score (intersection over union, IoU) from a given confusion matrix.
    This is the confusion-matrix based variant of `jaccard`.

    Parameters
    ----------
    cm : array-like, optional
        Confusion matrix, array-like of shape `(n_classes, n_classes)` or `(n_labels, 2, 2)`.
        Mutually exclusive with `tp`, `fp`, `tn` and `fn`.
    tp : scalar | array-like, optional
        Number/fraction of true positives, scalar or array-like of shape `(n_labels,)`. If given, `fp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fp : scalar | array-like, optional
        Number/fraction of false positives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `tn` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    tn : scalar | array-like, optional
        Number/fraction of true negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `fn`
        must be given, too.
        Mutually exclusive with `cm`.
    fn : scalar | array-like, optional
        Number/fraction of false negatives, scalar or array-like of shape `(n_labels,)`. If given, `tp`, `fp` and `tn`
        must be given, too.
        Mutually exclusive with `cm`.
    average : str, optional
        Averaging to perform for multiclass and multilabel input.
    zero_division : float | str, default="warn"
        The value to return if there is a division by zero.

    Returns
    -------
    Jaccard score.

    See Also
    --------
    jaccard
    """
    return _calc_single_cm_metric(_jaccard_cm_aux, cm, tp, fp, tn, fn, average=average, zero_division=zero_division)


def _jaccard_cm_aux(tp, fp, tn, fn, zero_division="warn"):
    return _safe_divide(tp, tp + fn + fp, zero_division=zero_division, inplace=False)


def hamming_loss_cm(*, cm=None, tp=None, fp=None, tn=None, fn=None, **kwargs) -> Union[float, int, np.ndarray]:
    """Hamming loss is 1 - accuracy, but their multilabel default averaging policy differs:
    accuracy returns subset accuracy by default (i.e., all labels must match), whereas hamming loss returns label-wise
    macro average by default.
    """
    cm = _get_cm(cm, tp, fp, tn, fn)
    if cm.ndim == 2 or "average" in kwargs:
        return 1.0 - accuracy_cm(cm=cm, **kwargs)
    else:
        # multilabel without explicitly specified averaging policy => macro averaging
        return 1.0 - accuracy_cm(cm=cm, average="macro", **kwargs)


f1_cm = partial(fbeta_cm, beta=1)
sensitivity_cm = partial(recall_cm, zero_division=0)
specificity_cm = partial(recall_cm, swap_pos_neg=True, zero_division=0)
positive_predictive_value_cm = partial(precision_cm, zero_division=1)
negative_predictive_value_cm = partial(precision_cm, swap_pos_neg=True, zero_division=1)


def informedness_cm(*, cm=None, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = "binary"):
    """Informedness (aka Youden index or Youden's J statistic) is the sum of sensitivity and specificity, minus 1."""
    sens = sensitivity_cm(cm=cm, tp=tp, fp=fp, tn=tn, fn=fn, average=average)
    spec = specificity_cm(cm=cm, tp=tp, fp=fp, tn=tn, fn=fn, average=average)
    return sens + spec - 1


def markedness_cm(*, cm=None, tp=None, fp=None, tn=None, fn=None, average: Optional[str] = "binary"):
    """Markedness is the sum of positive- and negative predictive value, minus 1."""
    ppv = positive_predictive_value_cm(cm=cm, tp=tp, fp=fp, tn=tn, fn=fn, average=average)
    npv = negative_predictive_value_cm(cm=cm, tp=tp, fp=fp, tn=tn, fn=fn, average=average)
    return ppv + npv - 1


def _calc_single_cm_metric(func, cm, tp, fp, tn, fn, average=None, weighted_zero_division=1, **kwargs):
    if average == "samples":
        raise ValueError('average="samples" is not supported in confusion-matrix based metrics.')

    cm = _get_cm(cm, tp, fp, tn, fn)
    if average == "binary":
        if cm.ndim == 2 and cm.shape[0] == 2:
            return func(cm[1, 1], cm[0, 1], cm[0, 0], cm[1, 0], **kwargs)
        else:
            raise ValueError(
                'Target is multiclass or multilabel-indicator but average="binary". Please choose another average'
                ' setting, one of None, "micro", "macro" or "weighted".'
            )
    elif average == "micro":
        cm = cm.sum(axis=0) if cm.ndim == 3 else sum(_iter_cm(cm))
        return func(cm[1, 1], cm[0, 1], cm[0, 0], cm[1, 0], **kwargs)
    else:
        values = [
            (func(cm_binary[1, 1], cm_binary[0, 1], cm_binary[0, 0], cm_binary[1, 0], **kwargs), cm_binary[1].sum())
            for cm_binary in _iter_cm(cm)
        ]
        if average == "macro":
            return sum(v for v, _ in values) / len(values)
        elif average == "weighted":
            return _safe_divide(
                sum(v * w for v, w in values),
                sum(w for _, w in values),
                zero_division=weighted_zero_division,
                inplace=True,
            )
        elif average == "list":
            return [v for v, _ in values]
        elif average is None:
            return np.array([v for v, _ in values])
        else:
            raise ValueError(
                'average has to be one of None, "binary", "micro", "macro" or "weighted", but got "{}"'.format(average)
            )


def _sort_cumsum_groupby_last(
    index: np.ndarray, values: np.ndarray, ascending: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    # `index` must have shape `(N,)`, `values` must have shape `(N, ...)`, `N` must be greater than 0
    # sort and group by `index`, take last element of `cumsum(values)` in each group, return new index and values
    # note: `values_out[-1]` is equal to `values.sum(axis=0)`
    sorter = np.argsort(index if ascending else -index)
    index = index[sorter]
    values = values[sorter]
    mask = np.ones(index.shape, dtype=bool)
    mask[:-1] = index[:-1] != index[1:]
    index = index[mask]  # (N0,)
    cs = np.cumsum(values, axis=0)[mask]  # (N0, ...)
    return index, cs


def _get_cm(cm, tp, fp, tn, fn) -> np.ndarray:
    # returns array of shape
    #   * (2, 2)        binary classification       [[tn, fp], [fn, tp]]
    #   * (N, N)        multiclass classification
    #   * (N, 2, 2)     multilabel classification
    if cm is None:
        tp, fp, tn, fn = to_num_arrays(tp, fp, tn, fn, arg_names=("tp", "fp", "tn", "fn"))
        if np.ndim(tp) == 0:
            return np.array([[tn, fp], [fn, tp]])
        else:
            return np.stack([tn, fp, fn, tp], axis=1).reshape(-1, 2, 2)
    else:
        for x, n in [(tp, "tp"), (fp, "fp"), (tn, "tn"), (fn, "fn")]:
            if x is not None:
                raise ValueError("If `cm` is specified, `{}` must be None.".format(n))
        cm = to_num_array(cm, arg_name="cm", rank=(2, 3))
        if cm.shape[-2] != cm.shape[-1]:
            raise ValueError("Expected `cm` to be square, but got array of shape {}".format(cm.shape))
        elif cm.ndim == 3 and cm.shape[-1] != 2:
            raise ValueError("`cm` must have shape (N, 2, 2), but got array of shape {}".format(cm.shape))
        elif cm.shape[-1] < 2:
            raise ValueError("`cm` must represent at least 2 classes, but got array of shape {}".format(cm.shape))
        return cm


def _iter_cm(cm: np.ndarray):
    if cm.ndim == 3:
        for i in range(cm.shape[0]):
            yield cm[i]
    else:
        s = cm.sum()
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i].sum() - tp
            yield np.array([[s - fp - fn - tp, fp], [fn, tp]])


def _safe_divide(num, denom, zero_division: Union[str, float] = 0, inplace: bool = False):
    mask = denom == 0
    if np.any(mask):
        if zero_division == "warn":
            zero_division = 0
        if np.ndim(mask) == 0:
            return zero_division
        else:
            denom[mask] = 1
            if inplace and getattr(num, "dtype", np.dtype(np.int8)).kind == "f":
                num /= denom
            else:
                num = num / denom
            num[mask] = zero_division
            return num
    elif inplace and getattr(num, "dtype", np.dtype(np.int8)).kind == "f":
        num /= denom
        return num
    else:
        return num / denom


_BINARY_EXCEPTION = 'average has to be one of None, "global", "micro", "macro" or "weighted", but got "binary"'
