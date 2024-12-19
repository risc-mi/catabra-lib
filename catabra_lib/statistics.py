#  Copyright (c) 2024. RISC Software GmbH.
#  All rights reserved.

# DeLong-related code was copied and adapted from https://github.com/jiesihu/AUC_Delongtest__python
#   (no license specified)

from typing import Optional, Tuple

import numpy as np
from scipy import stats

from .util import to_num_array, to_num_arrays


def mann_whitney_u(x, y, nan_policy: str = "omit", **kwargs) -> float:
    """Mann-Whitney U test for testing whether two independent samples are equal (more precisely: have equal median).
    Only applicable to numerical observations; categorical observations should be treated with the chi square test.

    Parameters
    ----------
    x : array-like
        First sample, array-like with numerical values.
    y : array-like
        Second sample, array-like with numerical values.
    nan_policy : str, default="omit"
        Specifies how to handle NaN values:

        * "omit": Perform the test on all non-NaN values.
        * "propagate": Return NaN if at least one input value is NaN.
        * "raise": Raise a ValueError if at least one input value is NaN.

    **kwargs
        Keyword arguments passed to `scipy.stats.mannwhitneyu()`.

    Returns
    -------
    p_value : float
        P-value. Smaller values mean that `x` and `y` are distributed differently.

    See Also
    --------
    scipy.stats.mannwhitneyu

    Notes
    -----
    This test is symmetric between `x` and `y` if `alternative` is set to "two-sided" (default), i.e.,
    `mann_whitney_u(x, y)` equals `mann_whitney_u(y, x)`.

    The Mann-Whitney U test is a special case of the Kruskal-Wallis H test, which works for more than two samples.
    """
    args = []
    for a, n in ((x, "x"), (y, "y")):
        a, is_na = _prepare_for_numeric_test(a, n, "Mann-Whitney U test", nan_policy=nan_policy)
        if is_na or len(a) == 0:
            return np.nan
        args.append(a)

    try:
        return stats.mannwhitneyu(*args, **kwargs).pvalue
    except TypeError:
        return np.nan


def chi_square(x, y, nan_policy: str = "omit", **kwargs) -> float:
    """Chi square test for testing whether a sample of categorical observations is distributed according to another
    sample of categorical observations.

    Parameters
    ----------
    x : array-like
        First sample, array-like with categorical values.
    y : array-like
        Second sample, array-like with categorical values.
    nan_policy : str, default="omit"
        Specifies how to handle NaN values:

        * "omit": Perform the test on all non-NaN values.
        * "propagate": Return NaN if at least one input value is NaN.
        * "raise": Raise a ValueError if at least one input value is NaN.

    **kwargs
        Keyword arguments passed to `scipy.stats.chisquare()`.

    Returns
    -------
    p_value : float
        p-value. Smaller values mean that `x` is distributed differently from `y`.

    See Also
    --------
    scipy.stats.chisquare

    Notes
    -----
    This test is *not* symmetric between `x` and `y`, i.e., `chi_square(x, y)` differs from `chi_square(y, x)`
    in general.
    """

    x, xc, is_na = _prepare_for_category_test(x, "x", "Chi-Square test", nan_policy=nan_policy)
    if is_na or len(x) == 0:
        return np.nan

    y, yc, is_na = _prepare_for_category_test(y, "y", "Chi-Square test", nan_policy=nan_policy)
    if is_na or len(y) == 0:
        return np.nan

    yc = yc * (xc.sum() / yc.sum())

    all_cats = np.union1d(x, y)

    idx = np.searchsorted(all_cats, x, side="left")  # `idx` is, for each element in `x`, the position in `all_cats`
    x = np.zeros(len(all_cats), dtype=np.float64)
    x[idx] = xc

    idx = np.searchsorted(all_cats, y, side="left")  # `idx` is, for each element in `y`, the position in `all_cats`
    y = np.zeros(len(all_cats), dtype=np.float64)
    y[idx] = yc

    old = np.seterr(all="ignore")
    try:
        out = stats.chisquare(x, y, **kwargs).pvalue
        np.seterr(**old)
        return out
    except:  # noqa
        np.seterr(**old)
        return np.nan


def delong_test(y_true, y_hat_1, y_hat_2, sample_weight=None, nan_policy: str = "omit") -> float:
    """Compute the p-value of the DeLong test for the null hypothesis that two ROC-AUCs are equal.

    Parameters
    ----------
    y_true : array-like
        Ground truth, 1D array-like of shape `(n_samples,)` with values in {0, 1}.
    y_hat_1 : array-like
        Predictions of the first classifier, 1D array-like of shape `(n_samples,)` with arbitrary values. Larger values
        correspond to a higher predicted probability that a sample belongs to the positive class.
    y_hat_2 : array-like
        Predictions of the second classifier, 1D array-like of shape `(n_samples,)` with arbitrary values. Larger values
        correspond to a higher predicted probability that a sample belongs to the positive class.
    sample_weight : array-like, optional
        Sample weights. None defaults to uniform weights.
    nan_policy : str, default="omit"
        Specifies how to handle NaN values:

        * "omit": Perform the test on all non-NaN values. Since this is a paired test, all observations that are NaN in
            any of the three arrays are dropped.
        * "propagate": Return NaN if at least one input value is NaN.
        * "raise": Raise a ValueError if at least one input value is NaN.

    Returns
    -------
    p_value : float
        p-value for the null hypothesis that the ROC-AUCs of the two classifiers are equal. If this value is smaller
        than a certain pre-defined threshold (e.g., 0.05) the null hypothesis can be rejected, meaning that there is a
        statistically significant difference between the two ROC-AUCs.

    See Also
    --------
    roc_auc_confidence_interval: Confidence interval for the ROC-AUC of a given classifier.
    """

    y_true, y_hat_1, y_hat_2 = to_num_arrays(
        y_true, y_hat_1, y_hat_2, arg_names=("y_true", "y_hat_1", "y_hat_2"), rank=(1,)
    )
    if y_true.dtype.kind == "b":
        y_true = y_true.astype(np.float32)

    mask = np.isnan(y_true) | np.isnan(y_hat_1) | np.isnan(y_hat_2)
    if mask.any():
        if nan_policy == "omit":
            mask = ~mask
            y_true = y_true[mask]
            y_hat_1 = y_hat_1[mask]
            y_hat_2 = y_hat_2[mask]
            if sample_weight is not None:
                sample_weight = sample_weight[mask]
        elif nan_policy == "propagate":
            return np.nan
        else:
            raise ValueError("Input of delong_test contains NaN.")

    order, n_positive, ordered_sample_weight = _compute_ground_truth_statistics(y_true, sample_weight)
    predictions_sorted_transposed = np.vstack((y_hat_1, y_hat_2))[:, order]
    aucs, sigma = _fast_delong(predictions_sorted_transposed, n_positive, sample_weight=ordered_sample_weight)
    return _calc_pvalue(aucs[0], aucs[1], sigma)


def roc_auc_confidence_interval(
    y_true, y_hat, alpha: float = 0.95, sample_weight=None, nan_policy: str = "omit"
) -> Tuple[float, float, float]:
    """Return the confidence interval and ROC-AUC of given ground-truth and model predictions.

    Parameters
    ----------
    y_true : array-like
        Ground truth, 1D array-like of shape `(n_samples,)` with values in {0, 1}.
    y_hat : array-like
        Predictions of the classifier, 1D array-like of shape `(n_samples,)` with arbitrary values. Larger values
        correspond to a higher predicted probability that a sample belongs to the positive class.
    alpha : float, default=0.95
        Confidence level, between 0 and 1.
    sample_weight : array-like, optional
        Sample weights. None defaults to uniform weights.
    nan_policy : str, default="omit"
        Specifies how to handle NaN values:

        * "omit": Perform the test on all non-NaN values.
        * "propagate": Return NaN if at least one input value is NaN.
        * "raise": Raise a ValueError if at least one input value is NaN.

    Returns
    -------
    auc : float
        ROC-AUC of the given ground-truth and predictions.
    ci_left : float
        Left endpoint of the confidence interval.
    ci_right : float
        Right endpoint of the confidence interval.

    Notes
    -----
    The output always satisfies `0 <= ci_left <= auc <= ci_right <= 1`.

    See Also
    --------
    delong_test: Statistical test for the null hypothesis that the ROC-AUCs of two classifiers are equal.
    """
    y_true, y_hat = to_num_arrays(y_true, y_hat, arg_names=("y_true", "y_hat"), rank=(1,))
    if y_true.dtype.kind == "b":
        y_true = y_true.astype(np.float32)

    mask = np.isnan(y_true) | np.isnan(y_hat)
    if mask.any():
        if nan_policy == "omit":
            mask = ~mask
            y_true = y_true[mask]
            y_hat = y_hat[mask]
            if sample_weight is not None:
                sample_weight = sample_weight[mask]
        elif nan_policy == "propagate":
            return np.nan, np.nan, np.nan
        else:
            raise ValueError("Input of roc_auc_confidence_interval contains NaN.")

    auc, auc_cov = _delong_roc_variance(y_true, y_hat, sample_weight=sample_weight)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    if auc_std == 0.0:
        ci = (auc, auc)
    else:
        ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
        ci.clip(min=0, max=1, out=ci)
    return auc, ci[0], ci[1]


def _prepare_for_numeric_test(x, arg_name: str, method_name: str, nan_policy: str = "omit") -> tuple[np.ndarray, bool]:
    x = to_num_array(x, arg_name=arg_name, dtype=None, rank=(1,))
    if x.dtype.kind == "b":
        x = x.astype(np.float32)
    elif x.dtype.kind == "m":
        x = x / np.timedelta64(1, "s")
    elif x.dtype.kind == "M":
        x = (x - np.datetime64(0, "s")) / np.timedelta64(1, "s")
    elif x.dtype.kind not in "uif":
        raise ValueError(
            "Input `{}` of {} has non-numeric data type {}".format(
                "" if arg_name is None else arg_name, method_name, x.dtype.name
            )
        )
    try:
        mask = np.isnan(x)
    except:  # noqa
        is_na = False
    else:
        is_na = mask.any()
        if is_na:
            if nan_policy == "raise":
                raise ValueError(
                    "Input `{}` of {} contains NaN.".format("" if arg_name is None else arg_name, method_name)
                )
            elif nan_policy == "omit":
                x = x[~mask]
                is_na = False
    return x, is_na


def _prepare_for_category_test(
    x, arg_name: str, method_name: str, nan_policy: str = "omit"
) -> tuple[np.ndarray, np.ndarray, bool]:
    if hasattr(x, "value_counts"):
        x = x.value_counts(dropna=(nan_policy == "omit"))

        # with categorical data types, all categories are listed even if they don't actually appear; must be dropped
        x = x[x > 0]

        is_na = x.isna().any()
        xc = np.asarray(x)
        x = np.asarray(x.index)

        if is_na and nan_policy == "raise":
            raise ValueError("Input `{}` of {} contains NaN.".format("" if arg_name is None else arg_name, method_name))
    else:
        x = to_num_array(x, arg_name="x", rank=(1,), dtype=None)
        try:
            mask = np.isnan(x)
        except:  # noqa
            is_na = False
        else:
            is_na = mask.any()
            if is_na:
                if nan_policy == "raise":
                    raise ValueError(
                        "Input `{}` of {} contains NaN.".format("" if arg_name is None else arg_name, method_name)
                    )
                elif nan_policy == "omit":
                    x = x[~mask]
                    is_na = False
        x, xc = np.unique(x, return_counts=True)

    return x, xc, is_na


def _compute_midrank(x: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
    # AUC comparison adapted from https://github.com/Netflix/vmaf/
    jj = np.argsort(x)
    x_sorted = x[jj]
    if sample_weight is None:
        cumulative_weight = None
    else:
        cumulative_weight = np.cumsum(sample_weight[jj])
    n = len(x)
    t = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        if cumulative_weight is None:
            t[i:j] = 0.5 * (i + j - 1) + 1
        else:
            t[i:j] = cumulative_weight[i:j].mean()
        i = j
    out = np.empty(n, dtype=t.dtype)
    out[jj] = t
    return out


def _fast_delong(
    predictions_sorted_transposed: np.ndarray,
    n_positive: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """The fast version of DeLong's method for computing the covariance of unadjusted AUC [1].

    Parameters
    ----------
    predictions_sorted_transposed : np.ndarray
        2D array of shape `(n_classifiers, n_samples)` sorted such that positive samples are first.
    n_positive : int
        Number of positive samples.
    sample_weight : np.ndarray, optional
        Sample weights. None defaults to uniform weights.

    Returns
    -------
    aucs : np.ndarray
        ROC-AUC of each classifier, 1D array of shape `(n_classifiers,)`.
    sigma : np.ndarray
        Covariance matrix, 2D array of shape `(n_classifiers, n_classifiers)`.

    References
    ----------
    .. [1] Xu Sun and Weichao Xu. Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated
           Receiver Operating Characteristic Curves. IEEE Signal Processing Letters 21(11): 1389-1393, 2014.
    """
    # Short variables are named as they are in the paper

    m = n_positive
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    if sample_weight is None:
        positive_weight = negative_weight = None
    else:
        positive_weight = sample_weight[:m]
        negative_weight = sample_weight[m:]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)

    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :], sample_weight=positive_weight)
        ty[r, :] = _compute_midrank(negative_examples[r, :], sample_weight=negative_weight)
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :], sample_weight=sample_weight)

    if sample_weight is None:
        total_positive_weights = m
        total_negative_weights = n
        aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) * 0.5) / n
    else:
        total_positive_weights = positive_weight.sum()
        total_negative_weights = negative_weight.sum()
        pair_weights = np.dot(positive_weight[..., np.newaxis], negative_weight[np.newaxis])
        total_pair_weights = pair_weights.sum()

        aucs = (positive_weight * (tz[:, :m] - tx)).sum(axis=1) / total_pair_weights

    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / total_positive_weights

    return aucs, np.cov(v01) / m + np.cov(v10) / n


def _calc_pvalue(auc_1: float, auc_2: float, sigma: np.ndarray) -> float:
    v = np.array([[1, -1]])  # (1, 2)
    z = np.abs(auc_1 - auc_2) / (np.sqrt(np.dot(np.dot(v, sigma), v.T)) + 1e-8)[0, 0]
    return 2 * (1 - stats.norm.cdf(z))


def _compute_ground_truth_statistics(y_true: np.ndarray, sample_weight=None):
    if not np.isin(y_true, np.arange(2, dtype=y_true.dtype)).all():
        raise ValueError(
            "DeLong test only supports binary classification, where all values of `y_true` must be in {0, 1}."
        )

    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        sample_weight = to_num_array(sample_weight, arg_name="sample_weight", rank=(1,))
        if len(sample_weight) != len(y_true):
            raise ValueError(
                "Length of `sample_weight` differs from length of `y_true`: {}, {}".format(
                    len(sample_weight), len(y_true)
                )
            )
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def _delong_roc_variance(y_true: np.ndarray, y_hat: np.ndarray, sample_weight=None):
    order, label_1_count, ordered_sample_weight = _compute_ground_truth_statistics(y_true, sample_weight)
    predictions_sorted_transposed = y_hat[np.newaxis, order]
    aucs, delongcov = _fast_delong(
        predictions_sorted_transposed,
        label_1_count,
        sample_weight=ordered_sample_weight,
    )

    return aucs[0], delongcov


_tests = dict(
    chi_square=dict(
        function=chi_square,
        quantitative=False,
        n_groups=2,
        null_hypothesis="equal class frequencies",
        permutation_invariance=False,
    ),
    pearson=dict(
        function=stats.pearsonr,
        task="correlation",
        quantitative=True,
        paired=True,
        n_groups=2,
        normal=True,
        null_hypothesis="no linear correlation",
        permutation_invariance=True,
    ),
    spearman=dict(
        function=stats.spearmanr,
        task="correlation",
        quantitative=True,
        paired=True,
        n_groups=2,
        null_hypothesis="no monotonic correlation",
        permutation_invariance=True,
    ),
    t_test_rel=dict(
        function=stats.ttest_rel,
        task="comparison",
        quantitative=True,
        paired=True,
        n_groups=2,
        null_hypothesis="equal mean",
        permutation_invariance=True,
        comment="The differences between observations must be normally distributed, not the observations themselves.",
    ),
    wilcoxon=dict(
        function=stats.wilcoxon,
        task="comparison",
        quantitative=True,
        paired=True,
        n_groups=2,
        null_hypothesis="differences are symmetric about 0",
        permutation_invariance=True,
    ),
    t_test_ind=dict(
        function=stats.ttest_ind,
        task="comparison",
        quantitative=True,
        paired=False,
        n_groups=2,
        normal=True,
        equal_variance=True,
        null_hypothesis="equal mean",
        permutation_invariance=True,
    ),
    mann_whitney=dict(
        function=mann_whitney_u,
        task="comparison",
        quantitative=True,
        n_groups=2,
        null_hypothesis="equal median",
        permutation_invariance=True,
    ),
    kruskal_wallis=dict(
        function=stats.kruskal,
        task="comparison",
        quantitative=True,
        n_groups=(2, 3),
        null_hypothesis="equal median",
        permutation_invariance=True,
    ),
    oneway_anova=dict(
        function=stats.f_oneway,
        task="comparison",
        quantitative=True,
        n_groups=(2, 3),
        normal=True,
        equal_variance=True,
        null_hypothesis="equal mean",
        permutation_invariance=True,
        comment="According to the docs, the null hypothesis is that at least two groups have equal mean."
        " However, in fact all groups must have the same mean.",
    ),
    alexander_govern=dict(
        function=stats.alexandergovern,
        task="comparison",
        quantitative=True,
        n_groups=(2, 3),
        normal=True,
        null_hypothesis="equal mean",
        permutation_invariance=True,
    ),
    bartlett=dict(
        function=stats.bartlett,
        task="comparison",
        quantitative=True,
        n_groups=(2, 3),
        normal=True,
        null_hypothesis="equal variance",
        permutation_invariance=True,
    ),
    levene=dict(
        function=stats.levene,
        task="comparison",
        quantitative=True,
        n_groups=(2, 3),
        null_hypothesis="equal variance",
        permutation_invariance=True,
    ),
    delong=dict(
        function=delong_test,
        task="comparison",
        quantitative=True,
        paired=True,
        n_groups=2,
        null_hypothesis="equal ROC-AUC",
        permutation_invariance=True,
    ),
)


def suggest_test(
    task: str = "comparison",
    quantitative: bool = True,
    paired: bool = False,
    n_groups: int = 2,
    normal: bool = False,
    equal_variance: bool = False,
) -> dict:
    """Suggest statistical hypothesis tests for comparing two or more groups (samples). The list of suggested tests is
    by no means exhaustive, but includes some of the most frequently used tests in practice.

    See Notes for some general comments on statistical testing.

    Parameters
    ----------
    task : str, default="comparison"
        The objective of the test, can be either "comparison" or "correlation":

        * "comparison": The objective of the test is to determine whether the given groups were drawn from the same
            distribution. This usually, but not necessarily, happens by comparing group statistics, like mean, median
            or variance.
        * "correlation": The objective of the test is to determine whether the given (paired) groups are correlated.
            Groups can be correlated even when drawn from distinct distributions.

    quantitative : bool, default=True
        The observations in the given groups are quantitative, i.e., drawn from continuous or discrete distributions,
        such that each observation has a numerical value.
        The alternative are categorical observations.
    paired : bool, default=False
        The observations in the given groups are paired, i.e., the `i`-th observation in the first group corresponds
        with the `i`-observation in the second group. Correspondence can mean, for instance, that observations
        originate from the same subject, measurement device, etc. Note that this implies that all groups must have the
        same size.
        The alternative are independent groups.
    n_groups : int, default=2
        The number of groups the test should handle. Some tests are restricted to two groups, others can handle
        arbitrarily many groups.
    normal : bool, default=True
        The observations are known to be drawn from a normal distribution. Some tests need this assumption to work
        properly, others (called "non-parametric tests") can deal with arbitrary underlying distributions.
    equal_variance : bool, default=False
        The observations are known to be drawn from distributions with equal variance (usually normal distributions).
        Some tests need this assumption to work properly, others can deal with arbitrary variances.
        This property is also known as homoscedasticity.

    Returns
    -------
    dict
        Dict of suggested tests (possibly empty), keys are names and values are dicts with main properties.
        Carefully read the documentation of each test to select the one appropriate for your data.

    Notes
    -----
    A statistical test is usually performed by find evidence _against_ the null hypothesis of the test, e.g., using the
    t-test to show that two groups have _different_ mean values. The converse is not true, though: if a test does not
    produce evidence against the null hypothesis, we cannot conclude that the null-hypothesis must be true -- only that
    we have not found any evidence against it. This holds true even if the p-values are close to 1.
    More concisely: null hypothesis true ==> (relatively) large p-value. Note the implication, not equivalence!

    One common assumption of most statistical tests is that all observations in a group are independent, i.e., all are
    drawn independently from the same underlying distribution (i.i.d. assumption). Whether is property holds true also
    _between_ groups can be controlled with parameter `paired`.

    There are many resources for finding the right statistical test on the internet, e.g., _[1].

    References
    ----------
    .. [1] https://www.scribbr.com/statistics/statistical-tests/
    """
    if n_groups > 2:
        n_groups = 3

    out = {}
    for k, v in _tests.items():
        if task is not None:
            t = v.get("task")
            if t is not None:
                if not isinstance(t, (list, tuple)):
                    t = [t]
                if task not in t:
                    continue

        if quantitative is not None:
            if v.get("quantitative") not in (quantitative, None):
                continue

        if paired is not None:
            if v.get("paired") not in (paired, None):
                continue

        if n_groups is not None:
            n = v.get("n_groups")
            if n is not None:
                if not isinstance(n, (list, tuple)):
                    n = [n]
                if n_groups not in n:
                    continue

        if normal is not None:
            if v.get("normal") not in (normal, None):
                continue

        if equal_variance is not None:
            if v.get("equal_variance") not in (equal_variance, None):
                continue

        out[k] = v

    return out
