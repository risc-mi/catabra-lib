#  Copyright (c) 2025. RISC Software GmbH.
#  All rights reserved.

from typing import Optional, Union

import numpy as np
from matplotlib import pyplot as plt  # type: ignore

from ..util import to_num_array, to_num_arrays
from . import _common


def get_colormap(cmap, name=None):
    from matplotlib.colors import (  # noqa F821 # type: ignore
        Colormap,
        LinearSegmentedColormap,
    )

    if isinstance(cmap, str):
        try:
            return plt.get_cmap(cmap)
        except:  # noqa
            pass

        try:
            from ._plotly import PlotlyColorMap

            return PlotlyColorMap(cmap).to_matplotlib(LinearSegmentedColormap, name=name or cmap)
        except:  # noqa
            pass

        try:
            data = _common.load_saved_colormap(cmap)
        except:  # noqa
            pass
        else:
            return get_colormap(data, name=name or cmap)

    elif isinstance(cmap, Colormap):
        return cmap
    elif isinstance(cmap, dict):
        if "colors" in cmap:
            data = cmap
            cmap = LinearSegmentedColormap(
                name or "CUSTOM_CMAP",
                {
                    "red": [(scale, color[0], color[0]) for scale, color in data["colors"]],
                    "green": [(scale, color[1], color[1]) for scale, color in data["colors"]],
                    "blue": [(scale, color[2], color[2]) for scale, color in data["colors"]],
                    "alpha": [(scale, color[3], color[3]) for scale, color in data["colors"]],
                },
            )
            c = data.get("bad")
            if c is not None:
                cmap.set_bad(c)
            c = data.get("over")
            if c is not None:
                cmap.set_over(c)
            c = data.get("under")
            if c is not None:
                cmap.set_under(c)
            return cmap
        else:
            return LinearSegmentedColormap(name or "CUSTOM_CMAP", cmap)
    else:
        try:
            from ._plotly import PlotlyColorMap
        except:  # noqa
            pass
        else:
            if isinstance(cmap, PlotlyColorMap):
                return cmap.to_matplotlib(LinearSegmentedColormap, name=name)

    raise ValueError("The given colormap does not represent a valid matplotlib colormap.")


def training_history(
    x,
    ys,
    title: Optional[str] = "Training History",
    ax=None,
    figsize="auto",
    legend=None,
    **kwargs,
):
    """Plot training history, with timestamps on x- and metrics on y-axis.

    Parameters
    ----------
    x : array-like
        Timestamps, array-like of shape `(n,)`.
    ys : list of array-like
        Metrics, list of array-likes of shape `(n,)`.
    title : str, optional
        The title of the figure.
    ax : optional
        An existing axis, or None.
    figsize : default="auto".
        Figure size.
    legend : list of str, optional
        Names of the individual curves. None or a list of string with the same length as `ys`.

    Returns
    -------
    Matplotlib figure object.
    """
    if not isinstance(ys, list):
        ys = [ys]
    if ax is None:
        if figsize == "auto":
            figsize = (np.log10(len(x) + 1) + 5, 5)
        _, ax = plt.subplots(figsize=figsize)

    x_and_ys = to_num_arrays(x, *ys, rank=(1,), dtype=None)
    x = x_and_ys[0]
    ys = x_and_ys[1:]
    if x.dtype.kind == "m":
        unit, uom = _common.convert_timedelta(x)
        x = x / unit
        unit_suffix = f" [{uom}]"
    else:
        unit_suffix = ""

    y_scale = "linear"
    if ys:
        if legend is None:
            legend = [None] * len(ys)
        elif isinstance(legend, str):
            legend = [legend]
        assert len(legend) == len(ys)
        for y, lbl in zip(ys, legend):
            ax.plot(x, y, label=lbl, marker=".")
        if any(lbl is not None for lbl in legend):
            ax.legend(loc="best")
        maxes = [np.abs(y).max() for y in ys]
        maxes = [np.log10(m) for m in maxes if m > 0]
        if min(maxes) + 1 < max(maxes):
            # logarithmic scale
            y_scale = "symlog" if any((y < 0).any() for y in ys) else "log"

    ax.set(
        xlabel="Time" + unit_suffix,
        ylabel="Metric",
        yscale=y_scale,
        title=_common.make_title(None, title),
    )
    plt.close(ax.figure)
    return ax.figure


def regression_scatter(
    y_true,
    y_hat,
    sample_weight=None,
    name: Optional[str] = None,
    title: Optional[str] = "Truth-Prediction Plot",
    cmap="Blues",
    ax=None,
    figsize="auto",
    **kwargs,
):
    """Create a scatter plot with results of a regression task.

    Parameters
    ----------
    y_true : array-like
        Ground truth, array-like of shape `(n,)`. May contain NaN values.
    y_hat: array-like
        Predictions, array-like of shape `(n,)`. Must have same data type as `y_true`, and may also contain NaN values.
    sample_weight : array-like, optional
        Sample weights.
    name : str, optional
        Name of the target variable.
    title : str, default="Truth-Prediction Plot"
        The title of the resulting figure.
    cmap : str | colormap, default="Blues"
        The color map for coloring points according to `sample_weight`.
    ax
        An existing axis, or None.
    figsize : tuple | str, default="auto"
        Figure size. If "auto", the optimal figure size is determined automatically.

    Returns
    -------
    Matplotlib figure object.
    """
    y_true, y_hat = to_num_arrays(y_true, y_hat, arg_names=("y_true", "y_hat"), rank=(1,), dtype=None)
    assert y_true.dtype == y_hat.dtype
    if ax is None:
        if figsize == "auto":
            figsize = int(round(np.log10(len(y_true) + 1))) + 3
            figsize = (figsize, figsize)
        _, ax = plt.subplots(figsize=figsize)

    if y_true.dtype.kind == "m":
        unit, uom = _common.convert_timedelta(y_true)
        y_true = y_true / unit
        y_hat = y_hat / unit
        unit_suffix = f" [{uom}]"
    else:
        unit_suffix = ""

    d_min = min(y_true.min(), y_hat.min())
    d_max = max(y_true.max(), y_hat.max())
    ax.set_aspect(1.0)
    ax.plot([d_min, d_max], [d_min, d_max], ls="--", c="gray", lw=1.0)
    if sample_weight is None:
        ax.scatter(y_true, y_hat, marker=".", rasterized=len(y_true) > 500)
    else:
        sample_weight = to_num_array(sample_weight, arg_name="sample_weight", rank=(1,))
        cmap = get_colormap(cmap)
        s = ax.scatter(
            y_true,
            y_hat,
            marker=".",
            rasterized=len(y_true) > 500,
            c=sample_weight,
            cmap=cmap,
        )
        cb = plt.colorbar(s, ax=ax, shrink=0.82)
        cb.set_label("Sample weight")

    ax.set(
        ylabel="Predicted label" + unit_suffix,
        xlabel="True label" + unit_suffix,
        title=_common.make_title(name, title),
    )
    plt.close(ax.figure)
    return ax.figure


def confusion_matrix(
    cm,
    class_names: Optional[list] = None,
    name: Optional[str] = None,
    title: Optional[str] = "Confusion Matrix",
    cmap="Blues",
    ax=None,
    figsize="auto",
    **kwargs,
):
    """Plot a confusion matrix.

    Parameters
    ----------
    cm : array-like
        The confusion matrix to plot. An array-like of shape `(n, n)` whose rows correspond to true classes and whose
        columns correspond to predicted classes.
    class_names : list of str, optional
        Class names, list of string of length `n`. If the last class is called "__total__", it is assumed to contain
        row- or column totals.
        Ignored if `cm` is a pandas DataFrame, in which case class names are inferred from the row/column index.
    name : str, optional
        Name of the target variable.
    title : str, default="Confusion Matrix"
        The title of the resulting figure.
    cmap : colormap | str, default="Blues"
        The color map.
    ax
        An existing axis, or None.
    figsize : tuple | str, default="auto"
        Figure size. If "auto", the optimal figure size is determined automatically.

    Returns
    -------
    Matplotlib figure object.
    """

    cm, class_names = _common.validate_confusion_matrix(cm, class_names)
    n = len(cm) - 1

    if ax is None:
        if figsize == "auto":
            figsize = (n + 2) if n < 7 else n + 1
            figsize = (figsize, figsize)
        _, ax = plt.subplots(figsize=figsize)

    cm_rel = 2 * cm / np.maximum(1, cm.sum(axis=1, keepdims=True))
    cm_rel[:, -1] = 0.95
    cm_rel[-1] = 0.95
    cm_rel[-1, -1] = 1.0
    cmap = get_colormap(cmap)
    im = ax.imshow(cm_rel, interpolation="nearest", cmap=cmap)
    thresh = (cm_rel.max() + cm_rel.min()) / 2.0
    fmt = ".2f" if cm.dtype.kind == "f" else "d"

    cmap_min, cmap_max = im.cmap(0.0), im.cmap(1.0)

    for i in range(n):
        for j in range(n):
            text_cm = format(cm[i, j], fmt)  # + '\n' + format(100 * cm_rel[i, j], '.2f') + '%'
            color = cmap_max if cm_rel[i, j] < thresh else cmap_min
            ax.text(j, i, text_cm, ha="center", va="center", color=color, weight="bold")

        text_cm = format(cm[i, -1], fmt)
        if cm[i, -1] > 0:
            text_cm += "\n" + format(100 * cm_rel[i, i], ".2f") + "%"
        ax.text(n, i, text_cm, ha="center", va="center", color=cmap_min, weight="bold")

    for j in range(n):
        text_cm = format(cm[-1, j], fmt)
        if cm[-1, j] > 0:
            text_cm += "\n" + format(100 * cm[j, j] / cm[-1, j], ".2f") + "%"
        ax.text(j, n, text_cm, ha="center", va="center", color=cmap_min, weight="bold")

    text_cm = format(cm[-1, -1], fmt) + "\n" + format(100 * cm[:-1, :-1].trace() / cm[-1, -1], ".2f") + "%"
    ax.text(n, n, text_cm, ha="center", va="center", color=cmap_min, weight="bold")

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=_common.make_title(name, title),
    )
    ax.set_ylim((n + 0.5, -0.5))
    plt.setp(
        ax.get_xticklabels(),
        rotation="45" if any(len(str(c)) > 10 for c in class_names) else "horizontal",
    )
    plt.close(ax.figure)
    return ax.figure


def roc_pr_curve(
    xs,
    ys,
    deviation=None,
    sample_weight=None,
    from_predictions: bool = False,
    name: Optional[str] = None,
    title: Optional[str] = "auto",
    ax=None,
    figsize=(5, 5),
    roc: bool = True,
    legend=None,
    positive_prevalence: float = -1.0,
    **kwargs,
):
    """Plot ROC or PR curve(s).

    Parameters
    ----------
    xs : list of array-like
        If `from_predictions` is False: x-coordinates of the curve(s), either a single array-like or a list thereof.
        In the case of ROC curves they correspond to the false positive rates, in the case of PR curves they correspond
        to recall.
        If `from_predictions` is True: Ground-truth labels, either a single array-like with values in {0, 1}, or a list
        thereof.
    ys : list of array-like
        If `from_predictions` is False: y-coordinates of the curve(s), either a single array-like or a list thereof.
        In the case of ROC curves they correspond to the true positive rates, in the case of PR curves they correspond
        to precision.
        If `from_predictions` is True: Predictions, either a single array-like or a list thereof.
    deviation : list of array-like, optional
        y-coordinates of deviations of the curve(s), indicating errors, standard deviations, confidence intervals, and
        the like. None or a list of array-likes of shape `(2, n)`.
        Must be None if `from_predictions` is True.
    sample_weight : list of array-like, optional
        Sample weight, list of array-likes.
        Must be None if `from_predictions` is False.
    from_predictions : bool, default=False
        Interpret `xs` and `ys` as ground-truth and predictions, respectively.
    name : str, optional
        Name of the target variable.
    title : str, default="auto"
        The title of the figure. If "auto", the title is either "Receiver Operating Characteristic" or "Precision-Recall
        Curve" depending on the value of `roc`.
    ax
        An existing axis, or None.
    figsize : tuple, default=(5, 5)
        Figure size.
    roc : bool, default=True
        Whether ROC- or PR curves are plotted.
    legend : list of str, optional
        Names of the individual curves. None or a list of string with the same length as `xs` and `ys`.
    positive_prevalence: float, default=-1
        Prevalence of the positive class. Only relevant for PR curves.

    Returns
    -------
    Matplotlib figure object.
    """
    xs, ys, deviation = _common.validate_roc_pr(xs, ys, deviation, sample_weight, from_predictions, roc)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if xs:
        if legend is None:
            legend = [None] * len(ys)
        elif isinstance(legend, str):
            legend = [legend]
        assert len(legend) == len(ys)
        x_min = min(x.min() for x in xs)
        x_max = max(x.max() for x in xs)
        y_min = min(y.min() for y in ys)
        y_max = max(y.max() for y in ys)
        ax.set_xlim(min(-0.05, x_min), max(1.05, x_max))
        ax.set_ylim(min(-0.05, y_min), max(1.05, y_max))
        ax.set_aspect(1.0)
        if roc:
            ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1.0)
        elif positive_prevalence >= 0.0:
            # unfortunately, we cannot rely on `ys[0][0]` being the positive prevalence
            ax.plot(
                [0, 1],
                [positive_prevalence, positive_prevalence],
                ls="--",
                c="gray",
                lw=1.0,
            )
        for x, y, dv, lbl in zip(xs, ys, deviation, legend):
            line = ax.plot(x, y, label=lbl)[0]
            if dv is not None:
                ax.fill_between(x, dv[0], dv[1], alpha=0.2, facecolor=line.get_color())
        if any(lbl is not None for lbl in legend):
            ax.legend(loc="lower right" if roc else "best")

    if title == "auto":
        title = "Receiver Operating Characteristic" if roc else "Precision-Recall Curve"
    ax.set(
        xlabel="False positive rate" if roc else "Recall",
        ylabel="True positive rate" if roc else "Precision",
        title=_common.make_title(name, title),
    )
    plt.close(ax.figure)
    return ax.figure


def threshold_metric_curve(
    th,
    ys,
    threshold: Optional[float] = None,
    name: Optional[str] = None,
    title: Optional[str] = "Threshold-Metric Plot",
    ax=None,
    figsize=(10, 5),
    legend=None,
):
    """Plot threshold-vs.-metric curves, with thresholds on the x- and corresponding thresholded metrics on the y-axis.

    Parameters
    ----------
    th : array-like
        The thresholds, a single array-like of shape `(n,)`.
    ys : list of array-like
        The y-coordinates of the curve(s), either a single array-like of shape `(n,)` or a list thereof.
    threshold : float, optional
        Actual decision threshold used for making predictions, or None. If given, a dashed vertical line is drawn to
        indicate the threshold.
    name : str, optional
        Name of the target variable.
    title : str, default="Threshold-Metric Plot"
        The title of the figure.
    ax
        An existing axis, or None.
    figsize : tuple, default=(10,5)
        Figure size.
    legend : list of str, optional
        Names of the individual curves. None or a list of string with the same length as `ys`.

    Returns
    -------
    Matplotlib figure object.
    """

    if not isinstance(ys, list):
        ys = [ys]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if ys:
        th_and_ys = to_num_arrays(th, *ys, rank=(1,))
        th = th_and_ys[0]
        ys = th_and_ys[1:]
        if legend is None:
            legend = [None] * len(ys)
        elif isinstance(legend, str):
            legend = [legend]
        assert len(legend) == len(ys)
        x_min = th.min()
        x_max = th.max()
        y_min = min(y.min() for y in ys)
        y_max = max(y.max() for y in ys)
        ax.set_xlim(min(-0.05, x_min), max(1.05, x_max))
        ax.set_ylim(min(-0.05, y_min), max(1.05, y_max))
        if threshold is not None:
            ax.axvline(threshold, ls="--", c="gray", lw=1.0)
        for y, lbl in zip(ys, legend):
            ax.plot(th, y, label=lbl)
        if any(lbl is not None for lbl in legend):
            ax.legend(loc="best")

    ax.set(xlabel="Threshold", ylabel="Metric", title=_common.make_title(name, title))
    plt.close(ax.figure)
    return ax.figure


def calibration_curve(
    th_lower,
    th_upper,
    ys,
    name: Optional[str] = None,
    deviation=None,
    title: Optional[str] = "Calibration Curve",
    ax=None,
    figsize=(5, 5),
    legend=None,
    **kwargs,
):
    """Plot calibration curves.

    Parameters
    ----------
    th_lower : array-like
        Lower/left ends of threshold bins, array-like of shape `(n,)`.
    th_upper : array-like
        Upper/right ends of threshold bins, array-like of shape `(n,)`.
    ys : list of array-like
        y-coordinates of the curve(s), either a single array-like of shape `(n,)` or a list thereof.
    deviation : list of array-like, optional
        y-coordinates of deviations of the curve(s), indicating errors, standard deviations, confidence intervals, and
        the like. None or a list of array-likes of shape `(2, n)`.
    name : str, optional
        Name of the target variable.
    title : str, default="Calibration Curve"
        The title of the figure.
    ax
        An existing axis, or None.
    figsize : tuple, default=(5, 5)
        Figure size.
    legend : list of str, optional
        Names of the individual curves. None or a list of strings with the same length as `ys`.

    Returns
    -------
    Matplotlib figure object.
    """
    if not isinstance(ys, list):
        ys = [ys]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if ys:
        th_and_ys = to_num_arrays(th_lower, th_upper, *ys, rank=(1,))
        th_lower = th_and_ys[0]
        th_upper = th_and_ys[1]
        ys = th_and_ys[2:]
        if legend is None:
            legend = [None] * len(ys)
        elif isinstance(legend, str):
            legend = [legend]
        if deviation is None:
            deviation = [None] * len(ys)
        elif not isinstance(deviation, list):
            deviation = [deviation]
        assert len(deviation) == len(ys)
        assert len(legend) == len(ys)
        deviation = [dv if dv is None else to_num_array(dv, rank=(2,)) for dv in deviation]
        x = (th_lower + th_upper) * 0.5
        y_min = min(y.min() for y in ys)
        y_max = max(y.max() for y in ys)
        ax.set_ylim(min(-0.05, y_min), max(1.05, y_max))
        for y, dv, lbl in zip(ys, deviation, legend):
            line = ax.plot(x, y, label=lbl, marker=".")[0]
            if dv is not None:
                ax.fill_between(x, dv[0], dv[1], alpha=0.2, facecolor=line.get_color())
        if any(lbl is not None for lbl in legend):
            ax.legend(loc="best")

    ax.set(
        xlabel="Threshold",
        ylabel="Fraction of positive class",
        title=_common.make_title(name, title),
    )
    plt.close(ax.figure)
    return ax.figure


def beeswarm(
    values: "pandas.DataFrame",  # noqa F821 # type: ignore
    colors: Union["pandas.DataFrame", "pandas.Series", None] = None,  # noqa F821 # type: ignore
    color_name: Optional[str] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    ax=None,
    figsize="auto",
    cmap="red_blue",
    **kwargs,
):
    """Create a beeswarm plot, inspired by (and largely copied from) the shap package [1]. This plot type is very
    useful for displaying local feature importance scores for a set of samples.

    Parameters
    ----------
    values : DataFrame
        Values to plot, a pandas DataFrame whose columns correspond to the rows in the plot and whose rows correspond
        to individual samples.
    colors : DataFrame | Series, optional
        Colors of the points. A pandas DataFrame with the same shape and column names as `values`, or a pandas Series
        with the same number of rows as `values`.
    color_name : str, optional
        Name of the color bar; only relevant if `colors` is provided.
    title : str, optional
        Title of the figure.
    x_label : str, optional
        Label of the x-axis.
    ax
        An existing axis, or None.
    figsize : tuple | str, default="auto"
        Figure size or "auto".
    cmap : colormap | str, default="red_blue"
        Name of a color map.

    Returns
    -------
    Matplotlib figure object.

    References
    ----------
    .. [1] https://github.com/slundberg/shap
    """

    assert all(values[c].dtype.kind in "ifb" for c in values.columns)
    n_samples, n_features = values.shape
    row_height = 0.4
    if ax is None:
        if figsize == "auto":
            figsize = n_features * row_height + 1.5
            figsize = (8, figsize)
        _, ax = plt.subplots(figsize=figsize)

    color_ticks = ["Low", "High"]
    if colors is not None:
        assert len(colors) == len(values)
        if hasattr(colors, "cat"):  # `colors` is a Series with category data type
            color_ticks = colors.cat.categories
            colors = colors.cat.codes

    cmap = get_colormap(cmap)
    ax.axvline(x=0, color="#999999", zorder=-1)

    # beeswarm
    for pos, column in enumerate(reversed(values.columns)):
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        if len(np.shape(colors)) == 1:  # `np.shape(None)` is `()`
            col = colors
        elif hasattr(colors, "columns") and column in colors.columns:
            col = colors[column]
        else:
            col = None
        gray, colored = _common.beeswarm_feature(values[column], col, row_height)

        if gray is not None:
            if colors is None:
                col = cmap(0.0)
            else:
                col = "#777777"
            ax.scatter(
                gray[0],
                pos + gray[1],
                s=16,
                alpha=1,
                linewidth=0,
                zorder=3,
                color=col,
                rasterized=n_samples * n_features > 50000,
            )
        if colored is not None:
            ax.scatter(
                colored[0],
                pos + colored[1],
                cmap=cmap,
                c=colored[2],
                vmin=colored[3],
                vmax=colored[4],
                s=16,
                linewidth=0,
                alpha=1,
                zorder=3,
                rasterized=n_samples * n_features > 50000,
            )

    # color bar
    if colors is not None:
        import matplotlib.cm as cm  # type: ignore

        m = cm.ScalarMappable(cmap=cmap)
        m.set_array([0, 1])
        cb = plt.colorbar(m, ax=ax, ticks=np.linspace(0, 1, len(color_ticks)), aspect=1000)
        cb.set_ticklabels(color_ticks)
        if color_name is not None:
            cb.set_label(color_name, size=12, labelpad=0)
        cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.9) * 20)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params("y", length=20, width=0.5, which="major")

    ax.set(
        xlabel=x_label,
        yticks=range(n_features),
        yticklabels=reversed(values.columns),
        ylim=(-1, n_features),
        title=_common.make_title(None, title),
    )
    plt.close(ax.figure)
    return ax.figure


def horizontal_bar(
    values: Union["pandas.Series", "pandas.DataFrame"],  # noqa F821 # type: ignore
    groups: Optional[dict] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    ax=None,
    figsize="auto",
    cmap="red_blue",
    **kwargs,
):
    """Create a horizontal bar plot.

    Parameters
    ----------
    values : Series, DataFrame
        Values to plot, a pandas DataFrame whose rows correspond to the rows in the plot and whose columns correspond
        to grouped bars.
    groups : dict, optional
        Column groups. If given, group names must be mapped to lists of columns in `values`. For instance, if `values`
        has columns "label1_>0", "label1_<0" and "label2", then `groups` might be set to

        {
            "label1": ["label1_>0", "label1_<0"],
            "label2": ["label2"]
        }

    title : str, optional
        Title of the figure.
    x_label : str, optional
        Label of the x-axis.
    ax
        An existing axis, or None.
    figsize : tuple | str, default="auto"
        Figure size or "auto".
    cmap : colormap | str, default="red_blue"
        Name of a color map.

    Returns
    -------
    Matplotlib figure object.
    """

    n_features = len(values)
    if hasattr(values, "to_frame"):
        values = values.to_frame()
    if groups is None:
        groups = {c: [c] for c in values.columns}
    else:
        assert len(groups) >= 1
        assert all(all(c in values.columns for c in g) for g in groups.values())

    if ax is None:
        if figsize == "auto":
            figsize = (8, n_features * 0.5 * np.sqrt(len(groups)) + 1.5)
        _, ax = plt.subplots(figsize=figsize)

    negative_values_present = any((values[columns] < 0).any().any() for columns in groups.values())
    positive_values_present = any((values[columns] > 0).any().any() for columns in groups.values())
    cmap = get_colormap(cmap)
    neg_color = np.asarray(cmap(0.0)[:3])
    pos_color = np.asarray(cmap(1.0)[:3])

    if len(groups) > 1:
        if negative_values_present:
            color = "#777777" if positive_values_present else neg_color
        else:
            color = pos_color
        legend_handles = [ax.barh(1, [0.0], 0.0, label=g, color=color) for g in groups]
    else:
        legend_handles = None

    # draw the bars
    y_pos = np.arange(n_features, 0, -1)
    total_width = 0.7
    bar_width = total_width / len(groups)
    for i, (g, columns) in enumerate(groups.items()):
        ypos_offset = -((i - len(groups) / 2) * bar_width + bar_width / 2)
        for c in columns:
            ax.barh(
                y_pos + ypos_offset,
                values[c].values,
                bar_width,
                align="center",
                edgecolor=(1, 1, 1, 0.8),
                label=g,
                color=[neg_color if values[c].iloc[j] < 0 else pos_color for j in range(n_features)],
            )

    # add text
    xlen = ax.get_xlim()[1] - ax.get_xlim()[0]
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen / width
    for i, (g, columns) in enumerate(groups.items()):
        ypos_offset = -((i - len(groups) / 2) * bar_width + bar_width / 2)
        for c in columns:
            for j in range(n_features):
                v = values[c].iloc[j]
                if v <= 0:
                    if negative_values_present:
                        # only add text if there are strictly negative values
                        ax.text(
                            v - (5 / 72) * bbox_to_xscale,
                            y_pos[j] + ypos_offset,
                            "{:.2f}".format(v),
                            horizontalalignment="right",
                            verticalalignment="center",
                            color=neg_color,
                        )
                else:
                    ax.text(
                        v + (5 / 72) * bbox_to_xscale,
                        y_pos[j] + ypos_offset,
                        "+{:.2f}".format(v),
                        horizontalalignment="left",
                        verticalalignment="center",
                        color=pos_color,
                    )

    if len(groups) > 1 and n_features > 1:
        # add horizontal lines for each feature row
        for i in range(1, n_features):
            ax.axhline(i + 0.5, color="#888888", lw=0.5, dashes=(1, 5), zorder=-1)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if negative_values_present:
        ax.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)
        ax.spines["left"].set_visible(False)

    xmin, xmax = ax.get_xlim()

    if negative_values_present:
        ax.set_xlim(xmin - (xmax - xmin) * 0.05, xmax + (xmax - xmin) * 0.05)
    else:
        ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.05)

    if legend_handles is not None:
        ax.legend(handles=legend_handles, loc="best")

    ax.set(
        xlabel=x_label,
        yticks=y_pos,
        yticklabels=values.index,
        ylim=(0.5, n_features + 0.5),
        title=_common.make_title(None, title),
    )
    plt.close(ax.figure)
    return ax.figure
