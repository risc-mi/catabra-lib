#  Copyright (c) 2025. RISC Software GmbH.
#  All rights reserved.

from functools import partial
from typing import Optional, Union

import numpy as np
import plotly.colors
from plotly import graph_objects as go
from plotly import io as pio

from ..util import to_num_array, to_num_arrays
from . import _common


def get_colormap(cmap, name=None):
    if isinstance(cmap, str):
        try:
            return PlotlyColorMap(cmap)
        except:  # noqa
            pass

        try:
            from matplotlib import pyplot as plt  # noqa F821 # type: ignore

            return plt.get_cmap(cmap)  # mpl colormaps can be used as-is, since they return (arrays of) 4-tuples
        except:  # noqa
            pass

        try:
            data = _common.load_saved_colormap(cmap)
        except:  # noqa
            pass
        else:
            return get_colormap(data, name=name or cmap)

    elif isinstance(cmap, PlotlyColorMap):
        return cmap
    elif isinstance(cmap, list):
        return partial(PlotlyColorMap.sample_colorscale, cmap=cmap)
    elif isinstance(cmap, dict):
        return PlotlyColorMap(cmap)
    elif isinstance(cmap, callable):
        # this case covers mpl colormaps, too
        return cmap

    raise ValueError("The given colormap does not represent a valid plotly colormap.")


class PlotlyColorMap:
    def __init__(self, cmap: Union[str, dict]):
        if isinstance(cmap, str):
            self.cmap = getattr(plotly.colors.sequential, cmap)
            self.under = (0.0, 0.0, 0.0)
            self.over = (0.0, 0.0, 0.0)
            self.bad = (0.0, 0.0, 0.0)
        else:
            self.cmap = cmap["colors"]
            self.under = cmap.get("under", (0.0, 0.0, 0.0))[:3]
            self.over = cmap.get("over", (0.0, 0.0, 0.0))[:3]
            self.bad = cmap.get("bad", (0.0, 0.0, 0.0))[:3]

    def __call__(self, value: float) -> tuple:
        if isinstance(value, np.ndarray):
            return np.array([self.__call__(v) for v in value])
        elif isinstance(value, list):
            return [self.__call__(v) for v in value]
        elif np.isnan(value):
            return self.bad
        elif value < 0:
            return self.under
        elif value > 1:
            return self.over
        else:
            return PlotlyColorMap.sample_colorscale(cmap=self.cmap, value=value)

    def to_matplotlib(self, ctor, name=None):
        cmap = ctor(
            name or "CUSTOM_CMAP",
            {
                "red": [(scale, color[0], color[0]) for scale, color in self.cmap],
                "green": [(scale, color[1], color[1]) for scale, color in self.cmap],
                "blue": [(scale, color[2], color[2]) for scale, color in self.cmap],
                "alpha": [(scale, color[3], color[3]) for scale, color in self.cmap],
            },
        )
        cmap.set_bad(self.bad)
        cmap.set_under(self.under)
        cmap.set_over(self.over)

        return cmap

    @staticmethod
    def sample_colorscale(cmap=None, value: float = 0.0) -> tuple:
        return plotly.colors.sample_colorscale(cmap, samplepoints=[value], colortype="tuple")[0]

    @staticmethod
    def sample_discrete(i: int, template: Optional[str] = None) -> str:
        colorway = pio.templates[template or pio.templates.default]["layout"]["colorway"]
        value = colorway[i % len(colorway)].lstrip("#")
        hex_total_length = len(value)
        rgb_section_length = hex_total_length // 3
        return ",".join(
            [str(int(value[j : j + rgb_section_length], 16)) for j in range(0, hex_total_length, rgb_section_length)]
        )

    @staticmethod
    def to_rgb(c: tuple) -> str:
        return f"rgb({round(c[0] * 256)},{round(c[1] * 256)},{round(c[2] * 256)})"


def training_history(x, ys, title: Optional[str] = "Training History", legend=None, text=None, **kwargs):
    """Plot training history, with timestamps on x- and metrics on y-axis.

    Parameters
    ----------
    x : array-like
        Timestamps, array-like of shape `(n,)`.
    ys : list of array-like
        Metrics, list of array-likes of shape `(n,)`.
    title : str, optional
        The title of the figure.
    legend : list of str, optional
        Names of the individual curves. None or a list of string with the same length as `ys`.
    text : list of str, optional
        Additional information to display when hovering over a point. Same for all metrics. None or a list of strings
        with the same length as `x`.

    Returns
    -------
    Plotly figure object.
    """
    if not isinstance(ys, list):
        ys = [ys]
    fig = go.Figure()

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
        if isinstance(text, str):
            text = [text]
        assert len(legend) == len(ys)
        assert text is None or len(text) == len(x)
        for y, lbl in zip(ys, legend):
            fig.add_trace(go.Scatter(x=x, y=y, name=lbl, mode="lines+markers", text=text))
        maxes = [y.abs().max() for y in ys]
        maxes = [np.log10(m) for m in maxes if m > 0]
        if min(maxes) + 1 < max(maxes):
            # logarithmic scale
            # Unfortunately, plotly does not support "symlog", and workarounds are hacky.
            #   (https://community.plotly.com/t/unable-to-see-negative-log-axis-scatter-plot/1364/3)
            y_scale = "log"

    fig.update_layout(
        xaxis=dict(title="Time" + unit_suffix),
        yaxis=dict(title="Metric", type=y_scale),
        title=dict(text=_common.make_title(None, title), x=0.5, xref="paper"),
        showlegend=len(ys) > 1 or any(lbl is not None for lbl in legend),
        hovermode="x unified" if len(ys) > 1 else "closest",
    )
    return fig


def regression_scatter(
    y_true,
    y_hat,
    sample_weight=None,
    name: Optional[str] = None,
    title: Optional[str] = "Truth-Prediction Plot",
    cmap="Blues",
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

    Returns
    -------
    Plotly figure object.
    """

    if hasattr(y_true, "index"):
        text = y_true.index
    else:
        text = None

    y_true, y_hat = to_num_arrays(y_true, y_hat, arg_names=("y_true", "y_hat"), rank=(1,), dtype=None)
    assert y_true.dtype == y_hat.dtype
    if y_true.dtype.kind == "m":
        unit, uom = _common.convert_timedelta(y_true)
        y_true = y_true / unit
        y_hat = y_hat / unit
        unit_suffix = f" [{uom}]"
    else:
        unit_suffix = ""

    d_min = min(y_true.min(), y_hat.min())
    d_max = max(y_true.max(), y_hat.max())
    if sample_weight is None:
        try:
            import pandas as pd
            from plotly import express as px
        except ImportError:
            if text is None:
                text = ["y_true={}<br>y_hat={}".format(y_true[j], y_hat[j]) for j in range(len(y_true))]
            else:
                text = [
                    "ID={}<br>y_true={}<br>y_hat={}".format(text[j], y_true[j], y_hat[j]) for j in range(len(y_true))
                ]
            fig = go.Figure()
            fig.add_trace(
                go.Scattergl(
                    x=y_true,
                    y=y_hat,
                    mode="markers",
                    hovertemplate="%{text}",
                    text=text,
                    name="",
                )
            )
        else:
            df = pd.DataFrame(data=dict(y_true=y_true, y_hat=y_hat))
            if text is not None:
                df["ID"] = text
                text = "ID"
            fig = px.scatter(
                df,
                x="y_true",
                y="y_hat",
                text=text,
                marginal_x="histogram",
                marginal_y="histogram",
            )
            fig.update_traces(histnorm="probability", selector={"type": "histogram"})
    else:
        sample_weight = to_num_array(sample_weight, arg_name="sample_weight", rank=(1,))
        cmap = get_colormap(cmap)
        if text is None:
            text = [
                "y_true={}<br>y_hat={}<br>sample_weight={}".format(y_true[j], y_hat[j], sample_weight[j])
                for j in range(len(y_true))
            ]
        else:
            text = [
                "ID={}<br>y_true={}<br>y_hat={}<br>sample_weight={}".format(
                    text[j], y_true[j], y_hat[j], sample_weight[j]
                )
                for j in range(len(y_true))
            ]
        colors = cmap((sample_weight - sample_weight.min()) / (sample_weight.max() - sample_weight.min()))
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=y_true,
                y=y_hat,
                mode="markers",
                marker=dict(color=[PlotlyColorMap.to_rgb(c) for c in colors]),
                hovertemplate="%{text}",
                text=text,
                name="",
            )
        )
    fig.add_shape(
        type="line",
        line=dict(dash="dash", width=0.5),
        x0=d_min,
        y0=d_min,
        x1=d_max,
        y1=d_max,
    )

    fig.update_layout(
        title=dict(text=_common.make_title(name, title, sep="<br>"), x=0.5, xref="paper"),
        xaxis=dict(title="True label" + unit_suffix),
        yaxis=dict(scaleanchor="x", title="Predicted label" + unit_suffix),
    )
    return fig


def confusion_matrix(
    cm,
    class_names: Optional[list] = None,
    name: Optional[str] = None,
    title: Optional[str] = "Confusion Matrix",
    cmap="Blues",
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

    Returns
    -------
    Plotly figure object.
    """

    matrix, classes = _common.validate_confusion_matrix(cm, class_names)
    pred_totals = matrix[-1, :-1]
    true_totals = matrix[:-1, -1]
    matrix = matrix[:-1, :-1]  # get rid of row/column totals
    n = matrix.shape[0]
    n_samples = true_totals.sum()

    cm_rel = matrix / np.maximum(1, true_totals[..., np.newaxis])
    thresh = (cm_rel.max() + cm_rel.min()) / 2.0

    fig = go.Figure()
    coords = [i + 0.5 for i in range(n + 1)]
    mg = np.meshgrid(coords[:-1], coords[1:])
    x = list(mg[0].flatten())
    y = list(mg[1].flatten())
    z = matrix[::-1].flatten()

    part_acc = (
        [100 * matrix[i, i] / max(1, s) for i, s in enumerate(pred_totals)]
        + [100 * matrix[i, i] / max(1, s) for i, s in enumerate(true_totals)]
        + [100 * np.trace(matrix) / n_samples]
    )
    part_totals = list(pred_totals) + list(true_totals) + [n_samples]

    x += coords[:-1] + [coords[-1]] * (n + 1)
    y += coords[:1] * n + coords[1:][::-1] + coords[:1]

    mg = np.meshgrid(classes, classes)
    x_class = mg[0].flatten()
    y_class = mg[1][::-1].flatten()

    text = ["<b>{}</b><br>{:.2f}%".format(np.round(t, 2), 100 * t / n_samples) for t in z]
    text += [
        "<b>{}</b><br>{:.2f}%".format(np.round(t, 2), a) if t > 0 else "<b>{}</b>".format(np.round(t, 2))
        for t, a in zip(part_totals, part_acc)
    ]

    if n == 2:
        correct_names = ["NPV", "PPV", "Specificity", "Sensitivity"]
    else:
        correct_names = ["Correct"] * n * 2
    correct_names.append("Accuracy")
    hovertext = [f"True label: {j}<br>Predicted label: {i}<br>Count: {k}" for i, j, k in zip(x_class, y_class, z)]
    hovertext += [
        ("<b>{}</b><br>Count: {}<br>{}: {:.2f}%" if t > 0 else "<b>{}</b><br>Count: {}").format(
            "Total" if i == 2 * n else f"Predicted label: {classes[i]}" if i < n else f"True label: {classes[i - n]}",
            t,
            cn,
            a,
        )
        for i, (t, a, cn) in enumerate(zip(part_totals, part_acc, correct_names))
    ]

    cmap = get_colormap(cmap)
    cmap_min = PlotlyColorMap.to_rgb(cmap(0.0))
    cmap_max = PlotlyColorMap.to_rgb(cmap(1.0))

    text_color = [cmap_max if t < thresh * max(1, true_totals[n - i // n - 1]) else cmap_min for i, t in enumerate(z)]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            text=text,
            hovertext=hovertext,
            hoverinfo="text",
            mode="text",
            textfont=dict(color=text_color + ["White"] * (2 * n + 1)),
        )
    )

    # Set axes properties
    fig.update_xaxes(range=[0, n + 1], showgrid=False)
    fig.update_yaxes(range=[0, n + 1], showgrid=False)

    shapes = []
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            shapes.append(
                go.layout.Shape(
                    type="rect",
                    x0=j,
                    y0=n + 1 - i,
                    x1=j + 1,
                    y1=n - i,
                    line=dict(width=1, color="White"),
                    fillcolor=PlotlyColorMap.to_rgb(cmap(val / max(1, true_totals[i]))),
                    layer="below",
                )
            )
    for i in range(len(classes)):
        shapes.append(
            go.layout.Shape(
                type="rect",
                x0=i,
                y0=0,
                x1=i + 1,
                y1=1,
                line=dict(width=1, color="White"),
                fillcolor="DarkGray",
                layer="below",
            )
        )
        shapes.append(
            go.layout.Shape(
                type="rect",
                x0=n,
                y0=i + 1,
                x1=n + 1,
                y1=i + 2,
                line=dict(width=1, color="White"),
                fillcolor="DarkGray",
                layer="below",
            )
        )
    shapes.append(
        go.layout.Shape(
            type="rect",
            x0=n,
            y0=0,
            x1=n + 1,
            y1=1,
            line=dict(width=1, color="White"),
            fillcolor="Gray",
            layer="below",
        )
    )

    fig.update_layout(
        xaxis=dict(
            title="Predicted label",
            ticktext=classes,
            tickvals=coords[:-1],
            showline=False,
            zeroline=False,
            constrain="domain",
        ),
        yaxis=dict(
            title="True label",
            ticktext=list(reversed(classes)),
            tickvals=coords[1:],
            scaleanchor="x",
            scaleratio=1,
            showline=False,
            zeroline=False,
        ),
        shapes=shapes,
        hovermode="closest",
        title=dict(text=_common.make_title(name, title, sep="<br>"), x=0.5, xref="paper"),
    )

    return fig


def roc_pr_curve(
    xs,
    ys,
    deviation=None,
    sample_weight=None,
    from_predictions: bool = False,
    name: Optional[str] = None,
    title: Optional[str] = "auto",
    roc: bool = True,
    legend=None,
    deviation_legend=None,
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
    roc : bool, default=True
        Whether ROC- or PR curves are plotted.
    legend : list of str, optional
        Names of the individual curves. None or a list of string with the same length as `xs` and `ys`.
    deviation_legend : list of str, optional
        Names of the deviation curves. None or a list of strings with the same length as `xs` and `ys`.
    positive_prevalence: float, default=-1
        Prevalence of the positive class. Only relevant for PR curves.

    Returns
    -------
    Plotly figure object.
    """
    xs, ys, deviation = _common.validate_roc_pr(xs, ys, deviation, sample_weight, from_predictions, roc)
    fig = go.Figure()
    if xs:
        if legend is None:
            legend = [None] * len(ys)
        elif isinstance(legend, str):
            legend = [legend]
        if deviation_legend is None:
            deviation_legend = [None] * len(ys)
        elif isinstance(deviation_legend, str):
            deviation_legend = [deviation_legend]
        assert len(deviation_legend) == len(ys)
        assert len(legend) == len(ys)
        x_min = min(x.min() for x in xs)
        x_max = max(x.max() for x in xs)
        y_min = min(y.min() for y in ys)
        y_max = max(y.max() for y in ys)
        if roc:
            fig.add_shape(type="line", line=dict(dash="dash", width=0.5), x0=0, y0=0, x1=1, y1=1)
        elif positive_prevalence >= 0.0:
            # unfortunately, we cannot rely on `ys[0][0]` being the positive prevalence
            fig.add_shape(
                type="line",
                line=dict(dash="dash", width=0.5),
                x0=0,
                y0=positive_prevalence,
                x1=1,
                y1=positive_prevalence,
            )
        for i, (x, y, dv, lbl, dv_lbl) in enumerate(zip(xs, ys, deviation, legend, deviation_legend)):
            color = PlotlyColorMap.sample_discrete(i)
            if dv is not None:
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([dv[1], dv[0, ::-1]]),
                        fill="toself",
                        fillcolor="rgba(" + color + ",0.2)",
                        line_color="rgba(255,255,255,0)",
                        name=dv_lbl,
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=lbl,
                    mode="lines",
                    line=dict(color="rgb(" + color + ")"),
                )
            )
    else:
        x_min = x_max = y_min = y_max = 0

    if title == "auto":
        title = "Receiver Operating Characteristic" if roc else "Precision-Recall Curve"
    fig.update_layout(
        xaxis=dict(
            title="False positive rate" if roc else "Recall",
            range=[min(-0.05, x_min), max(1.05, x_max)],
            constrain="domain",
        ),
        yaxis=dict(
            title="True positive rate" if roc else "Precision",
            scaleanchor="x",
            range=[min(-0.05, y_min), max(1.05, y_max)],
            constrain="domain",
        ),
        title=dict(text=_common.make_title(name, title, sep="<br>"), x=0.5, xref="paper"),
        showlegend=len(xs) > 1
        or any(lbl is not None for lbl in legend)
        or any(lbl is not None for lbl in deviation_legend),
    )
    return fig


def threshold_metric_curve(
    th,
    ys,
    threshold: Optional[float] = None,
    name: Optional[str] = None,
    title: Optional[str] = "Threshold-Metric Plot",
    legend=None,
    **kwargs,
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
    legend : list of str, optional
        Names of the individual curves. None or a list of string with the same length as `ys`.

    Returns
    -------
    Plotly figure object.
    """
    if not isinstance(ys, list):
        ys = [ys]
    fig = go.Figure()
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
        if threshold is not None:
            fig.add_shape(
                type="line",
                line=dict(dash="dash", width=0.5),
                x0=threshold,
                y0=min(0, y_min),
                x1=threshold,
                y1=max(1, y_max),
            )
        for y, lbl in zip(ys, legend):
            fig.add_trace(go.Scatter(x=th, y=y, name=lbl, mode="lines"))
    else:
        x_min = x_max = y_min = y_max = 0

    fig.update_layout(
        xaxis=dict(
            title="Threshold",
            range=[min(-0.05, x_min), max(1.05, x_max)],
            constrain="domain",
        ),
        yaxis=dict(
            title="Metric",
            range=[min(-0.05, y_min), max(1.05, y_max)],
            constrain="domain",
        ),
        title=dict(text=_common.make_title(name, title, sep="<br>"), x=0.5, xref="paper"),
        showlegend=len(ys) > 1 or any(lbl is not None for lbl in legend),
        hovermode="x unified" if len(ys) > 1 else "closest",
    )
    return fig


def calibration_curve(
    th_lower,
    th_upper,
    ys,
    name: Optional[str] = None,
    deviation=None,
    title: Optional[str] = "Calibration Curve",
    legend=None,
    deviation_legend=None,
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
    legend : list of str, optional
        Names of the individual curves. None or a list of strings with the same length as `ys`.
    deviation_legend : list of str, optional
        Names of the deviation curves. None or a list of strings with the same length as `ys`.

    Returns
    -------
    Plotly figure object.
    """
    if not isinstance(ys, list):
        ys = [ys]
    fig = go.Figure()
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
        if deviation_legend is None:
            deviation_legend = [None] * len(ys)
        elif isinstance(deviation_legend, str):
            deviation_legend = [deviation_legend]
        assert len(deviation) == len(ys)
        assert len(deviation_legend) == len(ys)
        assert len(legend) == len(ys)
        deviation = [dv if dv is None else to_num_array(dv, rank=(2,)) for dv in deviation]
        x = (th_lower + th_upper) * 0.5
        y_min = min(y.min() for y in ys)
        y_max = max(y.max() for y in ys)
        for i, (y, dv, lbl, dv_lbl) in enumerate(zip(ys, deviation, legend, deviation_legend)):
            color = PlotlyColorMap.sample_discrete(i)
            if dv is not None:
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([dv[1], dv[0, ::-1]]),
                        fill="toself",
                        fillcolor="rgba(" + color + ",0.2)",
                        line_color="rgba(255,255,255,0)",
                        name=dv_lbl,
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=lbl,
                    mode="lines+markers",
                    line=dict(color="rgb(" + color + ")"),
                )
            )
    else:
        y_min = y_max = 0

    fig.update_layout(
        xaxis=dict(title="Threshold", constrain="domain"),
        yaxis=dict(
            title="Fraction of positive class",
            range=[min(-0.05, y_min), max(1.05, y_max)],
            constrain="domain",
        ),
        title=dict(text=_common.make_title(name, title, sep="<br>"), x=0.5, xref="paper"),
        showlegend=len(ys) > 1
        or any(lbl is not None for lbl in legend)
        or any(lbl is not None for lbl in deviation_legend),
        hovermode="x unified" if len(ys) > 1 else "closest",
    )
    return fig


def beeswarm(
    values: "pandas.DataFrame",  # noqa F821 # type: ignore
    colors: Union["pandas.DataFrame", "pandas.Series", None] = None,  # noqa F821 # type: ignore
    color_name: Optional[str] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
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
    cmap : colormap | str, default="red_blue"
        Name of a color map.

    Returns
    -------
    Plotly figure object.

    References
    ----------
    .. [1] https://github.com/slundberg/shap/
    """

    try:
        import pandas as pd
    except ImportError:
        raise ValueError("")

    assert all(values[c].dtype.kind in "ifb" for c in values.columns)
    n_samples, n_features = values.shape
    row_height = 0.4

    colors_orig = colors
    if colors is not None:
        assert len(colors) == len(values)
        if hasattr(colors, "cat"):  # `colors` is a Series with category data type
            colors = colors.cat.codes

    cmap = get_colormap(cmap)
    fig = go.Figure()

    values_min = values.min().min()
    values_max = values.max().max()

    # beeswarm
    for pos, column in enumerate(reversed(values.columns)):
        fig.add_shape(
            type="line",
            line=dict(dash="dash", color="#cccccc", width=0.5),
            x0=values_min,
            y0=pos,
            x1=values_max,
            y1=pos,
        )
        if len(np.shape(colors)) == 1:  # `np.shape(None)` is `()`
            col = colors
            col_orig = colors_orig
        elif hasattr(colors, "columns") and column in colors.columns:
            col = colors[column]
            col_orig = colors_orig[column]
        else:
            col = None
            col_orig = None
        gray, colored = _common.beeswarm_feature(values[column], col, row_height)

        if gray is not None:
            x_gray, y_gray, i_gray = gray
            if colors is None:
                c_gray = (cmap(np.zeros((1,), dtype=np.float32))[0, :3] * 255).astype(np.uint8)
                c_gray = [f"rgb({c_gray[0]},{c_gray[1]},{c_gray[2]})"] * len(x_gray)
            else:
                c_gray = ["#777777"] * len(x_gray)
        if colored is not None:
            x_colored, y_colored = colored[:2]
            c_colored = (
                cmap((np.clip(colored[2], colored[3], colored[4]) - colored[3]) / (colored[4] - colored[3]))[:, :3]
                * 255
            ).astype(np.uint8)
            c_colored = [f"rgb({c[0]},{c[1]},{c[2]})" for c in c_colored]

        if gray is None:
            if colored is not None:
                x = x_colored
                y = y_colored
                c = c_colored
                i = colored[5]
        elif colored is None:
            x = x_gray
            y = y_gray
            c = c_gray
            i = i_gray
        else:
            x = pd.concat([x_gray, x_colored])
            y = np.concatenate([y_gray, y_colored])
            c = c_gray + c_colored
            i = np.concatenate([i_gray, colored[5]])

        if col_orig is None:
            text = ["ID={}<br>X={}".format(x.index[j], x.iloc[j]) for j in range(len(x))]
        else:
            text = [
                "ID={}<br>X={}<br>{}={}".format(x.index[j], x.iloc[j], color_name, col_orig.iloc[i[j]])
                for j in range(len(x))
            ]

        fig.add_trace(
            go.Scattergl(
                x=x,
                y=pos + y,
                mode="markers",
                marker=dict(size=4, color=c),
                hovertemplate="%{text}",
                text=text,
                name=column,
            )
        )

    fig.update_layout(
        xaxis=dict(title=x_label, constrain="domain"),
        yaxis=dict(
            ticktext=list(reversed(values.columns)),
            tickvals=list(range(n_features)),
            zeroline=False,
        ),
        title=dict(text=_common.make_title(None, title), x=0.5, xref="paper"),
        showlegend=False,
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(range=[-1, n_features], showgrid=False)

    return fig


def horizontal_bar(
    values: Union["pandas.Series", "pandas.DataFrame"],  # noqa F821 # type: ignore
    groups: Optional[dict] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
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
    cmap : colormap | str, default="red_blue"
        Name of a color map.

    Returns
    -------
    Plotly figure object.
    """

    n_features = len(values)
    if hasattr(values, "to_frame"):
        values = values.to_frame()
    if groups is None:
        groups = {c: [c] for c in values.columns}
    else:
        assert len(groups) >= 1
        assert all(all(c in values.columns for c in g) for g in groups.values())

    fig = go.Figure()

    cmap = get_colormap(cmap)
    neg_color = (np.asarray(cmap(0.0)[:3]) * 255).astype(np.uint8)
    pos_color = (np.asarray(cmap(1.0)[:3]) * 255).astype(np.uint8)
    pos_color = f"rgb({pos_color[0]},{pos_color[1]},{pos_color[2]})"
    neg_color = f"rgb({neg_color[0]},{neg_color[1]},{neg_color[2]})"

    # draw the bars
    for i, (g, columns) in enumerate(groups.items()):
        for c in columns:
            fig.add_trace(
                go.Bar(
                    name=g,
                    orientation="h",
                    x=values[c].iloc[::-1],
                    y=values.index[::-1],
                    offsetgroup=i,
                    marker=dict(
                        color=[neg_color if values[c].iloc[j] < 0 else pos_color for j in range(n_features - 1, -1, -1)]
                    ),
                )
            )

    fig.update_layout(
        xaxis=dict(title=x_label, constrain="domain"),
        title=dict(text=_common.make_title(None, title), x=0.5, xref="paper"),
        showlegend=False,
    )

    return fig
