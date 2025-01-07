#  Copyright (c) 2025. RISC Software GmbH.
#  All rights reserved.

from typing import Optional, Tuple

import numpy as np

from ..util import to_num_array, to_num_arrays

_cached_colormaps = {}


def make_title(name: Optional[str], title: Optional[str], sep: str = "\n") -> Optional[str]:
    if title is None:
        return name
    elif name is None:
        return title
    else:
        return title + sep + name


def load_saved_colormap(name: str):
    data = _cached_colormaps.get(name)

    if data is None:
        import pickle
        from pathlib import Path

        with open(Path(__file__).parent / (name + ".pkl"), mode="rb") as fh:
            data = pickle.load(fh)

        _cached_colormaps[name] = data

    return data


def convert_timedelta(x: np.ndarray) -> Tuple[np.timedelta64, str]:
    mean = np.abs(x).mean()
    unit = np.timedelta64(int(365.2525 * 24 * 3600 * 1000000000), "ns")
    if mean > unit:
        uom = "y"
    else:
        unit = np.timedelta64(24 * 3600 * 1000000000, "ns")
        if mean > unit:
            uom = "d"
        else:
            unit = np.timedelta64(3600 * 1000000000, "ns")
            if mean > unit:
                uom = "h"
            else:
                unit = np.timedelta64(60 * 1000000000, "ns")
                if mean > unit:
                    uom = "m"
                else:
                    unit = np.timedelta64(1000000000, "ns")
                    uom = "s"
    return unit, uom


def validate_confusion_matrix(cm, class_names: Optional[list]) -> tuple[np.ndarray, list]:
    # returns array of shape `(n + 1, n + 1)` and list of length `n`
    # last row/column are column/row totals

    out = to_num_array(cm, arg_name="cm", rank=(2,))

    if abs(out.shape[0] - out.shape[1]) > 1:
        raise ValueError("Confusion matrix must be square, but got shape {}".format(out.shape))
    elif out.shape[0] < 1 or out.shape[1] < 1:
        raise ValueError("Confusion matrix must be contain at least one class, but got shape {}".format(out.shape))

    if hasattr(cm, "index") and hasattr(cm, "columns"):
        if out.shape[0] < out.shape[1]:
            class_names = list(cm.columns)
        else:
            class_names = list(cm.index)
    elif class_names is None:
        raise ValueError("`class_names` must be provided when the confusion matrix is no pandas DataFrame.")

    if len(class_names) == min(out.shape):
        if out.shape[0] == out.shape[1]:
            if (
                class_names[-1] == "__total__"
                and out.shape[0] > 1
                and (out[-1] == out[:-1].sum(axis=0)).all()
                and (out[:, -1] == out[:, :-1].sum(axis=1)).all()
            ):
                class_names = class_names[:-1]
            else:
                # add row- and column totals
                tmp = np.zeros((out.shape[0] + 1, out.shape[1] + 1), dtype=out.dtype)
                tmp[:-1, :-1] = out
                tmp[-1] = tmp.sum(axis=0)
                tmp[:, -1] = tmp.sum(axis=1)
                out = tmp

        if class_names[-1] == "__total__":
            print(
                'Last class is assumed to be a proper class, although it is called "__total__".'
                " Change name of last class to silence this message."
            )
    elif len(class_names) == max(out.shape):
        if class_names[-1] != "__total__":
            print(
                "Last {} of confusion matrix is assumed to contain {} totals."
                ' Set last class name to "__total__" to silence this message.'.format(
                    "row" if out.shape[0] > out.shape[1] else "column",
                    "column" if out.shape[0] > out.shape[1] else "row",
                )
            )
        class_names = class_names[:-1]
    elif out.shape[0] == out.shape[1] and len(class_names) + 1 == out.shape[0]:
        if not (out[:-1].sum(axis=0) == out[-1]).all():
            raise ValueError(
                "Last row of confusion matrix is expected to contain column totals, but values do not match."
            )
        elif not (out[:, :-1].sum(axis=1) == out[:, -1]).all():
            raise ValueError(
                "Last column of confusion matrix is expected to contain row totals, but values do not match."
            )
        elif class_names[-1] == "__total__":
            print(
                'Last class is assumed to be a proper class, although it is called "__total__".'
                " Change name of last class to silence this message."
            )
    else:
        raise ValueError(
            "Length of `class_names` ({}) does not match dimensions of confusion matrix ({})".format(
                len(class_names), out.shape
            )
        )

    if out.shape[0] < out.shape[1]:
        if not (out[:, -1] == out[:, :-1].sum(axis=1)).all():
            raise ValueError(
                "Last column of confusion matrix is expected to contain row totals, but values do not match."
            )

        # add column totals
        tmp = np.zeros((out.shape[0] + 1, out.shape[1]), dtype=out.dtype)
        tmp[:-1] = out
        tmp[-1] = tmp.sum(axis=0)
        out = tmp
    elif out.shape[0] > out.shape[1]:
        if not (out[-1] == out[:-1].sum(axis=0)).all():
            raise ValueError(
                "Last row of confusion matrix is expected to contain column totals, but values do not match."
            )

        # add column totals
        tmp = np.zeros((out.shape[0], out.shape[1] + 1), dtype=out.dtype)
        tmp[:, :-1] = out
        tmp[:, -1] = tmp.sum(axis=1)
        out = tmp

    return out, class_names


def validate_roc_pr(xs, ys, deviation, sample_weight, from_predictions: bool, roc: bool) -> tuple[list, list, list]:
    if not isinstance(xs, list):
        xs = [xs]
    if not isinstance(ys, list):
        ys = [ys]

    if len(xs) != len(ys):
        raise ValueError("Length of `xs` ({}) differs from length of `ys` ({})".format(len(xs), len(ys)))

    if xs:
        xs_and_ys = [to_num_arrays(x, y, rank=(1,)) for x, y in zip(xs, ys)]
        xs = [x for x, _ in xs_and_ys]
        ys = [y for _, y in xs_and_ys]

        if from_predictions:
            if deviation is not None:
                raise ValueError("`deviation` must be None if `from_predictions` is True.")

            if roc:
                from sklearn.metrics import roc_curve as curve
            else:
                from sklearn.metrics import precision_recall_curve as curve

            xs_and_ys = [curve(x, y, sample_weight=sample_weight)[:2] for x, y in zip(xs, ys)]
            xs = [x for x, _ in xs_and_ys]
            ys = [y for _, y in xs_and_ys]

            if not roc:
                # swap precision and recall
                xs, ys = ys, xs
        elif sample_weight is not None:
            raise ValueError("`sample_weight` must be None if `from_predictions` is False.")

    if deviation is None:
        deviation = [None] * len(ys)
    else:
        if not isinstance(deviation, list):
            deviation = [deviation]

        if len(deviation) != len(ys):
            raise ValueError(
                "Length of `deviation` ({}) differs from length of `ys` ({})".format(len(deviation), len(ys))
            )

        deviation = [dv if dv is None else to_num_array(dv, rank=(2,)) for dv in deviation]

    return xs, ys, deviation


def beeswarm_feature(values, colors, row_height: float):
    n_bins = 100
    values = to_num_array(values, "values", rank=(1,))
    n_samples = len(values)
    inds = np.random.permutation(n_samples)
    values = values[inds]
    if colors is None:
        pass
    elif colors.dtype.name == "category":
        colors = None
    else:
        colors = to_num_array(colors, "colors", rank=(1,), dtype=None)

        if len(colors) != n_samples:
            raise ValueError("`values` and `colors` have different length: {} vs {}".format(len(colors), n_samples))
        try:
            colors = np.asarray(colors, dtype=np.float64)  # make sure this can be numeric
        except:  # noqa
            colors = None
        else:
            colors = colors[inds]

    quant = np.round(n_bins * (values - values.min()) / (values.max() - values.min() + 1e-8))
    layer = 0
    last_bin = -1
    ys = np.zeros(n_samples)
    for ind in np.argsort(quant + np.random.randn(n_samples) * 1e-6):
        if quant[ind] != last_bin:
            layer = 0
        ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quant[ind]
    ys *= 0.9 * (row_height / np.max(ys + 1))

    if colors is None:
        return (values, ys, inds), None
    else:
        # trim the color range, but prevent the color range from collapsing
        vmin = np.nanpercentile(colors, 5)
        vmax = np.nanpercentile(colors, 95)
        if vmin == vmax:
            vmin = np.nanpercentile(colors, 1)
            vmax = np.nanpercentile(colors, 99)
            if vmin == vmax:
                vmin = np.min(colors)
                vmax = np.max(colors)
        if vmin > vmax:  # fixes rare numerical precision issues
            vmin = vmax

        nan_mask = np.isnan(colors)
        not_nan_mask = ~nan_mask
        cvals = colors[not_nan_mask].copy()
        cvals[cvals > vmax] = vmax
        cvals[cvals < vmin] = vmin

        return ((values[nan_mask], ys[nan_mask], inds[nan_mask]) if nan_mask.any() else None), (
            values[not_nan_mask],
            ys[not_nan_mask],
            cvals,
            vmin,
            vmax,
            inds[not_nan_mask],
        )
