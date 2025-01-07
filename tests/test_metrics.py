from functools import partial

import numpy as np
import pandas as pd
import pytest

from catabra_lib import metrics


def test_regression():
    rng = np.random.RandomState(1234)
    y_true = rng.randn(578) * 2.0 + np.pi
    noise = rng.randn(*y_true.shape)

    assert 0.8 < metrics.get("r2")(y_true, 1.2 * y_true + 0.1 * noise) < 0.9
    assert 0.15 < metrics.get("mean_absolute_error(mean:100)")(y_true, pd.Series(y_true + 0.2 * noise)) < 0.16
    ci1, ci2 = metrics.get("root_mean_squared_error(ci.95:100)")(
        pd.DataFrame(data=dict(A=y_true, B=np.sin(y_true))),
        pd.DataFrame(data=dict(A=noise, B=np.cos(y_true))),
        sample_weight=rng.uniform(0.5, 1.5, size=y_true.shape),
    )
    assert 2.0 < ci1 < ci2 < 3.0

    assert 1.0 < metrics.max_error(pd.Series(np.sin(y_true)), np.cos(y_true), sample_weight=np.ones_like(y_true)) < 2.0


def test_binary():
    rng = np.random.RandomState(1234)
    y_true = (rng.uniform(0, 1, size=697) < 0.15).astype(np.float32)
    noise = rng.randn(*y_true.shape)
    sample_weight = rng.uniform(0.5, 1.5, y_true.shape)

    # probabilities
    assert 0.8 < metrics.get("roc_auc")(y_true, y_true + 0.75 * noise) < 0.85
    assert 0.4 < metrics.get("average_precision")(y_true, y_true + 0.9 * noise) < 0.5
    assert 0.4 < metrics.pr_auc(y_true, y_true + 0.9 * noise) < 0.5
    assert metrics.maybe_thresholded(metrics.brier_loss)(y_true, 0.1 + 0.8 * y_true) < 1e-2
    assert (
        0.5 < metrics.maybe_thresholded(metrics.matthews_correlation_coefficient)(y_true, y_true + 0.55 * noise) < 0.6
    )

    # classes
    assert 0.7 < metrics.get("accuracy@0.5")(y_true, y_true + 0.8 * noise) < 0.8
    assert metrics.get("sensitivity@balance(std:50)")(y_true, y_true + 0.8 * noise, sample_weight=sample_weight) < 0.05
    assert 0.3 < metrics.get("recall@prevalence")(y_true, y_true + 1.5 * noise) < 0.4
    assert 0.57 < metrics.get("f1@zero_one")(y_true, y_true + 0.6 * noise) < 0.65
    assert (
        0.1 < metrics.get("markedness@argmax(min:10)")(y_true, y_true + 2.0 * noise, sample_weight=sample_weight) < 0.25
    )
    assert 0.45 < metrics.get("youden_index@argmax(max:10)")(y_true, y_true + 1.0 * noise) < 0.55
    assert 0.13 < metrics.get("jaccard(mean:10)")(y_true, rng.randint(0, 2, size=y_true.shape)) < 0.17

    assert not isinstance(metrics.maybe_thresholded(metrics.bootstrapped(metrics.roc_auc)), metrics.thresholded)
    assert isinstance(metrics.maybe_thresholded(metrics.bootstrapped(metrics.f1)), metrics.thresholded)

    try:
        metrics.accuracy(y_true, y_true * 0.5)
    except ValueError:
        pass
    else:
        assert False

    try:
        metrics.get("accuracy@xyz")
    except ValueError:
        pass
    else:
        assert False

    try:
        metrics.get("accuracy(mean:1000")
    except ValueError:
        pass
    else:
        assert False

    try:
        metrics.get("accuracy(mean)")
    except ValueError:
        pass
    else:
        assert False

    try:
        metrics.get("accuracy(mean:)")
    except ValueError:
        pass
    else:
        assert False

    try:
        metrics.get("accuracy(100)")
    except ValueError:
        pass
    else:
        assert False

    try:
        metrics.get("accuracy(mean:0.5)")
    except ValueError:
        pass
    else:
        assert False

    try:
        metrics.get("accuracy(mean:100)@0.5")
    except ValueError:
        pass
    else:
        assert False


def test_multiclass():
    rng = np.random.RandomState(1234)
    y_score = rng.randn(318, 5) + np.array([[-0.7, 0.7, -0.5, 0.0, 0.5]])
    y_true = np.argmax(y_score, axis=1)
    noise = rng.randn(*y_score.shape)
    sample_weight = rng.uniform(0.5, 1.5, size=y_true.shape)

    def sm(x: np.ndarray, t: float = 1.0) -> np.ndarray:
        e = np.exp(t * x)
        return e / e.sum(axis=1, keepdims=True)

    # probabilities
    assert 0.75 < metrics.get("roc_auc_ovo")(y_true, sm(y_score + 1.5 * noise)) < 0.8
    assert 0.7 < metrics.get("roc_auc_ovr_weighted")(y_true, sm(y_score + 2 * noise)) < 0.75
    assert 2.5 < metrics.log_loss(y_true, sm(y_score + 3 * noise)) < 2.7
    assert 1.2 < metrics.hinge_loss(y_true, sm(y_score + 3 * noise), sample_weight=sample_weight) < 1.3

    # classes
    assert (
        0.38 < metrics.get("balanced_accuracy@0.5")(y_true, sm(y_score + 2 * noise), sample_weight=sample_weight) < 0.41
    )
    ci1, ci2 = metrics.get("specificity_macro@0.5(ci.95:100)")(y_true, sm(y_score + 2 * noise))
    assert 0.8 < ci1 < ci2 < 0.9
    assert 0.75 < metrics.get("negative_predictive_value_weighted@0.5")(y_true, sm(y_score + 2 * noise)) < 0.8
    ci1, ci2 = metrics.get("cohen_kappa_micro@0.5(iqr:10)")(y_true, sm(y_score + 5 * noise))
    assert 0.0 < ci1 < ci2 < 0.11
    assert 0.2 < metrics.get("hamming_loss@0.5")(y_true, sm(y_score + 0.5 * noise)) < 0.3


def test_multilabel():
    rng = np.random.RandomState(1234)
    y_true = (rng.uniform(0, 1, size=(408, 3)) < np.array([[0.1, 0.3, 0.5]])).astype(np.float32)
    noise = rng.randn(*y_true.shape)

    # probabilities
    assert 0.8 < metrics.get("roc_auc")(y_true, y_true + np.array([[0.75, 1.0, 0.6]]) * noise) < 0.85
    assert 0.8 < metrics.get("roc_auc_weighted")(y_true, y_true + np.array([[0.75, 1.0, 0.6]]) * noise) < 0.85
    assert 0.6 < metrics.average_precision_micro(y_true, y_true + np.array([[0.75, 1.0, 0.6]]) * noise) < 0.7

    # classes
    assert 0.7 < metrics.get("accuracy_micro@0.5")(y_true, y_true + np.array([[0.75, 1.0, 0.6]]) * noise) < 0.8
    assert 0.4 < metrics.get("accuracy@0.5(mean:10)")(y_true, y_true + np.array([[0.75, 1.0, 0.6]]) * noise) < 0.5
    assert 0.7 < metrics.get("accuracy_weighted@0.5")(y_true, y_true + np.array([[0.75, 1.0, 0.6]]) * noise) < 0.8

    try:
        # thresholding strategies do not work with multilabel problems
        metrics.get("f1_macro@argmax")(y_true, y_true + np.array([[0.75, 1.0, 0.6]]) * noise)
    except ValueError:
        pass
    else:
        assert False


def test_classification(n_reps: int = 10, seed: int = 30047):
    rng = np.random.RandomState(seed)

    def _fbeta_like(name: str, m: str) -> list:
        return [
            (metrics.get(name), m, "b"),
            (metrics.no_average(metrics.get(name)), partial(m, average=None), "bcl"),
            (metrics.get(name + "_micro"), partial(m, average="micro"), "bcl"),
            (metrics.get(name + "_macro"), partial(m, average="macro"), "bcl"),
            (metrics.get(name + "_weighted"), partial(m, average="weighted"), "bcl"),
            (metrics.get(name + "_samples"), partial(m, average="samples"), "l"),
        ]

    fbeta_like = [
        (m1, m2, typ)
        for _name, _m in [
            ("f1", metrics.skl_metrics.f1_score),
            ("sensitivity", partial(metrics.skl_metrics.recall_score, zero_division=0)),
            ("positive_predictive_value", partial(metrics.skl_metrics.precision_score, zero_division=1)),
            ("jaccard", partial(metrics.skl_metrics.jaccard_score, zero_division=1)),
        ]
        for m1, m2, typ in _fbeta_like(_name, _m)
    ]

    # "b": binary, "c": multiclass, "l": multilabel
    for m1, m2, typ in [
        (metrics.accuracy, metrics.skl_metrics.accuracy_score, "bcl"),
        (metrics.hamming_loss, metrics.skl_metrics.hamming_loss, "bcl"),
        (metrics.balanced_accuracy, metrics.skl_metrics.balanced_accuracy_score, "bc"),
        (metrics.cohen_kappa, metrics.skl_metrics.cohen_kappa_score, "bc"),
        (metrics.matthews_correlation_coefficient, metrics.skl_metrics.matthews_corrcoef, "bc"),
    ] + fbeta_like:
        for _ in range(n_reps):
            for t in typ:
                n = rng.randint(100, 1000)
                if t == "b":
                    y_true = rng.uniform(0, 1, size=n) < 0.25
                    y_pred = y_true.astype(np.float32) + rng.uniform(0.5, 1.5) * rng.randn(*y_true.shape) > 0.5
                elif t == "c":
                    y_score = rng.randn(n, 4) + np.array([[0.7, -0.5, 0.0, 0.5]])
                    y_true = np.argmax(y_score, axis=1)
                    y_pred = np.argmax(y_score + rng.uniform(0.5, 3.0, size=(1, 4)) * rng.randn(*y_score.shape), axis=1)
                elif t == "l":
                    y_true = rng.uniform(0, 1, size=(n, 11)) < np.linspace(0.1, 0.5, 11)[np.newaxis]
                    y_pred = (
                        y_true.astype(np.float32) + rng.uniform(0.5, 1.5, size=(1, 11)) * rng.randn(*y_true.shape) > 0.5
                    )

                assert np.all(np.isclose(m1(y_true, y_pred), m2(y_true, y_pred)))


def test_to_score():
    assert metrics.to_score(metrics.r2) is metrics.r2
    assert metrics.to_score(metrics.average_precision) is metrics.average_precision
    assert metrics.to_score(metrics.f1_samples) is metrics.f1_samples
    assert metrics.to_score(metrics.informedness) is metrics.informedness

    m = metrics.get("jaccard@prevalence(mean:50)")
    assert metrics.to_score(m) is m

    assert metrics.to_score(metrics.log_loss_weighted) is not metrics.log_loss_weighted
    assert metrics.to_score(metrics.mean_absolute_percentage_error) is not metrics.mean_absolute_percentage_error

    m = metrics.get("hamming_loss_samples@0.5(mean:50)")
    assert metrics.to_score(m) is not m


def test_calibration_curve():
    rng = np.random.RandomState(9876)
    y_true = (rng.uniform(0, 1, size=697) < 0.3).astype(np.float32)
    noise = rng.randn(*y_true.shape)

    fractions, thresholds = metrics.calibration_curve(y_true, y_true + 0.8 * noise - 4)
    assert len(fractions) + 1 == len(thresholds)

    fractions, thresholds = metrics.calibration_curve(y_true, y_true + 0.8 * noise, thresholds=np.linspace(-1, 2, 50))
    assert len(fractions) == 49
    assert len(thresholds) == 50


def test_roc_pr_curve():
    rng = np.random.RandomState(53719)
    y_true = (rng.uniform(0, 1, size=697) < 0.5).astype(np.float32)
    y_score = y_true + 0.9 * rng.randn(*y_true.shape)
    sample_weight = rng.uniform(0.5, 1.5, y_true.shape)

    fpr, tpr, thresholds_roc, _, recall, thresholds_pr = metrics.roc_pr_curve(
        y_true, y_score, sample_weight=sample_weight
    )
    assert (fpr[1:] >= fpr[:-1]).all()
    assert (tpr[1:] >= tpr[:-1]).all()
    assert (thresholds_roc[1:] < thresholds_roc[:-1]).all()
    assert (recall[1:] <= recall[:-1]).all()
    assert (thresholds_pr[1:] > thresholds_pr[:-1]).all()


@pytest.mark.parametrize(
    argnames=["seed"],
    argvalues=[
        (28640,),
        (9999,),
        (1,),
        (42,),
        (34805,),
        (40633,),
    ],
)
def test_confusion_matrix(seed: int):
    rng = np.random.RandomState(seed)

    y_true_ml = rng.uniform(0, 1, size=(rng.randint(100, 1000), 3)) < np.array([[0.5, 0.4, 0.01]])
    y_pred_ml = y_true_ml.astype(np.float32) + np.array([[0.6, 0.8, 1.0]]) * rng.randn(*y_true_ml.shape) > 0.5

    y_score = rng.randn(len(y_true_ml), 4) + np.array([[0.7, -0.5, 0.0, 0.5]])
    y_true_mc = np.argmax(y_score, axis=1)
    y_pred_mc = np.argmax(y_score + rng.uniform(0.5, 3.0, size=(1, 4)) * rng.randn(*y_score.shape), axis=1)

    if rng.randint(2) == 0:
        sw = None
    else:
        sw = rng.uniform(0.5, 5, size=len(y_true_ml))

    cm_ml = metrics.confusion_matrix(y_true_ml, y_pred_ml, sample_weight=sw)  # (3, 2, 2)
    cm_mc = metrics.confusion_matrix(y_true_mc, y_pred_mc, sample_weight=sw)  # (4, 4)

    def _check_equality(a, b):
        if isinstance(a, tuple):
            assert isinstance(b, tuple)
            assert len(a) == len(b)
            assert np.isclose(a[:-1], b[:-1]).all()
            assert (a[-1] is None and b[-1] is None) or np.all(a[-1] == b[-1]) or np.all(np.isclose(a[-1], b[-1]))
        else:
            assert not isinstance(b, tuple)
            assert np.all(np.isclose(a, b))

    for func, func_gt in [
        (metrics.precision_recall_fscore_support_cm, metrics.precision_recall_fscore_support),
        (metrics.positive_predictive_value_cm, metrics.positive_predictive_value),
        (metrics.sensitivity_cm, metrics.sensitivity),
        (metrics.accuracy_cm, metrics.accuracy),
        (metrics.balanced_accuracy_cm, metrics.balanced_accuracy),
        (metrics.f1_cm, metrics.f1),
        (metrics.cohen_kappa_cm, metrics.cohen_kappa),
        (partial(metrics.cohen_kappa_cm, weights="linear"), partial(metrics.cohen_kappa, weights="linear")),
        (partial(metrics.cohen_kappa_cm, weights="quadratic"), partial(metrics.cohen_kappa, weights="quadratic")),
        (metrics.matthews_correlation_coefficient_cm, metrics.matthews_correlation_coefficient),
        (metrics.jaccard_cm, metrics.jaccard),
        (metrics.hamming_loss_cm, metrics.hamming_loss),
        (metrics.specificity_cm, metrics.specificity),
        (metrics.negative_predictive_value_cm, metrics.negative_predictive_value),
        (metrics.informedness_cm, metrics.informedness),
        (metrics.markedness_cm, metrics.markedness),
    ]:
        for cm, y_true, y_pred in [
            (cm_ml[1], y_true_ml[:, 1], y_pred_ml[:, 1]),
            (cm_ml, y_true_ml, y_pred_ml),
            (cm_mc, y_true_mc, y_pred_mc),
        ]:
            try:
                func(cm=cm, average="samples")
            except ValueError:
                pass
            else:
                assert False

            try:
                res = func(cm=cm)
            except Exception as ex:  # noqa E722
                if isinstance(ex, ValueError) and cm.ndim == 3 and '"global"' in ex.args[0]:
                    # "global" is not suported in multilabel CM-based metrics, but in their non-CM-based counterparts
                    pass
                else:
                    try:
                        func_gt(y_true, y_pred, sample_weight=sw)
                    except:  # noqa E722
                        pass
                    else:
                        assert False
            else:
                _check_equality(res, func_gt(y_true, y_pred, sample_weight=sw))

            for average in (None, "global", "binary", "micro", "macro", "weighted"):
                try:
                    res = func(cm=cm, average=average)
                except:  # noqa E722
                    try:
                        func_gt(y_true, y_pred, average=average, sample_weight=sw)
                    except:  # noqa E722
                        pass
                    else:
                        assert average == "global"
                        assert cm.ndim == 3
                else:
                    _check_equality(res, func_gt(y_true, y_pred, average=average, sample_weight=sw))
