import numpy as np
import pandas as pd

from catabra_lib.plotting._common import (
    convert_timedelta,
    validate_confusion_matrix,
    validate_roc_pr,
)


def test_convert_timedelta():
    assert convert_timedelta(np.timedelta64(98, "m")) == (
        np.timedelta64(3600 * 1000000000, "ns"),
        "h",
    )

    assert convert_timedelta(-np.timedelta64(27, "s")) == (
        np.timedelta64(1000000000, "ns"),
        "s",
    )

    assert convert_timedelta(pd.to_timedelta(pd.Series([-45, 28, -3]), unit="h")) == (
        np.timedelta64(24 * 3600 * 1000000000, "ns"),
        "d",
    )


def test_validate_confusion_matrix():
    class_names = ["A", "B"]
    cm = np.array([[25, 1, 26], [3, 18, 21], [28, 19, 47]])

    # array
    cm_out, classes = validate_confusion_matrix(cm, class_names)
    assert (cm_out == cm).all()
    assert classes == class_names

    cm_out, classes = validate_confusion_matrix(cm[:-1], class_names + ["__total__"])
    assert (cm_out == cm).all()
    assert classes == class_names

    cm_out, classes = validate_confusion_matrix(cm[:, :-1], class_names)
    assert (cm_out == cm).all()
    assert classes == class_names

    cm_out, classes = validate_confusion_matrix(cm[:-1, :-1], class_names)
    assert (cm_out == cm).all()
    assert classes == class_names

    cm_out, classes = validate_confusion_matrix(cm[:-1, :-1], ["A", "__total__"])
    assert (cm_out == cm).all()
    assert classes == ["A", "__total__"]

    cm_out, classes = validate_confusion_matrix(cm, class_names + ["C"])
    assert (cm_out[:-1, :-1] == cm).all()
    assert classes == class_names + ["C"]

    cm_out, classes = validate_confusion_matrix(cm[:, :-1], class_names + ["C"])
    assert (cm_out == cm).all()
    assert classes == class_names

    # DataFrame
    cm_out, classes = validate_confusion_matrix(
        pd.DataFrame(
            data=cm,
            columns=class_names + ["__total__"],
            index=class_names + ["__total__"],
        ),
        None,
    )
    assert (cm_out == cm).all()
    assert classes == class_names

    # no matrix
    try:
        validate_confusion_matrix(cm[0], class_names)
    except ValueError:
        pass
    else:
        assert False

    # not square
    try:
        validate_confusion_matrix(cm[:1], class_names)
    except ValueError:
        pass
    else:
        assert False

    # empty
    try:
        validate_confusion_matrix(cm[:1, :0], class_names[:1])
    except ValueError:
        pass
    else:
        assert False

    # no class_names
    try:
        validate_confusion_matrix(cm, None)
    except ValueError:
        pass
    else:
        assert False

    # invalid row/column totals
    try:
        validate_confusion_matrix(cm[:-1, :-1], class_names[:1])
    except ValueError:
        pass
    else:
        assert False

    # invalid length of class_names
    try:
        validate_confusion_matrix(cm, class_names[:1])
    except ValueError:
        pass
    else:
        assert False

    # invalid row totals
    try:
        validate_confusion_matrix(cm[:-1, :1], class_names[:1])
    except ValueError:
        pass
    else:
        assert False

    # invalid column totals
    try:
        validate_confusion_matrix(cm[:1, :-1], class_names[:1])
    except ValueError:
        pass
    else:
        assert False


def test_validate_roc_pr():
    assert validate_roc_pr([], [], None, None, False, True) == ([], [], [])

    x = np.linspace(0, 1, 100, dtype=np.float32)
    y = np.empty(100, dtype=np.float32)
    assert all(a is b[0] for a, b in zip((x, y, None), validate_roc_pr([x], y, None, None, False, True)))

    validate_roc_pr(
        np.random.randint(0, 2, size=100),
        np.random.uniform(-1, 1, size=100),
        None,
        np.random.uniform(0, 1, 100),
        True,
        False,
    )

    x = np.empty(10)
    dv = np.empty((2, 10))
    try:
        validate_roc_pr([], x, None, None, False, True)
    except ValueError:
        pass
    else:
        assert False

    try:
        validate_roc_pr(x, x, None, x, False, True)
    except ValueError:
        pass
    else:
        assert False

    try:
        validate_roc_pr(x, x, dv, None, True, True)
    except ValueError:
        pass
    else:
        assert False

    try:
        validate_roc_pr(x, x, [dv, dv], None, False, False)
    except ValueError:
        pass
    else:
        assert False

    try:
        validate_roc_pr(x, x, [dv[0]], None, False, False)
    except ValueError:
        pass
    else:
        assert False
