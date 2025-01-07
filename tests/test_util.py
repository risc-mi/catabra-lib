import numpy as np
import pandas as pd

from catabra_lib.util import to_num_array, to_num_arrays


def test_to_num_array():
    # scalar
    assert to_num_array(-8.7) == -8.7
    assert to_num_array(np.pi) == np.pi
    x = np.zeros((), dtype=bool)
    assert to_num_array(x) is x
    try:
        to_num_array(0, rank=(1, 2))
    except ValueError:
        pass
    else:
        assert False

    # array
    x = np.empty((128, 7), dtype=np.float32)
    assert to_num_array(x, rank=(1, 2)) is x

    # Series
    s = pd.to_timedelta(pd.Series(np.arange(10)), unit="s")
    assert to_num_array(s, dtype=None) is s.values
    try:
        to_num_array(s)
    except ValueError:
        pass
    else:
        assert False

    # DataFrame
    df = pd.DataFrame(data=dict(A=[2, 3, 4], B=[0, 0, 0]))
    assert (to_num_array(df, rank=(2,)) == df.values).all()
    try:
        to_num_array(df)
    except ValueError:
        pass
    else:
        assert False


def test_to_num_arrays():
    x = np.empty((128, 7), dtype=np.float32)
    y = np.empty_like(x)
    assert all(a is b for a, b in zip((x, y), to_num_arrays(x, y, rank=(2,))))

    # different data types don't matter
    y = np.empty(x.shape, dtype=np.uint8)
    assert all(a is b for a, b in zip((x, y), to_num_arrays(x, y, rank=(2,))))

    try:
        to_num_arrays(x, np.empty((128, 6)))
    except ValueError:
        pass
    else:
        assert False
