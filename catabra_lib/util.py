from typing import Optional, Union

import numpy as np


def to_num_array(
    x, arg_name: Optional[str] = None, rank=(0, 1), dtype: str = "uifb"
) -> Union[int, float, bool, np.ndarray]:
    """Convert a given array-like into a scalar or Numpy array, unless it is one already. Ensure that it has a given
    rank and data type.

    Parameters
    ----------
    x : array-like
        The array-like to convert. Currently supported are scalars (float, int, bool), Numpy arrays, pandas Series and
        DataFrames, and all other objects that implement a `to_numpy()` method or that can be converted using
        `numpy.asarray()`.
    arg_name : str, optional
        Name of `x`, as it is displayed in exceptions and warnings.
    rank : tuple of int, default=(0, 1)
        Allowed ranks (i.e., number of dimensions) of `x`. 0 corresponds to scalars, 1 to vectors, 2 to matrices.
    dtype : str, optional
        Allowed data types of `x`, string of single-character data type kinds like "i", "f", "b". If None, no data type
        check happens. Note that `x` is not cast to the specified type; instead, an exception is raised if it does not
        have a suitable type.

    Returns
    -------
    scalar | ndarray
        Scalar or Numpy array corresponding to `x`. Could be `x` itself, or one of its class members.

    See Also
    --------
    to_num_arrays
    """
    if isinstance(x, (int, float, bool)):
        if 0 not in rank:
            raise ValueError(
                "Expected `{}` to be a {}D array, but got scalar {}".format(
                    "" if arg_name is None else arg_name,
                    rank[0] if len(rank) == 1 else tuple(rank),
                    x,
                )
            )
        return x
    else:
        if hasattr(x, "to_numpy"):
            # works for Timestamp, Timedelta, DataFrame, Series (in which case it is identical to `Series.values`)
            x = x.to_numpy()
        else:
            # works for scalar, DataFrame, Series (in which case it is identical to `Series.values`)
            x = np.asarray(x)
        if x.ndim not in rank:
            raise ValueError(
                "Expected `{}` to be a {}D array, but got {}D array of shape {}".format(
                    "" if arg_name is None else arg_name,
                    rank[0] if len(rank) == 1 else tuple(rank),
                    x.ndim,
                    x.shape,
                )
            )
        elif dtype is not None and x.dtype.kind not in dtype:
            raise ValueError(
                "Expected `{}` to have numeric data type, but got {}".format(
                    "" if arg_name is None else arg_name, x.dtype.name
                )
            )
        return x


def to_num_arrays(*args, arg_names=None, check_equal_shape: bool = True, **kwargs) -> tuple:
    """Convert one or more array-likes into scalars or Numpy arrays, by applying `to_num_array()`.
    Additionally, check that all arrays have the same shape.

    Parameters
    ----------
    *args : array-like
        Array-likes to convert.
    arg_names : iterable of str, optional
        Names of the arrays, as they are displayed in exceptions and warnings.
        None defaults to `["#0", "#1", ...]`. If not None, must have the same length as `args`.
    check_equal_shape : bool, default=True
        Check that all arrays have the same shape, raise a ValueError if not.
    **kwargs
        Additional keyword arguments passed to function `to_num_array`, notably `rank` and `dtype`.

    Returns
    -------
    tuple of ndarray
        Tuple of scalars or ndarrays, with one element for each element in `args`.

    See Also
    --------
    to_num_array
    """
    if arg_names is None:
        arg_names = ["#" + str(i) for i in range(len(args))]
    out = tuple(to_num_array(a, arg_name=n, **kwargs) for a, n in zip(args, arg_names))
    if check_equal_shape and len(out) > 1:
        for a, n in zip(out, arg_names):
            if np.shape(a) != np.shape(out[0]):
                raise ValueError(
                    "`{}` and `{}` have different shapes: {} vs {}".format(
                        arg_names[0], n, np.shape(out[0]), np.shape(a)
                    )
                )
    return out
