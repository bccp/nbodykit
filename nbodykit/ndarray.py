"""
Helper routines for ndarray objects

These are really handy things if numpy had them.

"""
import numpy

def extend_dtype(data, extra_dtypes):
    """
    Extend the data type of a structured array
    
    Parameters
    ----------
    data : array_like
        a structured array 
    extra_dtypes : list of (str, dtyple)
        a list of data types, specified by the their name and data type
    
    Returns
    -------
    new : array_like
        a copy of `data`, with the extra data type fields, initialized 
        to zero
    """
    # copy the data
    dtype = list(data.dtype.descr) # make copy
    existing_names = list(data.dtype.names) # make copy
    
    # make sure we aren't overwriting any named field
    new_names = [dt[0] for dt in extra_dtypes]
    if any(name in existing_names for name in new_names):
        raise ValueError("adding a new data type with name already present in structured array")
        
    dtype += extra_dtypes

    # work around numpy dtype reference counting bugs.
    dtype = numpy.dtype(dtype)

    # add old variables
    new = numpy.zeros(data.shape, dtype=dtype)
    for col in existing_names:
        new[col][:] = data[col]
        
    return new

def equiv_class(labels, values, op, dense_labels=False, identity=None, minlength=None):
    """
    apply operation to equivalent classes by label, on values

    Parameters 
    ----------
    labels : array_like
        the label of objects, starting from 0.
    values : array_like
        the values of objects (len(labels), ...)
    op : :py:class:`numpy.ufunc`
        the operation to apply
    dense_labels : boolean
        If the labels are already dense (from 0 to Nobjects - 1)
        If False, :py:meth:`numpy.unique` is used to convert
        the labels internally

    Returns
    -------
    result : 
        the value of each equivalent class

    Examples
    --------
    >>> x = numpy.arange(10)
    >>> print equiv_class(x, x, numpy.fmin, dense_labels=True)
    [0 1 2 3 4 5 6 7 8 9]

    >>> x = numpy.arange(10)
    >>> v = numpy.arange(20).reshape(10, 2)
    >>> x[1] = 0
    >>> print equiv_class(x, 1.0 * v, numpy.fmin, dense_labels=True, identity=numpy.inf)
    [[  0.   1.]
     [ inf  inf]
     [  4.   5.]
     [  6.   7.]
     [  8.   9.]
     [ 10.  11.]
     [ 12.  13.]
     [ 14.  15.]
     [ 16.  17.]
     [ 18.  19.]]

    """
    # dense labels
    if not dense_labels:
        junk, labels = numpy.unique(labels, return_inverse=True)
        del junk
    N = numpy.bincount(labels)
    offsets = numpy.concatenate([[0], N.cumsum()], axis=0)[:-1]
    arg = labels.argsort()
    if identity is None: identity = op.identity
    if minlength is None:
        minlength = len(N)

    # work around numpy dtype reference counting bugs
    # be a simple man and never steal anything.

    dtype = numpy.dtype((values.dtype, values.shape[1:]))

    result = numpy.empty(minlength, dtype=dtype)
    result[:len(N)] = op.reduceat(values[arg], offsets)

    if (N == 0).any():
        result[:len(N)][N == 0] = identity

    if len(N) < minlength:
        result[len(N):] = identity

    return result

def replacesorted(arr, sorted, b, out=None):
    """
    replace a with corresponding b in arr

    Parameters
    ----------
    arr : array_like
        input array
    sorted   : array_like 
        sorted

    b   : array_like

    out : array_like,
        output array
    Result
    ------
    newarr  : array_like
        arr with a replaced by corresponding b

    Examples
    --------
    >>> print replacesorted(numpy.arange(10), numpy.arange(5), numpy.ones(5))
    [1 1 1 1 1 5 6 7 8 9]

    """
    if out is None:
        out = arr.copy()
    if len(sorted) == 0:
        return out
    ind = sorted.searchsorted(arr)
    ind.clip(0, len(sorted) - 1, out=ind)
    arr = numpy.array(arr)
    found = sorted[ind] == arr
    out[found] = b[ind[found]]
    return out

