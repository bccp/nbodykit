from .. import os, cache_dir
from . import results

def assert_numpy_allclose(this, ref, **kws):
    """
    Wrapper around ``numpy.testing.assert_allclose`` that will loop
    over each element of a complex data type, i.e., structured array
    """
    from numpy.testing import assert_allclose

    # normal numpy array
    if ref.dtype.names is None:
        assert_allclose(this, ref, **kws)
    # structured array
    else:
        for name in ref.dtype.names:
            if name not in this.dtype.names:
                raise AssertionError("dtype mismatch in structured data")
            assert_allclose(this[name], ref[name], **kws)

def _make_exc(self, string):
    with open("run.stderr", 'r') as stderr:
        lines = stderr.read()
    return AssertionError("%s \nCmdline\n %s\nstderr:\n%s\n" % (string, self.run_fixture.cmd, lines))

def test_exit_code(self):
    """
    Assert that the exit code is equal to 0
    """
    if not self.run_fixture.exit_code == 0:
        raise _make_exc(self, "Exit code nonzero.")

def test_exception(self):
    """
    Assert that no exception occurred, by analyzing the 
    ``stderr`` output
    """
    with open("run.stderr", 'r') as stderr:
        lines = stderr.read()
        if "Error" in lines or "Traceback" in lines:
            raise _make_exc(self, "An exception is raised.")

def test_result_md5sum(self):
    """
    Assert that the result file from the pipeline run has the same 
    MD5 sum as the reference result file in the cache directory
    """
    try:
        this, ref = results.get_result_paths(self.output_file)
        assert results.compute_md5sum(this) == results.compute_md5sum(ref)
    except Exception as e:
        raise _make_exc(self, str(e))

def test_pandas_hdf_result(self, dataset, **kws):
    """
    Test that the pandas HDF5 result file is the same as the reference
    result file, by checking that the ``DataFrame`` objects are 
    the same
    
    Parameters
    ----------
    dataset : str
        the name of the HDF5 dataset to load
    **kws : 
        any additional keywords to pass to 
        :func:`~pandas.util.testing.assert_frame_equal`
    """
    try:
        from pandas.util.testing import assert_frame_equal
        this, ref = results.get_result_paths(self.output_file)

        # the data
        this = results.load_pandas_hdf(this, dataset)
        ref = results.load_pandas_hdf(ref, dataset)

        # assert the DataFrames are equal
        assert_frame_equal(this, ref, **kws)
    except Exception as e:
        raise _make_exc(self, str(e))

def test_hdf_result(self, dataset, rtol=1e-5, atol=1e-8):
    """
    Test that the HDF5 result file is the same as the reference
    result file, using ``h5py``

    Parameters
    ----------
    dataset : str
        the name of the HDF5 dataset to load
    rtol : float, optional
        the relative tolerance; default is 1e-5
    atol : float, optional
        the absolute tolerance; default is 1e-8
    """
    try:

        this, ref = results.get_result_paths(self.output_file)

        # data
        this = results.load_hdf(this, dataset)
        ref = results.load_hdf(ref, dataset)

        assert_numpy_allclose(this, ref, rtol=rtol, atol=atol)
    except Exception as e:
        raise _make_exc(self, str(e))

def test_bigfile_result(self, dataset, rtol=1e-5, atol=1e-8):
    """
    Test that the HDF5 result file is the same as the reference
    result file, using ``h5py``

    Parameters
    ----------
    dataset : str
        the name of the HDF5 dataset to load
    rtol : float, optional
        the relative tolerance; default is 1e-5
    atol : float, optional
        the absolute tolerance; default is 1e-8
    """
    try:

        this, ref = results.get_result_paths(self.output_file)

        # data
        this = results.load_bigfile(this, dataset)
        ref = results.load_bigfile(ref, dataset)

        assert_numpy_allclose(this, ref, rtol=rtol, atol=atol)
    except Exception as e:
        raise _make_exc(self, str(e))

def test_dataset_result(self, dim, stat, rtol=1e-2, atol=1e-5, skip_imaginary=True):
    """
    Test that the ``DataSet`` result file is the same as the reference
    result file

    This loads the `DataSet` and checks that the data
    arrays are ``allclose`` and also checks the meta-data

    Parameters
    ----------
    dim : {'1d', '2d'}
        load either a '1d' or '2d' `DataSet`
    stat : {'power', 'corr'}
        load either a power or correlation function measurement
    rtol : float, optional
        the relative tolerance; default is 1e-2
    atol : float, optional
        the absolute tolerance; default is 1e-8
    skip_imaginary : bool, optional
        if `True`, do not check the imaginary components of the data arrays;
        default is True
    """
    try:
        from numpy.testing import assert_array_almost_equal
        this, ref = results.get_result_paths(self.output_file)

        # load datasets
        loader = results.get_dataset_loader(dim, stat)
        this = loader(this); ref = loader(ref)

        # check each variable
        for name in ref.variables:

            # skip imaginary elements that are very close to zero
            # due to numerical differences
            if skip_imaginary and 'imag' in name:
                continue

            if name not in this.variables:
                raise AssertionError("variables name mismatch in ``assert_dataset_result``")
            assert_numpy_allclose(this[name], ref[name], rtol=rtol, atol=atol)

        # check the meta-data
        for name in ref.attrs:
            if name not in this.attrs:
                raise AssertionError("meta-data name mismatch in ``assert_dataset_result``")
            assert_array_almost_equal(this.attrs[name], ref.attrs[name])

    except Exception as e:
        raise _make_exc(self, str(e))
