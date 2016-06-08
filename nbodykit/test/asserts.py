from . import os, cache_dir
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

def test_exit_code(self):
    """
    Assert that the exit code is equal to 0
    """
    assert self.run_fixture.exit_code == 0
    
def test_exception(self):
    """
    Assert that no exception occurred, by analyzing the 
    ``stderr`` output
    """
    with open("run.stderr", 'r') as stderr:
        lines = stderr.read()
        assert "Error" not in lines and "Traceback" not in lines
            
def test_result_md5sum(self):
    """
    Assert that the result file from the pipeline run has the same 
    MD5 sum as the reference result file in the cache directory
    """
    this, ref = results.get_result_paths(self.output_file)
    assert results.compute_md5sum(this) == results.compute_md5sum(ref)

def test_pandas_hdf_result(self, dataset):
    """
    Test that the pandas HDF5 result file is the same as the reference
    result file, by checking that the ``DataFrame`` objects are 
    the same
    """
    from pandas.util.testing import assert_frame_equal
    this, ref = results.get_result_paths(self.output_file)
    
    # the data
    this = results.load_pandas_hdf(this, dataset)
    ref = results.load_pandas_hdf(ref, dataset)
    
    # assert the DataFrames are equal
    assert_frame_equal(this, ref)
    
def test_hdf_result(self, dataset):
    """
    Test that the HDF5 result file is the same as the reference
    result file, using ``h5py``
    """
    this, ref = results.get_result_paths(self.output_file)
    
    # data
    this = results.load_hdf(this, dataset)
    ref = results.load_hdf(ref, dataset)
    
    assert_numpy_allclose(this, ref)
        
def test_dataset_result(self, dim, stat, skip_imaginary=True):
    """
    Test that the ``DataSet`` result file is the same as the reference
    result file
    """
    from numpy.testing import assert_array_almost_equal  
    this, ref = results.get_result_paths(self.output_file)
    
    # load datasets
    this = results.load_dataset(this, dim, stat)
    ref = results.load_dataset(ref, dim, stat)
    
    # check each variable
    for name in ref.variables:
        
        # skip imaginary elements that are very close to zero
        # due to numerical differences
        if skip_imaginary and 'imag' in name:
            continue
        
        if name not in this.variables:
            raise AssertionError("variables name mismatch in ``assert_dataset_result``")
        assert_numpy_allclose(this[name], ref[name], rtol=0.01, atol=1e-5)
        
    # check the meta-data
    for name in ref.attrs:
        if name not in this.attrs:
            raise AssertionError("meta-data name mismatch in ``assert_dataset_result``")
        assert_array_almost_equal(this.attrs[name], ref.attrs[name])
        
        