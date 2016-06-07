from . import os, cache_dir
from .pipeline import reference_result_md5, compute_md5sum
from .. import examples_dir
import shutil

def test_exit_code(self):
    """
    Test the exit code
    """
    assert self.run_fixture.exit_code == 0
    
def test_exception(self):
    """
    Test for an exception, by looking in the stderr output
    """
    with open("run.stderr", 'r') as stderr:
        lines = stderr.read()
        assert "Error" not in lines and "Traceback" not in lines
        
def test_result_md5(self):
    """
    Test that the result file has the same MD5 sum as the reference
    result file
    """
    this_output = os.path.join(examples_dir, 'output', self.output_file)
    with open(this_output, 'r') as ff:
        print this_output + '\n-'*20
        print ff.read()
    assert compute_md5sum(this_output) == reference_result_md5(self.output_file)
    
def test_pandas_hdf_result(self, dataset):
    """
    Test that the pandas HDF5 result file is the same as the reference
    result file
    """
    import pandas as pd
    from pandas.util.testing import assert_frame_equal
    
    # this result
    this_output = os.path.join(examples_dir, 'output', self.output_file)
    this_output = pd.read_hdf(this_output, dataset)
    
    # reference result
    ref = os.path.join(os.path.expanduser(cache_dir), 'results', self.output_file)
    ref = pd.read_hdf(ref, dataset)
    
    assert_frame_equal(this_output, ref)
    
def test_hdf_result(self, dataset):
    """
    Test that the HDF5 result file is the same as the reference
    result file, using ``h5py``
    """
    import h5py
    from numpy.testing import assert_array_almost_equal
    
    # this result
    this_output = os.path.join(examples_dir, 'output', self.output_file)
    this_output = h5py.File(this_output, mode='r')[dataset][...]
    
    # reference result
    ref = os.path.join(os.path.expanduser(cache_dir), 'results', self.output_file)
    ref = h5py.File(ref, mode='r')[dataset][...]
    
    # normal numpy array
    if ref.dtype.names is None:
        assert_array_almost_equal(this_output, ref)
    # test each element of structured array
    else:    
        for name in ref.dtype.names:
            if name not in this_output.dtype.names: return False
            assert_array_almost_equal(this_output[name], ref[name])