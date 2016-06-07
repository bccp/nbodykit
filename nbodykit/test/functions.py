from . import os, cache_dir
from .. import examples_dir
from .pipeline import reference_result_md5, compute_md5sum
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
            if name not in this_output.dtype.names:
                raise AssertionError("dtype mismatch in structured data")
            assert_array_almost_equal(this_output[name], ref[name])
            
def test_dataset(self, dim, stat):
    """
    Test that the ``DataSet`` result file is the same as the reference
    result file
    """
    from nbodykit import files, dataset
    from numpy.testing import assert_array_almost_equal
    
    # this result and  reference result
    this_output = os.path.join(examples_dir, 'output', self.output_file)
    ref = os.path.join(os.path.expanduser(cache_dir), 'results', self.output_file)

    with open(this_output, 'r') as ff:
        print ff.read()
        
    # determine the file loader
    if stat == 'power':
        if dim.lower() == '2d':
            loader = lambda f: dataset.Power2dDataSet.from_nbkit(*files.Read2DPlainText(f))
        else:
            loader = lambda f: dataset.Power1dDataSet.from_nbkit(*files.Read1DPlainText(f))
    else:
        if dim.lower() == '2d':
            loader = lambda f: dataset.Corr2dDataSet.from_nbkit(*files.Read2DPlainText(f))
        else:
            loader = lambda f: dataset.Corr1dDataSet.from_nbkit(*files.Read1DPlainText(f))
    
    # load
    this_output = loader(this_output)
    ref = loader(ref)
    
    # check each variable
    for name in ref.variables:
        if name not in this_output.variables:
            raise AssertionError("variables name mismatch")
        
        # determine the precision
        precision = 5
        if 'power' in name or 'corr' in name:
            precision = 0
            
        # check
        assert_array_almost_equal(this_output[name], ref[name], precision)
        
    # check the meta-data
    for name in ref.attrs:
        if name not in this_output.attrs:
            raise AssertionError("meta-data name mismatch")
        assert_array_almost_equal(this_output.attrs[name], ref.attrs[name], 5)
        
        