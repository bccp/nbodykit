from .utils.pipeline import RunAlgorithm, add_run_fixture
from .utils import asserts
from . import os, unittest
from .. import examples_dir

class RunCorrAlgorithm(RunAlgorithm):
    run_dir = os.path.join(examples_dir, 'corr')
           

@add_run_fixture(__name__, RunCorrAlgorithm, 'PairCountCorrelation')
class TestCrossCorr(unittest.TestCase):
    param_file  = "test_cross_corr.params"
    output_file = "test_corr_cross.dat"
    datasources = ['tpm_1.0000.bin.00', 'fof_ll0.200_1.0000.hdf5']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)
    
    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'corr')

  
@add_run_fixture(__name__, RunCorrAlgorithm, 'PairCountCorrelation')
class TestMWhiteCorr1D(unittest.TestCase):
    param_file  = "test_mwhite_halo_1d.params"
    output_file = "test_corr_mwhite_halo_1d.dat"
    datasources = ['mwhite_halo.fofp']

    def test_exit_code(self):
        asserts.test_exit_code(self)

    def test_exception(self):
        asserts.test_exception(self)
    
    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'corr')
        
@add_run_fixture(__name__, RunCorrAlgorithm, 'PairCountCorrelation')
class TestMWhiteCorr2D(unittest.TestCase):
    param_file  = "test_mwhite_halo_2d.params"
    output_file = "test_corr_mwhite_halo_2d.dat"
    datasources = ['mwhite_halo.fofp']

    def test_exit_code(self):
        asserts.test_exit_code(self)

    def test_exception(self):
        asserts.test_exception(self)
    
    def test_result(self):
        asserts.test_dataset_result(self, '2d', 'corr')
        
@add_run_fixture(__name__, RunCorrAlgorithm, 'PairCountCorrelation')
class TestMWhiteCorrPoles(unittest.TestCase):
    param_file  = "test_mwhite_halo_poles.params"
    output_file = "test_corr_mwhite_halo_poles.dat"
    datasources = ['mwhite_halo.fofp']

    def test_exit_code(self):
        asserts.test_exit_code(self)

    def test_exception(self):
        asserts.test_exception(self)
    
    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'corr')
        
@add_run_fixture(__name__, RunCorrAlgorithm, 'FFTCorrelation')
class TestFFTCorr(unittest.TestCase):
    param_file  = "test_fft_corr.params"
    output_file = "test_corr_fft.dat"
    datasources = ['tpm_1.0000.bin.00']

    def test_exit_code(self):
        asserts.test_exit_code(self)

    def test_exception(self):
        asserts.test_exception(self)
    
    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'corr')