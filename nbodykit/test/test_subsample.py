from .utils.pipeline import RunAlgorithm, add_run_fixture
from .utils import asserts, results
from . import os, unittest
from .. import examples_dir

class RunSubsampleAlgorithm(RunAlgorithm):
    run_dir = os.path.join(examples_dir, 'subsample')
           

@add_run_fixture(__name__, RunSubsampleAlgorithm, 'Subsample')
class TestTPMSnapshot(unittest.TestCase):
    param_file  = "test_tpmsnapshot.params"
    output_file = "test_subsample_tpm_1.0000.hdf5"
    datasources = ['tpm_1.0000.bin.00']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self) 
        
    def test_result(self):
        asserts.test_hdf_result(self, "Subsample")
                
@add_run_fixture(__name__, RunSubsampleAlgorithm, 'Subsample')
class TestTPMSnapshotMWhite(unittest.TestCase):
    param_file  = "test_tpmsnapshot_mwhite.params"
    output_file = "test_subsample_tpm_1.0000.mwhite"
    datasources = ['tpm_1.0000.bin.00']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)    

    def test_result(self):
        import numpy
        
        # load the binary data
        this, ref = results.get_result_paths(self.output_file)
        this = numpy.fromfile(this)
        ref = numpy.fromfile(ref)
        
        # assert loaded arrays are cose
        numpy.testing.assert_allclose(this, ref, rtol=1e-5, atol=1e-8)
        
