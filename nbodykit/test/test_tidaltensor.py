from .pipeline import RunAlgorithm, add_run_fixture
from . import os, asserts, unittest
from .. import examples_dir

class RunTidalTensorAlgorithm(RunAlgorithm):
    run_dir = os.path.join(examples_dir, 'tidaltensor')
           

@add_run_fixture(__name__, RunTidalTensorAlgorithm, 'TidalTensor')
class TestTPMSnapshot(unittest.TestCase):
    param_file  = "test_tpmsnapshot.params"
    output_file = "test_tidaltensor_tpm_1.0000.hdf5"
    datasources = ['tpm_1.0000.bin.00']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)   
    
    def test_result(self):
        asserts.test_hdf_result(self, "TidalTensor", rtol=1e-3)
