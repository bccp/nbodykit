from .utils.pipeline import RunAlgorithm, add_run_fixture
from .utils import asserts
from . import os, unittest
from .. import examples_dir

class RunFOF6DAlgorithm(RunAlgorithm):
    run_dir = os.path.join(examples_dir, 'subhalo')
           
@add_run_fixture(__name__, RunFOF6DAlgorithm, 'FOF6D')
class TestTPMSnapshot(unittest.TestCase):
    param_file  = "test_tpmsnapshot.params"
    output_file = "sub_0.180_100_1.0000.hdf5"
    datasources = ['tpm_1.0000.bin.00', 'fof_ll0.200_1.0000.labels']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)    
    
    def test_result(self):
        asserts.test_hdf_result(self, "Subhalos", rtol=1e-3, atol=1e-5)

