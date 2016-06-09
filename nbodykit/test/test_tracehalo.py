from .pipeline import RunAlgorithm, add_run_fixture
from . import os, asserts, unittest
from .. import examples_dir

class RunTraceHaloAlgorithm(RunAlgorithm):
    run_dir = os.path.join(examples_dir, 'trace-halo')
           

@add_run_fixture(__name__, RunTraceHaloAlgorithm, 'TraceHalo')
class TestTPMSnapshot(unittest.TestCase):
    param_file  = "test_tpmsnapshot.params"
    output_file = "fof_ll0.200_1.0000_ichalo.hdf5"
    datasources = ['tpm_1.0000.bin.00', 'fof_ll0.200_1.0000.labels']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)   
    
    def test_result(self):
        asserts.test_hdf_result(self, "TracedFOFGroups")

