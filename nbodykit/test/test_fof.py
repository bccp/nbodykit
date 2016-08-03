from .utils.pipeline import RunAlgorithm, add_run_fixture
from .utils import asserts
from . import os, unittest
from .. import examples_dir

class RunFOFAlgorithm(RunAlgorithm):
    run_dir = os.path.join(examples_dir, 'fof')
           

@add_run_fixture(__name__, RunFOFAlgorithm, 'FOF')
class TestFastPM(unittest.TestCase):
    param_file  = "test_fastpm.params"
    output_file = "fof_ll0.200_1.0000.hdf5"
    datasources = ['fastpm_1.0000']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)    
    
    def test_result(self):
        asserts.test_hdf_result(self, "FOFGroups")

