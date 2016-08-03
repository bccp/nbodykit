from .utils.pipeline import RunAlgorithm, add_run_fixture
from .utils import asserts
from . import os, unittest
from .. import examples_dir

class RunBoxSizeAlgorithm(RunAlgorithm):
    run_dir = os.path.join(examples_dir, 'boxsize')
           

@add_run_fixture(__name__, RunBoxSizeAlgorithm, 'TestBoxSize')
class TestBoxSize(unittest.TestCase):
    param_file  = "test_boxsize.params"
    output_file = "test_boxsize.dat"
    datasources = ['test_bianchi_randoms.dat']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)    
    
    def test_result(self):
        asserts.test_result_md5sum(self)

