from .utils.pipeline import RunAlgorithm, add_run_fixture
from .utils import asserts
from . import os, unittest
from .. import examples_dir

class RunDescribeAlgorithm(RunAlgorithm):
    run_dir = os.path.join(examples_dir, 'describe')
           

@add_run_fixture(__name__, RunDescribeAlgorithm, 'Describe')
class Test1(unittest.TestCase):
    param_file  = "test_describe.params"
    output_file = "test_describe.dat"
    datasources = ['test_bianchi_data.dat']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)  