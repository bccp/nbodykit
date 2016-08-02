from .utils.pipeline import RunAlgorithm, add_run_fixture
from .utils import asserts
from . import os, unittest
from .. import examples_dir

class RunZHistAlgorithm(RunAlgorithm):
    run_dir = os.path.join(examples_dir, 'zhist')
           

@add_run_fixture(__name__, RunZHistAlgorithm, 'RedshiftHistogram')
class Test1(unittest.TestCase):
    param_file  = "test_zhist_1.params"
    output_file = "test_zhist_1.dat"
    datasources = ['test_bianchi_randoms.dat']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)  
        
    def test_result(self):
        asserts.test_result_md5sum(self)
        
        
@add_run_fixture(__name__, RunZHistAlgorithm, 'RedshiftHistogram')
class Test2(unittest.TestCase):
    param_file  = "test_zhist_2.params"
    output_file = "test_zhist_2.dat"
    datasources = ['test_bianchi_randoms.dat']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)   
        
    def test_result(self):
        asserts.test_result_md5sum(self) 

