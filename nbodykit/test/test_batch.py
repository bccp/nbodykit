from .utils.pipeline import RunAlgorithm, add_run_fixture, skip_fedora
from .utils import asserts
from . import os, unittest
from .. import examples_dir, bin_dir

class RunBatchAlgorithm(RunAlgorithm):
    run_dir = os.path.join(examples_dir, 'batch')
           

def bash_fixture(cls, name, **kws):
    """
    Return a run fixture that uses bash to execute the param file
    """
    param_file = os.path.join(cls.run_dir, cls.param_file)
    cmd = "bash %s" %param_file
    return cls.class_fixture(cmd=cmd, stdout='run.stdout', stderr="run.stderr", **kws)
    
def batch_fixture(cls, name, **kws):
    """
    Return a run fixture that runs nbkit-batch
    """
    param_file = os.path.join(cls.run_dir, cls.param_file)
    extra_file = os.path.join(cls.run_dir, cls.extra_file)

    args = (bin_dir, name, param_file, extra_file)
    cmd = "mpirun -np 6 python %s/nbkit-batch.py %s 2 -c %s -i \"los: [x, y, z]\" --extras %s --debug" %args
    return cls.class_fixture(cmd=cmd, stdout='run.stdout', stderr="run.stderr", **kws)

@skip_fedora
@add_run_fixture(__name__, RunBatchAlgorithm, 'FFTPower', timeout=30, make_fixture=bash_fixture)
class TestStdin(unittest.TestCase):
    param_file  = "test_stdin.sh"
    output_file = "test_stdin.dat"
    datasources = ['fastpm_1.0000']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self) 
    
    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'power')
        
@add_run_fixture(__name__, RunBatchAlgorithm, 'FFTPower', timeout=120, make_fixture=batch_fixture)
class TestBatch(unittest.TestCase):
    param_file  = "test_power_batch.template"
    extra_file  = "extra.template"
    output_file = "test_batch_power_fastpm_1d_zlos_task_3.dat"
    datasources = ['fastpm_1.0000']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self) 
    
    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'power')

