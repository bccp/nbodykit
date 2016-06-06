from .pipeline import RunAlgorithm, add_run_fixture
from . import os, functions, unittest
from .. import examples_dir

class RunFiberCollisionsAlgorithm(RunAlgorithm):
    run_dir = os.path.join(examples_dir, 'fiber_collisions')
           

@add_run_fixture(__name__, RunFiberCollisionsAlgorithm, 'FiberCollisions')
class TestFiberCollisions(unittest.TestCase):
    param_file  = "test_fc.params"
    output_file = "test_fiber_collisions.hdf5"
    datasources = ['test_bianchi_data.dat']
    
    def test_exit_code(self):
        functions.test_exit_code(self)
    
    def test_exception(self):
        functions.test_exception(self) 
    
    def test_result(self):
        functions.test_pandas_hdf_result(self, "FiberCollisionGroups")

