from pytest_pipeline import PipelineRun, mark
from .. import verify_data_in_cache, cache_dir
from .. import os, sys, pytest
from ... import bin_dir, pkg_dir
import platform

def skip_fedora(test):
    return pytest.mark.skipif('FEDORA' in platform.platform().upper(), reason="https://bugzilla.redhat.com/show_bug.cgi?id=1235044")(test)

# global counter for the number of pipeline tests
_pipeline_tests = 0

def default_run_fixture(cls, name, timeout=60):
    """
    Create a class fixture to run the analysis pipeline
    
    Notes
    -----
    *   The input `cls` should have attributes `param_file` which specifies
        the input configuration file and `run_dir`, which specifies where 
        `param_file` lives
    *   The analysis run will be performed in parallel using 2 processes
    
    Parameters
    ----------
    cls : subclass of PipelineRun
        a subclass of PipelineRun that will create the run fixture
    name : str
        the name of the nbodykit Algorithm to run
    timeout : int, optional
        an optional timeout in seconds -- if this limit is exceeded, the
        analysis run is terminated and the test fails
    """
    param_file = os.path.join(cls.run_dir, cls.param_file)    
    cmd = "mpirun -n 2 python %s/nbkit.py %s %s" %(bin_dir, name, param_file)
    return cls.class_fixture(cmd=cmd, stdout='run.stdout', stderr="run.stderr", timeout=timeout)
    
def add_run_fixture(modname, basecls, name, make_fixture=default_run_fixture, timeout=60):
    """
    A class decorator that creates a run fixture and adds it to the class as 
    a ``pytest`` fixture 
    
    Parameters
    ----------
    modname : str
        the name of the module to which the fixture class should be attached; this
        is needed so that each test module knows where to find the relevant fixtures
    basecls : subclass of PipelineRun
        the base class for the run fixture, with any functions added that will be
        executed prior to the pipeline run
    name : str
        the name of the nbodykit Algorithm to run
    make_fixture : callable, optional
        a function that returns a class fixture that runs the pipeline; default is
        ``default_run_fixture``
    timeout : int, optional
        an optional timeout in seconds -- if this limit is exceeded, the
        analysis run is terminated and the test fails; default is 60 seconds
    """
    global _pipeline_tests
        
    def class_rebuilder(cls):        
        # create the run fixture class and attach the relevant info
        new_cls = type("RunClass%d" %_pipeline_tests, (basecls, cls), {})
    
        # make the run fixture and attach it to the module
        run_name = 'run_%d' %(_pipeline_tests)
        mod = sys.modules[modname]
        setattr(mod, run_name, make_fixture(new_cls, name, timeout=timeout))
    
        # initialize the mark decorator
        decorator = pytest.mark.usefixtures(run_name)
        
        return decorator(cls)
    
    _pipeline_tests += 1
    return class_rebuilder

class RunAlgorithm(PipelineRun):
    """
    Run a pipeline that tests an `Algorithm`. Prior to running
    this class will: 
    
        - load the necessary data to the cache directory
        - initialize the `output` directory in `nbodykit/examples`
        - set the `NBKIT_CACHE` and `NBKIT_HOME` directories
    """
    @mark.before_run
    def verify_cache(self):
        """
        Verify that all the necessary data has been downloaded
        and cached in order to run this test
        """
        for d in self.datasources:
            verify_data_in_cache(d)
            
    @mark.before_run
    def initialize_output(self):
        """
        Make the `output` directory in `nbodykit/examples`, if 
        it does not exist yet
        """
        results_dir = os.path.join(pkg_dir, 'examples', 'output')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
    @mark.before_run
    def set_env(self):
        """
        Set the ``NBKIT_CACHE`` and ``NBKTIT_HOME`` environment
        variables
        """
        if 'NBKIT_CACHE' not in os.environ:
            os.environ['NBKIT_CACHE'] = os.path.expanduser(cache_dir)
        if 'NBKIT_HOME' not in os.environ:
            os.environ['NBKIT_HOME'] = pkg_dir
