from .utils.pipeline import RunAlgorithm, add_run_fixture
from .utils import asserts
from . import os, unittest, pytest
from .. import examples_dir

# import halotools for HOD power tests
try: import halotools; missing_halotools = False
except: missing_halotools = True

# import classylss for Zeldovich power tests
try: import classylss; missing_classylss = False
except: missing_classylss = True
    
class RunPowerAlgorithm(RunAlgorithm):
    run_dir = os.path.join(examples_dir, 'power')
           

@add_run_fixture(__name__, RunPowerAlgorithm, 'BianchiFFTPower')
class TestBianchi1(unittest.TestCase):
    param_file  = "test_bianchi_1.params"
    output_file = "test_power_bianchi_1.dat"
    datasources = ['test_bianchi_data.dat', 'test_bianchi_randoms.dat']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)   
    
    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'power') 


@add_run_fixture(__name__, RunPowerAlgorithm, 'BianchiFFTPower')
class TestBianchi2(unittest.TestCase):
    param_file  = "test_bianchi_2.params"
    output_file = "test_power_bianchi_2.dat"
    datasources = ['test_bianchi_data.dat']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)    
    
    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'power')


@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestCrossMomentum(unittest.TestCase):
    param_file  = "test_cross_momentum.params"
    output_file = "test_power_cross_momentum.dat"
    datasources = ['fastpm_1.0000', 'fof_ll0.200_1.0000.hdf5']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)  
    
    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'power')


@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestCrossPower(unittest.TestCase):
    param_file  = "test_cross_power.params"
    output_file = "test_power_cross.dat"
    datasources = ['fastpm_1.0000', 'fof_ll0.200_1.0000.hdf5']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)   
    
    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'power')
         

@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestFastPM1D(unittest.TestCase):
    param_file  = "test_fastpm_1d.params"
    output_file = "test_power_fastpm_1d.dat"
    datasources = ['fastpm_1.0000']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)   
    
    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'power')
         

@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestFastPM2D(unittest.TestCase):
    param_file  = "test_fastpm_2d.params"
    output_file = "test_power_fastpm_2d.dat"
    datasources = ['fastpm_1.0000']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)   
    
    def test_result(self):
        asserts.test_dataset_result(self, '2d', 'power')
         

@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestFOFGroups(unittest.TestCase):
    param_file  = "test_fofgroups.params"
    output_file = "test_power_fofgroups.dat"
    datasources = ['fof_ll0.200_1.0000.hdf5']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)    
    
    def test_result(self):
        asserts.test_dataset_result(self, '2d', 'power')
         

# https://github.com/bccp/nbodykit/issues/298
@pytest.mark.skipif(True, reason="This seems to be sensitive to numpy version and round-off-errors")
@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestPandasHDF(unittest.TestCase):
    param_file  = "test_pandas_hdf.params"
    output_file = "test_power_pandas_hdf.dat"
    datasources = ['pandas_data.hdf5']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)    
         
    def test_result(self):
        asserts.test_dataset_result(self, '2d', 'power')

@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestPandasPlaintext(unittest.TestCase):
    param_file  = "test_pandas_plaintext.params"
    output_file = "test_power_pandas_plaintext.dat"
    datasources = ['plaintext_data.txt']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self) 
    
    def test_result(self):
        asserts.test_dataset_result(self, '2d', 'power')
         

@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestPlaintext(unittest.TestCase):
    param_file  = "test_plaintext.params"
    output_file = "test_power_plaintext.dat"
    datasources = ['plaintext_data.txt']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self) 
    
    def test_result(self):
        asserts.test_dataset_result(self, '2d', 'power')

         

@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestTPMSnapshot1D(unittest.TestCase):
    param_file  = "test_tpmsnapshot_1d.params"
    output_file = "test_power_tpmsnapshot_1d.dat"
    datasources = ['tpm_1.0000.bin.00']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)   
    
    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'power')

         

@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestTPMSnapshot2D(unittest.TestCase):
    param_file  = "test_tpmsnapshot_2d.params"
    output_file = "test_power_tpmsnapshot_2d.dat"
    datasources = ['tpm_1.0000.bin.00']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self) 
    
    def test_result(self):
        asserts.test_dataset_result(self, '2d', 'power')
         

@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestMWhiteHalo(unittest.TestCase):
    param_file  = "test_mwhite_halo.params"
    output_file = "test_power_mwhite_halo.dat"
    datasources = ['mwhite_halo.fofp']
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)
    
    def test_result(self):
        asserts.test_dataset_result(self, '2d', 'power')

@pytest.mark.skipif(missing_halotools, reason="requires `halotools` package")
@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestZhengHOD(unittest.TestCase):
    param_file  = "test_zheng_hod.params"
    output_file = "test_power_zheng_hod.dat"
    datasources = []
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self)
    
    def test_result(self):
        asserts.test_dataset_result(self, '2d', 'power')

@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestSubsample1D(unittest.TestCase):
    param_file  = "test_subsample_1d.params"
    output_file = "test_power_subsample_1d.dat"
    datasources = ['subsample_1.0000.hdf5']

    def test_exit_code(self):
        asserts.test_exit_code(self)

    def test_exception(self):
        asserts.test_exception(self)

    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'power')
        
# https://github.com/bccp/nbodykit/issues/298
@pytest.mark.skipif(True, reason="This seems to be sensitive to numpy version and round-off-errors")
@pytest.mark.skipif(missing_classylss, reason="requires `classylss` package")
@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestZeldovich1D(unittest.TestCase):
    param_file  = "test_zeldovich_pk.params"
    output_file = "test_power_zeldovich_pk.dat"
    datasources = []
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self) 
    
    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'power')
        
# https://github.com/bccp/nbodykit/issues/298
@pytest.mark.skipif(True, reason="This seems to be sensitive to numpy version and round-off-errors")
@pytest.mark.skipif(missing_classylss, reason="requires `classylss` package")
@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestZeldovich2D(unittest.TestCase):
    param_file  = "test_zeldovich_pkmu.params"
    output_file = "test_power_zeldovich_pkmu.dat"
    datasources = []
    
    def test_exit_code(self):
        asserts.test_exit_code(self)
    
    def test_exception(self):
        asserts.test_exception(self) 
    
    def test_result(self):
        asserts.test_dataset_result(self, '2d', 'power')

@add_run_fixture(__name__, RunPowerAlgorithm, 'FFTPower')
class TestGrid1D(unittest.TestCase):
    param_file  = "test_grid_1d.params"
    output_file = "test_power_grid_1d.dat"
    datasources = ['bigfile_grid']

    def test_exit_code(self):
        asserts.test_exit_code(self)

    def test_exception(self):
        asserts.test_exception(self)

    def test_result(self):
        asserts.test_dataset_result(self, '1d', 'power')

