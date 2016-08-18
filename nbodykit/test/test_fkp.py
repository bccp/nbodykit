from . import os, unittest, pytest

from .utils.datasource import UnitTestDataSource
from nbodykit.fkp import FKPCatalog
from nbodykit.cosmology import Cosmology

import logging
import numpy
        
class TestFKP(unittest.TestCase):
    """
    Test the FKPCatalog class
    """
    def setUp(self):
        """
        Do some basic setup of the unit testing suite
        """
        # basic logging setup
        logging.basicConfig()

        # initialize a particle mesh object
        from pmesh.pm import ParticleMesh
        from mpi4py import MPI
        self.pm = ParticleMesh(Nmesh=[128]*3, BoxSize=2.0, comm=MPI.COMM_WORLD)

        # a cosmology
        self.cosmo = Cosmology()
        
        # default size of data and randoms
        self.N = 100
            
    def test_required_cosmo(self):
        """
        The `data` and `randoms` DataSources must have a valid :attr:`cosmo` attribute.
        
        Test that a :exception:`ValueError` is raised if :attr`cosmo` is missing
        """
        data    = UnitTestDataSource(self.N, [('Position', (float, 3))])
        randoms = UnitTestDataSource(self.N, [('Position', (float, 3))])
        
        with pytest.raises(ValueError):
            catalog = FKPCatalog(data, randoms)

    def test_cosmo_mismatch(self):
        """
        The `data` and `randoms` DataSources must have a valid :attr:`cosmo` attribute.
        
        Test that a :exception:`ValueError` is raised if `data` and `randoms` have
        different :attr`cosmo` attributes
        """
        data    = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N, [('Position', (float, 3))])
        
        with pytest.raises(ValueError):
            catalog = FKPCatalog(data, randoms)
            
    def test_comm_mismatch(self):
        """
        Test that a :exception:`ValueError` is raised if `data` and `randoms` have
        different :attr`comm` attributes
        """
        from mpi4py import MPI
        data    = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo, comm=MPI.COMM_WORLD)
        randoms = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo, comm=None)
        
        with pytest.raises(ValueError):
            catalog = FKPCatalog(data, randoms)
            
    def test_closed_read(self):
        """
        Test that a :exception:`ValueError` is raised if the catalog is closed
        while attempting to call :func:`~nbodykit.fkp.FKPCatalog.read`
        """
        data    = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        
        catalog = FKPCatalog(data, randoms)
        with pytest.raises(ValueError):
            [pos] = catalog.read('data', ['Position'])
            
    def test_closed_paint(self):
        """
        Test that a :exception:`ValueError` is raised if the catalog is closed
        while attempting to call :func:`~nbodykit.fkp.FKPCatalog.paint`
        """
        data    = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        
        catalog = FKPCatalog(data, randoms)
        with pytest.raises(ValueError):
            stats = catalog.paint(self.pm)
            
    def test_missing_fsky(self):
        """
        Test that a :exception:`AttributeError` is raised if no :attr:`fsky` attribute
        is specified
        """
        data    = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        
        catalog = FKPCatalog(data, randoms)
        with pytest.raises(AttributeError):
            fsky = catalog.fsky
            
    def test_closed_alpha(self):
        """
        Test that a :exception:`ValueError` is raised if the catalog is closed
        while attempting to access :attr:`alpha`
        """
        data    = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        
        catalog = FKPCatalog(data, randoms)
        with pytest.raises(ValueError):
            alpha = catalog.alpha
    
    def test_data_setter(self):
        """
        Test that the :attr:`data` can be properly set when the FKPCatalog
        is either open or closed
        
        Test this by checking that the :attr:`alpha` attribute is properly computed
        """
        data1   = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        data2   = UnitTestDataSource(2*self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        
        # initialize the catalog with the first data1
        catalog = FKPCatalog(data1, randoms)
        
        # data1 has alpha = 1.0
        with catalog:
            self.assertTrue(catalog.alpha == 1.0)
        
        # assign data2 when closed
        catalog.data = data2
        with catalog:
            self.assertTrue(catalog.alpha == 2.0)
        
        # assign data2 when open
        with catalog:
            catalog.data = data2
            self.assertTrue(catalog.alpha == 2.0)
            
    def test_invalid_nbar_types(self):
        """
        Test that a :exception:`TypeError` or :exception:`ValueError` is raised 
        for invalid :attr:`nbar` types
        """
        data    = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        
        # string is not a valid filename
        with pytest.raises(ValueError):
            catalog = FKPCatalog(data, randoms, nbar="invalid_filename")
            
        # value is not a scalar
        with pytest.raises(TypeError):
            catalog = FKPCatalog(data, randoms, nbar=[0,1,2])
                            
    def test_nbar_file_columns(self):
        """
        Test that a :exception:`ValueError` is raised if the file specified
        for :attr:`nbar` has the wrong number of columns
        """
        import tempfile
        
        data    = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        
        # wrong number of columns
        with tempfile.NamedTemporaryFile() as ff:
            ff.write(("0 0 0 0 0 0 0\n0 0 0 0 0 0 0\n").encode()); ff.seek(0)
            with pytest.raises(ValueError):
                catalog = FKPCatalog(data, randoms, nbar=ff.name)
                
        # (z, nz) file
        with tempfile.NamedTemporaryFile() as ff:
            
            z = numpy.linspace(0, 1.0, 100)
            numpy.savetxt(ff, numpy.vstack([z, 2*z]).T)
            ff.seek(0)
            
            catalog = FKPCatalog(data, randoms, nbar=ff.name)
            numpy.testing.assert_allclose(2*z, catalog.nbar(z), atol=1e-3, rtol=1e-3)
            
        # (z_min, z_max, z_cen, nz) file
        with tempfile.NamedTemporaryFile() as ff:
            
            z = numpy.linspace(0, 1.0, 100)
            numpy.savetxt(ff, numpy.vstack([z, z, z, 2*z]).T)
            ff.seek(0)
            
            catalog = FKPCatalog(data, randoms, nbar=ff.name)
            numpy.testing.assert_allclose(2*z, catalog.nbar(z), atol=1e-3, rtol=1e-3)
            
    def test_constant_nbar(self):
        """
        Test that the `nbar` function returns a constant array of the correct length
        (internally using the strides technique to save memory) 
        """
        data    = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        
        # constant nbar
        catalog = FKPCatalog(data, randoms, nbar=1e-4)
        
        z = numpy.linspace(0., 1.0, 100)
        nbar = catalog.nbar(z)
        numpy.testing.assert_allclose(nbar, 1e-4)
        self.assertTrue(len(nbar) == 100)
        
    def test_randoms_nbar_exception(self):
        """
        Test that a warning is raised if the ``RedshiftHistogram`` algorithm fails
        when computing n(z) from the `randoms` catalog
        """
        data    = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N, [('Position', (float, 3)), ('Redshift', float)], cosmo=self.cosmo)
        
        # invalid fsky value --> will cause 
        catalog = FKPCatalog(data, randoms, fsky="1.0")
        
        # catch the warning raised when n(z) algorithm fails
        with pytest.warns(RuntimeWarning):
            with catalog:
               pass
               
    def test_invalid_read(self):
        """
        Test that when calling ``read`` that both the stream name and the column
        names are valid
        """
        from nbodykit.core import DataSource
        
        data    = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N*100, [('Position', (float, 3))], cosmo=self.cosmo)
        
        # initialize the catalog
        catalog = FKPCatalog(data, randoms, nbar=1e-4)
        
        # invalid column name
        with pytest.raises(DataSource.MissingColumn):
            with catalog:
                [col] = catalog.read('data', ['MissingColumn'])
                
        # invalid stream name
        with pytest.raises(ValueError):
            with catalog:
                [pos] = catalog.read('missing_stream', ['Position'])
                
    def test_compute_fkp_weights(self):
        """
        Test the calculation of FKP weights in FKPCatalog -- must specify :attr:`P0_fkp`
        if we want to compute n(z)
        """
        from nbodykit.core import DataSource
        
        data    = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N*100, [('Position', (float, 3))], cosmo=self.cosmo)
        
        # missing P0_fkp raises a ValueError
        with pytest.raises(ValueError):
            catalog = FKPCatalog(data, randoms, nbar=1e-4, compute_fkp_weights=True)
            with catalog:
                [weight] = catalog.read('data', ['FKPWeight'])
                
        # this should work fine
        catalog = FKPCatalog(data, randoms, nbar=1e-4, compute_fkp_weights=True, P0_fkp=1e4)
        with catalog:
            [weight] = catalog.read('data', ['FKPWeight'])
            numpy.testing.assert_allclose(weight, 0.5) # fkp_weight = 1.0 / (1.0 + nbar * P0_fkp) = 0.5
                
    def test_missing_z_for_nbar(self):
        """
        Test that if we need the `Redshift` column to compute n(z), that it is valid
        in the input data sources
        """
        import tempfile
        from nbodykit.core import DataSource
        
        data    = UnitTestDataSource(self.N, [('Position', (float, 3))], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N*100, [('Position', (float, 3))], cosmo=self.cosmo)
                
        with tempfile.NamedTemporaryFile() as ff:
            
            # write the (z, nz) file
            z = numpy.linspace(0, 1.0, 100)
            numpy.savetxt(ff, numpy.vstack([z, 2*z]).T)
            ff.seek(0)
            
            # "Redshift" column is missing from data
            catalog = FKPCatalog(data, randoms, nbar=ff.name)
            with pytest.raises(DataSource.MissingColumn):
                with catalog:
                    [nbar] = catalog.read('data', ['Nbar'])
                    
        # "Redshift" column is required to compute n(z)
        catalog = FKPCatalog(data, randoms)
        with pytest.raises(DataSource.MissingColumn):
            with catalog:
                [nbar] = catalog.read('data', ['Nbar'])
                    
    def test_missing_fsky(self):
        """
        Test that if we have to compute n(z) from the randoms, then :attr:`fsky`
        is supplied
        """        
        data    = UnitTestDataSource(self.N, [('Position', (float, 3)), ('Redshift', float)], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N*100, [('Position', (float, 3)), ('Redshift', float)], cosmo=self.cosmo)
                 
        # need fsky
        catalog = FKPCatalog(data, randoms)
        with pytest.raises(ValueError):
            with catalog:
                [nbar] = catalog.read('randoms', ['Nbar'])
                
    def test_no_z_in_randoms(self):
        """
        Test that if we have to compute n(z) from the randoms, then the `randoms`
        DataSource has a `Redshift` column
        """        
        data    = UnitTestDataSource(self.N, [('Position', (float, 3)), ('Redshift', float)], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N*100, [('Position', (float, 3))], cosmo=self.cosmo)
                 
        # need Redshift column in randoms too
        catalog = FKPCatalog(data, randoms, fsky=1.0)
        with pytest.raises(ValueError):
            with catalog:
                [nbar] = catalog.read('data', ['Nbar'])
                        
    def test_read(self):
        """
        Test a valid input to the :func:`FKPCatalog.read` function
        """
        data    = UnitTestDataSource(self.N, [('Position', (float, 3)), ('Redshift', float)], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N*100, [('Position', (float, 3)), ('Redshift', float)], cosmo=self.cosmo)

        # this should work fine
        catalog = FKPCatalog(data, randoms, fsky=1.0)
        with catalog:
            [[pos, nbar]] = catalog.read('data', ['Position', 'Nbar'], full=True)
            
    def test_small_box(self):
        """
        Test that a RuntimeWarning is thrown if the BoxSize is too small to hold all of
        the data particles
        """
        data    = UnitTestDataSource(self.N, [('Position', (float, 3)), ('Redshift', float)], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N, [('Position', (float, 3)), ('Redshift', float)], cosmo=self.cosmo)

        # box is too small
        catalog = FKPCatalog(data, randoms, fsky=1.0, BoxSize=0.5)
        with catalog:
            with pytest.warns(RuntimeWarning):
                [[pos, nbar]] = catalog.read('data', ['Position', 'Nbar'], full=True)
                
    def test_paint(self):
        """
        Test a valid input to the :func:`FKPCatalog.paint` function
        """
        data    = UnitTestDataSource(self.N, [('Position', (float, 3)), ('Nbar', float)], cosmo=self.cosmo)
        randoms = UnitTestDataSource(self.N*100, [('Position', (float, 3)), ('Nbar', float)], cosmo=self.cosmo)

        # this should work fine
        catalog = FKPCatalog(data, randoms, fsky=1.0)
        with catalog:
            real, stats = catalog.paint(self.pm)
            self.assertTrue(stats['alpha'] == 0.01)
