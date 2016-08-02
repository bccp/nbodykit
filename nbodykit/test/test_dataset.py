from . import os, unittest, pytest, download_results_file
from .utils import asserts
from .. import pkg_dir
import numpy
        
class TestDataSet(unittest.TestCase):
    """
    Test the DataSet class
    """
    
    def setUp(self):
        from nbodykit import files, dataset
        
        # make sure the download file exists
        results_file = 'test_power_plaintext.dat'
        output_dir = os.path.join(pkg_dir, 'examples', 'output')
        download_results_file(results_file, output_dir)
        
        # initialize
        filename = os.path.join(output_dir, results_file)
        self.dataset = dataset.Power2dDataSet.from_nbkit(*files.Read2DPlainText(filename))
    
    def test_getitem(self):
        
        with pytest.raises(KeyError) as e_info:
            self.dataset['error']
            
        k_cen = self.dataset['k_cen']
        sliced = self.dataset[['k', 'mu', 'power']]
        sliced = self.dataset[('k', 'mu', 'power')]
        
        with pytest.raises(KeyError) as e_info:
            self.dataset[['k', 'mu', 'error']]
        
    
    def test_array_slice(self):
        
        # get the first mu column
        sliced = self.dataset[:,0]
        self.assertTrue(sliced.shape[0] == self.dataset.shape[0])
        self.assertTrue(len(sliced.shape) == 1)
        self.assertTrue(sliced.dims == ['k_cen'])
        
        # get the first mu column but keep dimension
        sliced = self.dataset[:,[0]]
        self.assertTrue(sliced.shape[0] == self.dataset.shape[0])
        self.assertTrue(sliced.shape[1] == 1)
        self.assertTrue(sliced.dims == ['k_cen', 'mu_cen'])
        
        
    def test_list_array_slice(self):
                
        # get the first and last mu column
        sliced = self.dataset[:,[0, -1]]
        self.assertTrue(len(sliced.shape) == 2)
        self.assertTrue(sliced.dims == ['k_cen', 'mu_cen'])
        
        # make sure we grabbed the right data
        for var in self.dataset.variables:
            numpy.testing.assert_array_equal(self.dataset[var][:,[0,-1]], sliced[var])
            
    def test_variable_set(self):
        
        modes = numpy.ones(self.dataset.shape)
        self.dataset['modes'] = modes
        
    def test_copy(self):
        
        copy = self.dataset.copy()
        for var in self.dataset.variables:
            numpy.testing.assert_array_equal(self.dataset[var], copy[var])
            
    def test_rename_variable(self):
        
        test = numpy.zeros(self.dataset.shape)
        self.dataset['test'] = test
        self.dataset.rename_variable('test', 'renamed_test')
        
    def test_sel(self):
        
        # no exact match fails
        with pytest.raises(IndexError) as e_info:
            sliced = self.dataset.sel(k_cen=0.1)
        
        # this should be squeezed
        sliced = self.dataset.sel(k_cen=0.1, method='nearest')
        self.assertTrue(len(sliced.dims) == 1)
        
        # this is not squeezed
        sliced = self.dataset.sel(k_cen=[0.1], method='nearest')
        self.assertTrue(sliced.shape[0] == 1)
        
        # slice in a specific k-range
        sliced = self.dataset.sel(k_cen=slice(0.02, 0.15), mu_cen=[0.5], method='nearest')
        self.assertTrue(sliced.shape[1] == 1)
        self.assertTrue(numpy.alltrue((sliced['k'] >= 0.02)&(sliced['k'] <= 0.15)))
        
    def test_squeeze(self):
        
        # need to specify which dimension to squeeze
        with pytest.raises(ValueError) as e_info:
            squeezed = self.dataset.squeeze()
        with pytest.raises(ValueError) as e_info:
            squeezed = self.dataset[[0],[0]].squeeze()
            
        sliced = self.dataset[:,[2]]
        with pytest.raises(ValueError) as e_info:
            squeezed = sliced.squeeze('k_cen')
        squeezed = sliced.squeeze('mu_cen')
        self.assertTrue(len(squeezed.dims) == 1)
        self.assertTrue(squeezed.shape[0] == sliced.shape[0])
        
    def test_average(self):
        import warnings
        
        # unweighted
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            avg = self.dataset.average('mu_cen')
            for var in self.dataset.variables:
                if var in self.dataset._fields_to_sum:
                    x = numpy.nansum(self.dataset[var], axis=-1)
                else:
                    x = numpy.nanmean(self.dataset[var], axis=-1)
                numpy.testing.assert_allclose(x, avg[var])
            
            # weighted
            weights = numpy.random.random(self.dataset.shape)
            self.dataset['weights'] = weights
            avg = self.dataset.average('mu_cen', weights='weights')
        
        
            for var in self.dataset.variables:
                if var in self.dataset._fields_to_sum:
                    x = numpy.nansum(self.dataset[var], axis=-1)
                else:
                    x = numpy.nansum(self.dataset[var]* self.dataset['weights'], axis=-1)  / self.dataset['weights'].sum(axis=-1)
                numpy.testing.assert_allclose(x, avg[var])
            
            
    def test_reindex(self):
        import warnings
        
        with pytest.raises(ValueError) as e_info:
            new, spacing = self.dataset.reindex('k_cen', 0.005, force=True, return_spacing=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            weights = numpy.random.random(self.dataset.shape)
            self.dataset['weights'] = weights
            new, spacing = self.dataset.reindex('k_cen', 0.02, weights='weights', force=True, return_spacing=True)

            diff = numpy.diff(new.coords['k_cen'])
            self.assertTrue(numpy.alltrue(diff > numpy.diff(self.dataset.coords['k_cen'])[0]))
            
            with pytest.raises(ValueError) as e_info:
                new = self.dataset.reindex('mu_cen', 0.4, force=False)
            new = self.dataset.reindex('mu_cen', 0.4, force=True)
        
        
    def test_pickle(self):
        
        import pickle
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False) as ff:
            pickle.dump(self.dataset, ff)
            filename = ff.name
        
        new = pickle.load(open(filename, 'rb'))
        for var in self.dataset.variables:
            numpy.testing.assert_array_equal(self.dataset[var], new[var])
            
        if os.path.exists(filename):
            os.remove(filename)
            
        
        
        
            
        
        
            
            
        
        