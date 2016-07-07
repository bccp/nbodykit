from . import unittest, pytest
import numpy
from nbodykit.utils.selectionlanguage import Query, SelectionError

class TestSelection(unittest.TestCase):
    """
    Unit tests for :class:`~nbodykit.utils.selectionlanguage.Query`
    """
    def test_nan(self):
                
        data = numpy.zeros(5, dtype=[('mass', 'f8')])
        data['mass'][0] = numpy.nan
        q = Query("mass is nan")
        m = q.get_mask(data)
        
        self.assertTrue(m.sum() == 1)
        self.assertTrue(m[0] == True)
        
    def test_not_nan(self):
                
        data = numpy.zeros(5, dtype=[('mass', 'f8')])
        data['mass'][0] = numpy.nan
        q = Query("mass is not nan")
        m = q.get_mask(data)
        
        self.assertTrue(m.sum() == len(data)-1)
        self.assertTrue(m[0] == False)
        
    def test_inf(self):
        
        data = numpy.zeros(5, dtype=[('mass', 'f8')])
        data['mass'][0] = numpy.inf
        q = Query("mass is inf")
        m = q.get_mask(data)
        
        self.assertTrue(m.sum() == 1)
        self.assertTrue(m[0] == True)
        
    def test_not_inf(self):
        
        data = numpy.zeros(5, dtype=[('mass', 'f8')])
        data['mass'][0] = numpy.inf
        q = Query("mass is not inf")
        m = q.get_mask(data)
        
        self.assertTrue(m.sum() == len(data)-1)
        self.assertTrue(m[0] == False)
        
    def test_eq(self):
        
        data = numpy.zeros(5, dtype=[('mass', 'int')])
        data['mass'][0] = 1
        q = Query("mass == 1")
        m = q.get_mask(data)
        
        self.assertTrue(m.sum() == 1)
        self.assertTrue(m[0] == True)
        
    def test_neq(self):
        
        data = numpy.zeros(5, dtype=[('mass', 'int')])
        data['mass'][0] = 1
        q = Query("mass != 0")
        m = q.get_mask(data)
        
        self.assertTrue(m.sum() == 1)
        self.assertTrue(m[0] == True)
        
    def test_is_not(self):
        
        data = numpy.zeros(5, dtype=[('mass', 'int')])
        data['mass'][0] = 1
        q = Query("mass is not 0")
        m = q.get_mask(data)
        
        self.assertTrue(m.sum() == 1)
        self.assertTrue(m[0] == True)
        
    def test_ge(self):
        
        data = numpy.zeros(5, dtype=[('mass', 'int')])
        data['mass'][0] = 1
        data['mass'][1] = 2
        q = Query("mass >= 1")
        m = q.get_mask(data)
        
        self.assertTrue(m.sum() == 2)
        self.assertTrue((m[0] == True) and (m[1] == True))
        
    def test_gt(self):
        
        data = numpy.zeros(5, dtype=[('mass', 'int')])
        data['mass'][0] = 1
        data['mass'][1] = 2
        q = Query("mass > 1")
        m = q.get_mask(data)
        
        self.assertTrue(m.sum() == 1)
        self.assertTrue(m[1] == True)
        
    def test_le(self):
        
        data = numpy.zeros(5, dtype=[('mass', 'int')])
        data['mass'][0] = 1
        data['mass'][1] = 2
        q = Query("mass <= 1")
        m = q.get_mask(data)
        
        self.assertTrue(m.sum() == len(data)-1)
        self.assertTrue(m[1]==False)
        
    def test_lt(self):
        
        data = numpy.zeros(5, dtype=[('mass', 'int')])
        data['mass'][0] = 1
        data['mass'][1] = 2
        q = Query("mass < 1")
        m = q.get_mask(data)
        
        self.assertTrue(m.sum() == len(data)-2)
        self.assertTrue((m[0]==False) and (m[1] == False))
        
    def test_compound_and(self):
        
        data = numpy.zeros(100, dtype=[('mass', float), ('velocity', float)])
        data['mass'] = numpy.random.random(size=100)
        data['velocity'] = numpy.random.random(size=100)
        
        q = Query("mass < 0.5 and velocity > 0.5")
        sliced = data[q.get_mask(data)]
        self.assertTrue(numpy.alltrue(sliced['mass'] < 0.5))
        self.assertTrue(numpy.alltrue(sliced['velocity'] > 0.5))
        
        q = Query("(mass < 0.5) and (velocity > 0.5)")
        sliced = data[q.get_mask(data)]
        self.assertTrue(numpy.alltrue(sliced['mass'] < 0.5))
        self.assertTrue(numpy.alltrue(sliced['velocity'] > 0.5))
        
    def test_compound_or(self):
        
        data = numpy.zeros(100, dtype=[('mass', int), ('velocity', int)])
        data['mass'] = numpy.random.choice([0, 1, 2], size=100)
        data['velocity'] = numpy.random.choice([0, 1, 2], size=100)
        ans = (data['mass'] == 0)|(data['velocity'] != 1)
        
        q = Query("mass == 0 or velocity != 1")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
        q = Query("(mass == 0) or (velocity != 1)")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
    def test_compound_andor(self):
        
        data = numpy.zeros(100, dtype=[('mass', int), ('velocity', int)])
        data['mass'] = numpy.random.choice([0, 1, 2], size=100)
        data['velocity'] = numpy.random.choice([0, 1, 2], size=100)
        ans = (data['mass']==0)|((data['velocity'] != 1)&(data['mass'] < 1))
        
        q = Query("mass == 0 or velocity != 1 and mass < 1")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
        q = Query("(mass == 0) or (velocity != 1 and mass < 1)")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
    def test_not(self):
        
        data = numpy.zeros(100, dtype=[('mass', int), ('velocity', int)])
        data['mass'] = numpy.random.choice([0, 1, 2], size=100)
        ans = data['mass'] >= 1
        
        q = Query("not mass < 1")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
        q = Query("not (mass < 1)")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
    def test_compound_andnot(self):
        
        data = numpy.zeros(100, dtype=[('mass', int), ('velocity', int)])
        data['mass'] = numpy.random.choice([0, 1, 2], size=100)
        data['velocity'] = numpy.random.choice([0, 1, 2], size=100)
        ans = (data['mass']==0)&((data['velocity'] == 1))
        
        q = Query("mass == 0 and not velocity != 1")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
          
    def test_index(self):
        
        data = numpy.zeros(100, dtype=[('Position', (float,3))])
        data['Position'] = numpy.random.random(size=(100,3))
        ans = (data['Position'][:,0]<0.5)&(data['Position'][:,-1] >= 0.7)
        
        q = Query("Position[:,0] < 0.5 and Position[:,-1] >= 0.7")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
    def test_compound_index(self):
        
        data = numpy.zeros(100, dtype=[('Position', (float,3)), ('velocity', int)])
        data['Position'] = numpy.random.random(size=(100,3))
        data['velocity'] = numpy.random.choice([0, 1, 2], size=100)
        ans = (data['Position'][:,1]<0.5)|(data['velocity']==2)
        
        q = Query("Position[:,1] < 0.5 or velocity == 2")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
    
    def test_exp(self):
        
        data = numpy.zeros(100, dtype=[('mass', float)])
        data['mass'] = numpy.random.random(size=100)
        ans = (data['mass'] < 0.25)
        
        q = Query("mass < 0.5**2")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
    def test_mult(self):
        
        data = numpy.zeros(100, dtype=[('mass', float)])
        data['mass'] = numpy.random.random(size=100)
        ans = (data['mass'] < 0.25)
        
        q = Query("mass < 0.125*2")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
    def test_div(self):
        
        data = numpy.zeros(100, dtype=[('mass', float)])
        data['mass'] = numpy.random.random(size=100)
        ans = (data['mass'] < 0.25)
        
        q = Query("mass < 0.5/2.")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
    def test_plus(self):
        
        data = numpy.zeros(100, dtype=[('mass', float)])
        data['mass'] = numpy.random.random(size=100)
        ans = (data['mass'] < 0.25)
        
        q = Query("mass < 0.125+0.125")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
    def test_minus(self):
        
        data = numpy.zeros(100, dtype=[('mass', float)])
        data['mass'] = numpy.random.random(size=100)
        ans = (data['mass'] < 0.25)
        
        q = Query("mass < 0.5-0.125-0.125")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
    def test_compound_log(self):
        
        data = numpy.zeros(100, dtype=[('mass', float)])
        data['mass'] = numpy.random.random(size=100)
        ans = (data['mass'] < 0.5*numpy.log(2.))
        
        q = Query("mass < 0.5*log(2)")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))

    def test_log(self):
        
        data = numpy.zeros(100, dtype=[('mass', float)])
        data['mass'] = numpy.random.random(size=100)
        ans = (data['mass'] < numpy.log(2.))
        
        q = Query("mass < log(2)")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
    def test_log10(self):
        
        data = numpy.zeros(100, dtype=[('mass', float)])
        data['mass'] = numpy.random.random(size=100)
        ans = (data['mass'] < numpy.log10(2.))
        
        q = Query("mass < log10(2)")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
    def test_exp(self):
        
        data = numpy.zeros(100, dtype=[('mass', float)])
        data['mass'] = numpy.random.random(size=100)
        ans = (data['mass'] < numpy.exp(-0.9))
        
        q = Query("mass < exp(-0.9))")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
            
    def test_slice_index_fail(self):
        
        data = numpy.zeros(100, dtype=[('Position', (float,3))])
        data['Position'] = numpy.random.random(size=(100,3))
        
        # output index must be 1D or same dimension as input
        with pytest.raises(SelectionError):
            q = Query("Position[:,:2] < 0.5")
            mask = q.get_mask(data)
             
    def test_missing_column(self):
        
        data = numpy.zeros(5, dtype=[('mass', 'f8')])
        data['mass'][0] = numpy.nan
        
        with pytest.raises(SelectionError):
            q = Query("velocity is nan")
            m = q.get_mask(data)
            
    def test_indexing_single_dimension(self):
        
        data = numpy.zeros(5, dtype=[('mass', 'f8')])
        data['mass'][0] = numpy.nan
        
        with pytest.raises(SelectionError):
            q = Query("mass[:,0] is nan")
            m = q.get_mask(data)

    def test_indexing_out_of_range(self):
        
        data = numpy.zeros(10, dtype=[('Position', (float,3))])
        data['Position'] = numpy.random.random(size=(10,3))
        
        with pytest.raises(SelectionError):
            q = Query("Position[:,4] is nan")
            m = q.get_mask(data)
            
    def test_eval_fail(self):
        
        data = numpy.zeros(5, dtype=[('mass', 'f8')])
        data['mass'][0] = numpy.nan
        
        with pytest.raises(SelectionError):
            q = Query("mass < missing_func(0.3)")
            m = q.get_mask(data)
            
    def test_missing_op(self):
        
        data = numpy.zeros(5, dtype=[('mass', 'f8')])
        data['mass'][0] = numpy.nan
        
        with pytest.raises(SelectionError):
            q = Query("mass isnot 4)")
            m = q.get_mask(data)
            
    def test_column_log(self):
        
        data = numpy.zeros(100, dtype=[('mass', float)])
        data['mass'] = numpy.random.random(size=100)
        ans = numpy.log(data['mass']) > -0.5
        
        q = Query("log(mass) > -0.5")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))
        
    def test_column_log10(self):
        
        data = numpy.zeros(100, dtype=[('mass', float)])
        data['mass'] = numpy.random.random(size=100)
        ans = numpy.log10(data['mass']) > -0.5
        
        q = Query("log10(mass) > -0.5")
        mask = q.get_mask(data)
        self.assertTrue(numpy.all(ans == mask))      
    

    
    
        

        
        
        
        
            
        
        
            
            
        
        