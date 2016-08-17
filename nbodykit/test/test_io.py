from nbodykit import plugin_manager
from . import unittest
from .utils.iobase import IOTestBase        

import numpy
import tempfile
import os

class TestCSVFile(IOTestBase, unittest.TestCase):
    """
    Test the CSVFile plugin 
    """
    def make_data(self):
        
        names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        dtype = [(col, 'f4') for col in names]

        N = 1000
        data = numpy.empty(N, dtype=dtype)
        for col in names:
            data[col] = numpy.random.random(size=N)
            
        return data
        
    def write_to_disk(self):
        
        with tempfile.NamedTemporaryFile(delete=False, mode='wb') as ff:
            filename = ff.name
            numpy.savetxt(ff, self.data)
            
        return filename
        
    def load_data(self):
        
        CSVFile = plugin_manager.get_plugin('CSVFile')
        names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        return CSVFile(self.path, names, dtype='f4', delim_whitespace=True, header=None, blocksize=5000)
        
class TestBinaryFile(IOTestBase, unittest.TestCase):
    """
    Test the BinaryFile plugin 
    """
    def make_data(self):
        
        self.dtype = [('Position', ('f4', 3)), ('Velocity', ('f4', 3))]
        N = 1000
        data = numpy.empty(N, dtype=self.dtype)
        for col in ['Position', 'Velocity']:
            data[col] = numpy.random.random(size=data[col].shape)
            
        return data
        
    def write_to_disk(self):
        
        # open the temporary file (don't delete it yet)
        with tempfile.NamedTemporaryFile(delete=False, mode='wb') as ff:
            
            # store the filename
            filename = ff.name
        
            # write a fake header
            header = numpy.ones(10, dtype='i4')
            self.header_size = header.nbytes
            ff.write(header.tobytes())
            
            # write each column, consecutively
            for col in ['Position', 'Velocity']:
                ff.write(self.data[col].tobytes())
                
        return filename
        
    def load_data(self):
        
        BinaryFile = plugin_manager.get_plugin('BinaryFile')
        return BinaryFile(self.path, self.dtype, header_size=self.header_size)
        
class TestFileStack(IOTestBase, unittest.TestCase):
    """
    Test the FileStack plugin 
    """
    def make_data(self):
        
        self.dtype = [('Position', ('f4', 3)), ('Velocity', ('f4', 3))]
        N = 1000
        data = numpy.empty(N, dtype=self.dtype)
        for col in ['Position', 'Velocity']:
            data[col] = numpy.random.random(size=data[col].shape)
            
        return data
        
    def write_to_disk(self):
        
        # open the temporary file
        with tempfile.NamedTemporaryFile(delete=False, mode='wb') as ff:

            # store the filename
            filename = ff.name
        
            # write a fake header
            header = numpy.ones(10, dtype='i4')
            self.header_size = header.nbytes
            ff.write(header.tobytes())
            
            # write each column, consecutively
            for col in ['Position', 'Velocity']:
                ff.write(self.data[col].tobytes())
        
        return filename
        
    def load_data(self):
        
        FileStack = plugin_manager.get_plugin('FileStack')
        BinaryFile = plugin_manager.get_plugin('BinaryFile')
        
        # read the same data twice, stacked together
        path = [self.path]*2
        ff = FileStack(path, BinaryFile, dtype=self.dtype, header_size=self.header_size)
        
        # also stack the data
        self.data = numpy.concatenate([self.data, self.data], axis=0)
        
        return ff
        
        
        
    
        