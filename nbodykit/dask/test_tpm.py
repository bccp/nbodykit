from binary_file import BinaryFile
from fileiter import FileIterator
from dask.delayed import delayed
from dask import compute

import numpy

class DaskTPMSnapshot(object):
    """
    DataSource to read snapshot files from Martin White's TPM simulations
    """
    plugin_name = "TPMSnapshot"
    columns = ['Position', 'Velocity']
    
    def __init__(self, path, BoxSize, rsd=None, bunchsize=4*1024*1024):
        
        self.path = path
        self.BoxSize = BoxSize
        self.rsd = rsd
        self.bunchsize = bunchsize
        
        dtypes = [('Position', ('f4', 3)), ('Velocity', ('f4', 3)), ('ID', 'u8')]
        self.storage = BinaryFile(self.path, dtypes, header_size=28)
    
    @classmethod
    def register(cls):
        
        s = cls.schema
        s.description = "read snapshot files from Martin White's TPM"
        s.add_argument("path", type=str, help="the file path to load the data from")
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        s.add_argument("rsd", choices="xyz", help="direction to do redshift distortion")
        s.add_argument("bunchsize", type=int, help="number of particles to read per rank in a bunch")

    
    def Position(self, data):
        """
        The "Position" column
        """
        pos = data['Position'] * self.BoxSize
        
        if self.rsd is not None:
            dir = "xyz".index(self.rsd)
            pos[:, dir] += self.Velocity(data)[:, dir]
            pos[:, dir] %= self.BoxSize[dir]
            
        return pos
        
    def Velocity(self, data):
        """
        The "Velocity" column
        """
        return data['Velocity'] * self.BoxSize
        
    
    def read(self, columns, full=False):
        """ 
        Return the columns of this DataSource, by executing them
        as dask-delayed functions
        """
        if self.rsd is not None and 'Velocity' not in columns:
            columns.append('Velocity')
        
        # these are the column functions, delayed so they can execute together
        tasks = [delayed(getattr(self, col)) for col in columns]
        
        # loop over chunks of the file storage
        for data in FileIterator(self.storage, columns, chunksize=self.bunchsize, comm=self.comm):
            
            # the dask compute function is pretty smart --> execute tasks together and it optimizes
            # this optionally tasks a cache object
            yield compute(*[task(data) for task in tasks])

