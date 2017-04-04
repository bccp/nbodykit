from nbodykit.base.catalog import CatalogSource
from nbodykit import CurrentMPIComm
import numpy

class ArrayCatalog(CatalogSource):
    """
    A catalog source initialized from a :mod:`numpy` array
    """
    @CurrentMPIComm.enable
    def __init__(self, data, comm=None, use_cache=False, **kwargs):
        """
        Parameters
        ----------
        data : numpy.array
            a structured numpy array; fields of the array are interpreted
            as the columns of the catalog
        comm : MPI Communicator, optional
            the MPI communicator instance; default (``None``) sets to the
            current communicator  
        use_cache : bool, optional
            whether to cache data read from disk; default is ``False``
        **kwargs : 
            additional keywords to store as meta-data in :attr:`attrs`
        """
        self.comm    = comm
        self._source = data
        if data.dtype.names is None:
            raise ValueError("input data should be a structured numpy array")
        
        # verify data types are the same
        dtypes = self.comm.gather(data.dtype, root=0)
        if self.comm.rank == 0:
            if any(dt != dtypes[0] for dt in dtypes):
                raise ValueError("mismatch between dtypes across ranks in Array")
        
        CatalogSource.__init__(self, comm=comm, use_cache=use_cache)
        
        # update the meta-data
        self.attrs.update(kwargs)

    @property
    def size(self):
        return len(self._source)
        
    @property
    def hardcolumns(self):
        """
        The union of the columns in the file and any transformed columns
        """
        defaults = CatalogSource.hardcolumns.fget(self)
        return list(self._source.dtype.names) + defaults

    def get_hardcolumn(self, col):
        """
        Return a column from the underlying file source
        
        Columns are returned as dask arrays
        """
        if col in self._source.dtype.names: 
            return self.make_column(self._source[col])
        else:
            return CatalogSource.get_hardcolumn(self, col)

