from nbodykit.base.catalog import CatalogSource
from nbodykit import CurrentMPIComm
import numpy

class ArrayCatalog(CatalogSource):
    """
    A catalog source initialized from a dictionary or structred ndarray object
    """
    @CurrentMPIComm.enable
    def __init__(self, data, comm=None, use_cache=False, **kwargs):
        """
        Parameters
        ----------
        data : dict or ndarray
            a dictionary or structured ndarray; items are interpreted
            as the columns of the catalog; the length of any item is used
            as the size of the catalogue.
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

        if hasattr(data, 'dtype'):
            keys = sorted(data.dtype.names)
        else:
            keys = sorted(data.keys())

        dtype = numpy.dtype([(key, (data[key].dtype, data[key].shape[1:])) for key in keys])

        # verify data types are the same
        dtypes = self.comm.gather(dtype, root=0)
        if self.comm.rank == 0:
            if any(dt != dtypes[0] for dt in dtypes):
                raise ValueError("mismatch between dtypes across ranks in Array")

        self._size = len(self._source[keys[0]])

        for key in keys:
            if len(self._source[key]) != self._size:
                raise ValueError("column `%s` and column `%s` has different size" % (keys[0], key))

        self._dtype = dtype
        # update the meta-data
        self.attrs.update(kwargs)

        CatalogSource.__init__(self, comm=comm, use_cache=use_cache)

    @property
    def size(self):
        return self._size
        
    @property
    def hardcolumns(self):
        """
        The union of the columns in the file and any transformed columns
        """
        defaults = CatalogSource.hardcolumns.fget(self)
        return list(self._dtype.names) + defaults

    def get_hardcolumn(self, col):
        """
        Return a column from the underlying file source
        
        Columns are returned as dask arrays
        """
        if col in self._dtype.names: 
            return self.make_column(self._source[col])
        else:
            return CatalogSource.get_hardcolumn(self, col)

