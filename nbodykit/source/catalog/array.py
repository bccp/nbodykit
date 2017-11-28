from nbodykit.base.catalog import CatalogSource
from nbodykit.utils import is_structured_array
from nbodykit import CurrentMPIComm
from astropy.table import Table
import numpy

class ArrayCatalog(CatalogSource):
    """
    A CatalogSource initialized from an in-memory :obj:`dict`,
    structured :class:`numpy.ndarray`, or :class:`astropy.table.Table`.

    Parameters
    ----------
    data : obj:`dict`, :class:`numpy.ndarray`, :class:`astropy.table.Table`
        a dictionary, structured ndarray, or astropy Table; items are
        interpreted as the columns of the catalog; the length of any item is
        used as the size of the catalog.
    comm : MPI Communicator, optional
        the MPI communicator instance; default (``None``) sets to the
        current communicator
    **kwargs :
        additional keywords to store as meta-data in :attr:`attrs`
    """
    @CurrentMPIComm.enable
    def __init__(self, data, comm=None, **kwargs):

        # convert astropy Tables to structured numpy arrays
        if isinstance(data, Table):
            data = data.as_array()

        # check for structured data
        if not isinstance(data, dict):
            if not is_structured_array(data):
                raise ValueError(("input data to ArrayCatalog must have a "
                                   "structured data type with fields"))

        self.comm    = comm
        self._source = data

        # compute the data type
        if hasattr(data, 'dtype'):
            keys = sorted(data.dtype.names)
        else:
            keys = sorted(data.keys())
        dtype = numpy.dtype([(key, (data[key].dtype, data[key].shape[1:])) for key in keys])
        self._dtype = dtype

        # verify data types are the same
        dtypes = self.comm.gather(dtype, root=0)
        if self.comm.rank == 0:
            if any(dt != dtypes[0] for dt in dtypes):
                raise ValueError("mismatch between dtypes across ranks in Array")

        # the local size
        self._size = len(self._source[keys[0]])

        for key in keys:
            if len(self._source[key]) != self._size:
                raise ValueError("column `%s` and column `%s` has different size" % (keys[0], key))

        # update the meta-data
        self.attrs.update(kwargs)

        CatalogSource.__init__(self, comm=comm)

    @property
    def hardcolumns(self):
        """
        The union of the columns in the file and any transformed columns.
        """
        defaults = CatalogSource.hardcolumns.fget(self)
        return list(self._dtype.names) + defaults

    def get_hardcolumn(self, col):
        """
        Return a column from the underlying data array/dict.

        Columns are returned as dask arrays.
        """
        if col in self._dtype.names:
            return self.make_column(self._source[col])
        else:
            return CatalogSource.get_hardcolumn(self, col)
