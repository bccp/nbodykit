from nbodykit.base.catalog import CatalogSource
from pmesh.domain import GridND
from nbodykit.utils import split_size_3d
import numpy

class SubVolumesCatalog(CatalogSource):
    """ A catalog that distributes the particles spatially into subvolumes per
        MPI rank.

        Attributes
        ----------
        domain : :class:`pmesh.domain.GridND`;
            The domain objects for decomposition. If None, generate
            a domain to decompose the catalog into a 3d grid.

        layout : A large object that holds which particle belongs to which rank.
        source : the original source object

        Parameters
        ----------
        columns: list
            a list of columns to already exchange
    """
    def __init__(self, source, domain=None, position='Position', columns=None):
        comm = source.comm

        if domain is None:
            # determine processor division for domain decomposition
            np = split_size_3d(comm.size)

            if comm.rank == 0:
                self.logger.info("using cpu grid decomposition: %s" %str(np))

            grid = [
                numpy.linspace(0, source.attrs['BoxSize'][0], np[0] + 1, endpoint=True),
                numpy.linspace(0, source.attrs['BoxSize'][1], np[1] + 1, endpoint=True),
                numpy.linspace(0, source.attrs['BoxSize'][2], np[2] + 1, endpoint=True),
            ]

            domain = GridND(grid, comm=comm)

        self.domain = domain
        self.source = source

        layout = domain.decompose(source[position].compute())

        self._size = layout.newlength

        CatalogSource.__init__(self, comm=comm)
        self.attrs.update(source.attrs)

        self._frozen = {}
        if columns is None: columns = source.columns

        for column in columns:
            data = source[column].compute()
            self._frozen[column] = self.make_column(layout.exchange(data))

    @property
    def hardcolumns(self):
        return sorted(list(self._frozen.keys()))

    def get_hardcolumn(self, col):
        return self._frozen[col]
