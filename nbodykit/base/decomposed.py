from nbodykit.base.catalog import CatalogSource

class DecomposedCatalog(CatalogSource):
    """ A DomainDecomposedCatalog.

        Attributes
        ----------
        domain : :pyclass:`pmesh.domain.GridND`
        layout : A large object that holds which particle belongs to which rank.
        source : the original source object

        Parameters
        ----------
        freeze : list
            a list of columns to already exchange
    """
    def __init__(self, source, domain, position='Position', columns=[]):

        self.domain = domain
        self.source = source

        layout = domain.decompose(source[position].compute())

        self._size = layout.newlength

        CatalogSource.__init__(self, comm=source.comm)
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
