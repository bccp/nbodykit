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
    def __init__(self, source, domain, position='Position', freeze=[]):

        self.domain = domain
        self.source = source
        self.position = position
        self.layout = domain.decompose(source[position].compute())

        self._size = self.layout.newlength

        CatalogSource.__init__(self, comm=source.comm)
        self.attrs.update(source.attrs)
        self._frozen = {}

        self.freeze(freeze)

    def freeze(self, columns):
        for column in columns:
            if column not in self._frozen:
                self._frozen[column] = self.get_hardcolumn(column)

    @property
    def hardcolumns(self):
        return self.source.columns

    def get_hardcolumn(self, col):
        if col not in self._frozen:
            data = self.source[col].compute()
            return self.make_column(self.layout.exchange(data))
        else:
            return self.make_column(self._frozen[col])
