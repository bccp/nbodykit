from nbodykit.base.catalog import CatalogSourceBase, column
from nbodykit.utils import attrs_to_dict

import logging
import functools

def OnDemandColumn(source, col):
    """
    Return a column from the source on-demand.
    """
    return source[col]

class MultipleSpeciesCatalog(CatalogSourceBase):
    """
    A CatalogSource interface for handling multiples species
    of particles.

    Parameters
    ----------
    names : list of str
        list of strings specifying the names of the various species;
        data columns are prefixed with "species/" where "species" is
        in ``names``
    *species : two or more CatalogSource objects
        catalogs to be combined into a single catalog, which give the
        data for different species of particles; as many catalogs
        as names must be provided
    use_cache : bool, optional
        whether to cache data when reading; default is ``True``
    """
    logger = logging.getLogger('MultipleSpeciesCatalog')

    def __repr__(self):
        return "MultipleSpeciesCatalog(species=%s)" %self.attrs['species']

    def __init__(self, names, *species, **kwargs):

        # whether to use the cache
        use_cache = kwargs.get('use_cache', True)

        # input checks
        if len(species) < 2:
            raise ValueError("need at least 2 particle species to initialize MultipleSpeciesCatalog")
        if not all(cat.comm is species[0].comm for cat in species):
            raise ValueError("communicator mismatch in MultipleSpeciesCatalog")
        if len(names) != len(species):
            raise ValueError("a name must be provided for each species catalog provided")

        self.comm = species[0].comm
        self._sources = species
        self.species = names
        self.attrs['species'] = names

        # update the dictionary with data/randoms attrs
        for cat, name in zip(species, names):
            self.attrs.update(attrs_to_dict(cat, name + '.'))

        self.size = NotImplemented

        # init the base class
        CatalogSourceBase.__init__(self, self.comm, use_cache=use_cache)

        # turn on cache?
        if self.use_cache:
            for cat in species: cat.use_cache = True

        # prefixed columns in this source return on-demand from their
        # respective source objects
        for name, source in zip(names, self._sources):
            for col in source.columns:
                f = functools.partial(OnDemandColumn, col=col, source=source)
                self._overrides[name+'/'+col] = f

    def __setitem__(self, col, value):
        """
        Add columns to any of the species catalogs.

        .. note::
            New column names should be prefixed by 'species/' where
            'species' is a name in the :attr:`species` attribute.
        """
        fields = col.split('/')
        if len(fields) != 2:
            msg = "new column names should be prefixed by 'species/' where "
            msg += "'species' is one of %s" % str(self.species)
            raise ValueError(msg)

        if fields[0] not in self.species:
            args = (fields[0], str(self.species))
            raise ValueError("species '%s' is not valid; should be one of %s" %args)

        return CatalogSourceBase.__setitem__(self, col, value)

    def to_mesh(self, Nmesh, BoxSize, dtype='f4', interlaced=False,
                compensated=False, window='cic', weight='Weight',
                selection='Selection', value='Value', position='Position'):
        """
        Convert the catalog to a mesh, which knows how to "paint" the
        the combined density field, summed over all particle species.

        Parameters
        ----------
        Nmesh : int, 3-vector
            the number of cells per box side
        BoxSize : float, 3-vector
            the size of the box
        dtype : str, dtype, optional
            the data type of the mesh when painting
        interlaced : bool, optional
            whether to use interlacing to reduce aliasing when painting the
            particles on the mesh
        compensated : bool, optional
            whether to apply a Fourier-space transfer function to account for
            the effects of the gridding + aliasing
        window : str, optional
            the string name of the window to use when interpolating the
        weight : str, optional
            the name of the column specifying the weight for each particle
        value: str, optional
            the name of the column specifying the field value for each particle
        selection : str, optional
            the name of the column that specifies which (if any) slice
            of the CatalogSource to take
        position : str, optional
            the name of the column that specifies the position data of the
            objects in the catalog
        """
        from nbodykit.source.catalogmesh import MultipleSpeciesCatalogMesh

        # verify that all of the required columns exist
        for name in self.species:
            for col in [position, selection, weight, value]:
                _col = name+'/'+col
                if _col not in self:
                    raise ValueError("the '%s' species is missing the '%s' column" %(name, col))

        # initialize the mesh
        kws = {'Nmesh':Nmesh, 'BoxSize':BoxSize, 'dtype':dtype, 'selection':selection}
        mesh = MultipleSpeciesCatalogMesh(self,
                                          Nmesh=Nmesh,
                                          BoxSize=BoxSize,
                                          dtype=dtype,
                                          selection=selection,
                                          position=position,
                                          weight=weight,
                                          value=value)
        mesh.interlaced = interlaced
        mesh.compensated = compensated
        mesh.window = window

        return mesh
