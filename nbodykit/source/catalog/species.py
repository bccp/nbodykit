from nbodykit.base.catalog import CatalogSourceBase
from nbodykit.utils import attrs_to_dict

import numpy
import logging
from six import string_types


class MultipleSpeciesCatalog(CatalogSourceBase):
    """
    A CatalogSource interface for handling multiples species
    of particles.

    This CatalogSource stores a copy of the original CatalogSource objects
    for each species, providing access to the columns via the format
    ``species/`` where "species" is one of the species names provided.

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

    Examples
    --------

    Initialization:

    >>> data = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42)
    >>> randoms = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84)
    >>> cat = MultipleSpeciesCatalog(['data', 'randoms'], data, randoms)

    Accessing the Catalogs for individual species:

    >>> data = cat["data"] # a copy of the original "data" object

    Accessing individual columns:

    >>> data_pos = cat["data/Position"]

    Setting new columns:

    >>> cat["data"]["new_column"] = 1.0
    >>> assert "data/new_column" in cat

    """
    logger = logging.getLogger('MultipleSpeciesCatalog')

    def __repr__(self):
        return "MultipleSpeciesCatalog(species=%s)" %str(self.attrs['species'])

    def __init__(self, names, *species, **kwargs):

        # whether to use the cache
        self.use_cache = kwargs.pop('use_cache', True)

        # input checks
        if len(species) < 2:
            raise ValueError("need at least 2 particle species to initialize MultipleSpeciesCatalog")
        if len(set(names)) != len(names):
            raise ValueError("each species must have a unique name")
        if not all(cat.comm is species[0].comm for cat in species):
            raise ValueError("communicator mismatch in MultipleSpeciesCatalog")
        if len(names) != len(species):
            raise ValueError("a name must be provided for each species catalog provided")

        self.comm = species[0].comm
        self.attrs['species'] = names

        # update the dictionary with data/randoms attrs
        for cat, name in zip(species, names):
            self.attrs.update(attrs_to_dict(cat, name + '.'))

        # update the rest of meta-data
        self.attrs.update(kwargs)

        # no size!
        self.size = NotImplemented

        # store copies of the original input catalogs as (name:catalog) dict
        self._sources = {name:cat.copy() for name,cat in zip(names, species)}

        # turn on cache?
        if self.use_cache:
            for name in names:
                self._sources[name].use_cache = True

    @property
    def species(self):
        """
        List of species names
        """
        return self.attrs['species']

    @property
    def hardcolumns(self):
        """
        Columns for individual species can be accessed using a ``species/``
        prefix and the column name, i.e., ``data/Position``.
        """
        return ['%s/%s' %(species, col)
                for species in self.species
                for col in self._sources[species].columns]

    def get_hardcolumn(self, col):
        """
        Hard columns are accessed via the underlying CatalogSource object
        for each species.

        Here, it is assumed that ``col`` is of the form ``species/column``.
        """
        species, name = col.split('/')
        return self._sources[species][name]

    def __getitem__(self, key):
        """
        This provides access to the underlying data in two ways:

        - The CatalogSource object for a species can be accessed if ``key``
          is a species name.
        - Individual columns for a species can be accessed using the
          format: ``species/column``.
        """
        # return a new CatalogSource holding only the specific species
        if isinstance(key, string_types) and key in self.species:
            return self._sources[key]

        return CatalogSourceBase.__getitem__(self, key)

    def __setitem__(self, col, value):
        """
        This class is read-only. To add columns for an individual species,
        use the following syntax: ``cat[species][column] = data``.

        Here, ``species`` is the name of the species, and ``column`` is the
        column name.
        """
        msg = "%s does not support item assignment;" % self.__class__.__name__
        msg += " to add columns for an individual species, use ``cat[species][column] = data``"
        raise ValueError(msg)

    def to_mesh(self, Nmesh=None, BoxSize=None, dtype='f4', interlaced=False,
                compensated=False, window='cic', weight='Weight',
                selection='Selection', value='Value', position='Position'):
        """
        Convert the catalog to a mesh, which knows how to "paint" the
        the combined density field, summed over all particle species.

        Parameters
        ----------
        Nmesh : int, 3-vector, optional
            the number of cells per box side; can be inferred from ``attrs``
            if the value is the same for all species
        BoxSize : float, 3-vector, optional
            the size of the box; can be inferred from ``attrs``
            if the value is the same for all species
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
                if col not in self[name]:
                    raise ValueError("the '%s' species is missing the '%s' column" %(name, col))

        # try to find BoxSize and Nmesh
        if BoxSize is None:
            BoxSize = check_species_metadata('BoxSize', self.attrs, self.species)
        if Nmesh is None:
            Nmesh = check_species_metadata('Nmesh', self.attrs, self.species)

        # return the mesh
        return MultipleSpeciesCatalogMesh(self,
                                          Nmesh=Nmesh,
                                          BoxSize=BoxSize,
                                          dtype=dtype,
                                          selection=selection,
                                          position=position,
                                          weight=weight,
                                          value=value,
                                          interlaced=interlaced,
                                          compensated=compensated,
                                          window=window)


def check_species_metadata(name, attrs, species):
    """
    Check to see if there is a single value for
    ``name`` in the meta-data of all the species
    """
    _vals = []

    if name in attrs:
        _vals.append(attrs[name])

    for s in species:
        if s + '.' + name in attrs:
            _vals.append(attrs.get(s + '.' + name ))

    if len(_vals) == 0:
        raise ValueError("please specify ``%s`` attributes" % name)

    if not all(numpy.equal(_vals[0], i).all() for i in _vals):
        raise ValueError("please specify ``%s`` attributes that are consistent for each species and for the multi species catalog; " % name)

    return _vals[0]
