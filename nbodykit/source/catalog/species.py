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

        # input checks
        if len(species) < 2:
            raise ValueError("need at least 2 particle species to initialize MultipleSpeciesCatalog")
        if len(set(names)) != len(names):
            raise ValueError("each species must have a unique name")
        if not all(cat.comm is species[0].comm for cat in species):
            raise ValueError("communicator mismatch in MultipleSpeciesCatalog")
        if len(names) != len(species):
            raise ValueError("a name must be provided for each species catalog provided")

        CatalogSourceBase.__init__(self, species[0].comm)

        self.attrs['species'] = names

        # update the dictionary with data/randoms attrs
        for cat, name in zip(species, names):
            self.attrs.update(attrs_to_dict(cat, name + '.'))

        # update the rest of meta-data
        self.attrs.update(kwargs)

        # no size!
        self.size = NotImplemented
        self.csize = NotImplemented

        # store copies of the original input catalogs as (name:catalog) dict
        self._sources = {name:cat.copy() for name,cat in zip(names, species)}

    @property
    def species(self):
        """
        List of species names
        """
        return self.attrs['species']

    @property
    def columns(self):
        """
        Columns for individual species can be accessed using a ``species/``
        prefix and the column name, i.e., ``data/Position``.
        """
        return ['%s/%s' %(species, col)
                for species in self.species
                for col in self._sources[species].columns]

    @property
    def hardcolumns(self):
        """
        Hardcolumn of the form ``species/name``
        """
        return ['%s/%s' %(species, col)
                for species in self.species
                for col in self._sources[species].hardcolumns]

    def __getitem__(self, key):
        """
        This provides access to the underlying data in two ways:

        - The CatalogSource object for a species can be accessed if ``key``
          is a species name.
        - Individual columns for a species can be accessed using the
          format: ``species/column``.
        """
        # return a new CatalogSource holding only the specific species
        if isinstance(key, string_types):
            if key in self.species:
                return self._sources[key]

            species, subcol = split_column(key, self.species)
            return CatalogSourceBase.__getitem__(self._sources[species], subcol)

        return CatalogSourceBase.__getitem__(self, key)

    def __setitem__(self, col, value):
        """
        Add columns to any of the species catalogs.

        .. note::
            New column names should be prefixed by 'species/' where
            'species' is a name in the :attr:`species` attribute.
        """
        species, subcol = split_column(col, self.species)

        # check size
        size = len(self._sources[species])
        if not numpy.isscalar(value):
            if len(value) != size:
                args = (col, size, len(value))
                raise ValueError("error setting '%s' column, data must be array of size %d, not %d" % args)

        # add the column to the CatalogSource in "_sources"
        return CatalogSourceBase.__setitem__(self._sources[species], subcol, value)

    def __delitem__(self, col):
        """
        Delete a column of the form ``species/column``
        """
        species, subcol = split_column(col, self.species)
        return CatalogSourceBase.__delitem__(self._sources[species], subcol)


    def to_mesh(self, Nmesh=None, BoxSize=None, dtype='f4', interlaced=False,
                compensated=False, resampler='cic', weight='Weight',
                value='Value', selection='Selection', position='Position', window=None):
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
        resampler: str, optional
            the string name of the resampler to use when interpolating the
        weight : str, optional
            the name of the column specifying the weight for each particle
        selection : str, optional
            the name of the column that specifies which (if any) slice
            of the CatalogSource to take
        value: str, optional
            the name of the column specifying the field value for each particle
        position : str, optional
            the name of the column that specifies the position data of the
            objects in the catalog
        window : str, optional
            the string name of the window to use when interpolating  (deprecated, use resampler)
        """
        from nbodykit.source.mesh.species import MultipleSpeciesCatalogMesh

        if window is not None:
            raise RuntimeError("use resampler instead")

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
                                          resampler=resampler)


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

def split_column(col, species):
    """
    Split the column name of the form 'species/name'
    """
    fields = col.split('/')
    if len(fields) != 2:
        msg = "new column names should be prefixed by 'species/' where "
        msg += "'species' is one of %s" % str(species)
        raise ValueError(msg)

    this_species, subcol = fields
    if this_species not in species:
        args = (this_species, str(species))
        raise ValueError("species '%s' is not valid; should be one of %s" %args)
    return this_species, subcol
