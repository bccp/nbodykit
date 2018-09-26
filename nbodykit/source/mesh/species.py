from nbodykit.base.mesh import MeshSource
from nbodykit import _global_options
import numpy
import logging
import warnings

# for converting from particle to mesh
from pmesh.pm import RealField

from nbodykit.utils import attrs_to_dict
class MultipleSpeciesCatalogMesh(MeshSource):
    """
    A subclass of :class:`~nbodykit.base.catalogmesh.CatalogMesh`
    designed to paint the density field from a sum of multiple types
    of particles.

    The :func:`compute` function paints the density field summed over
    all particle species.

    Parameters
    ----------
    source : CatalogSource
        the input catalog that we wish to interpolate to a mesh
    BoxSize :
        the size of the box
    Nmesh : int, 3-vector
        the number of cells per mesh side
    dtype : str
        the data type of the values stored on mesh
    weight : str
        column in ``source`` that specifies the weight value for each
        particle in the ``source`` to use when gridding
    value : str
        column in ``source`` that specifies the field value for each particle;
        the mesh stores a weighted average of this column
    selection : str
        column in ``source`` that selects the subset of particles to grid
    position : str, optional
        column in ``source`` specifying the position coordinates; default
        is ``Position``
    """
    logger = logging.getLogger('MultipleSpeciesCatalogMesh')

    def __init__(self, source, Nmesh, BoxSize, dtype,
            selection, position, weight, value,
            interlaced, compensated, resampler):

        from nbodykit.source.catalog import MultipleSpeciesCatalog

        if not isinstance(source, MultipleSpeciesCatalog):
            raise TypeError(("the input source for MultipleSpeciesCatalogMesh "
                             "must be a MultipleSpeciesCatalog"))

        MeshSource.__init__(self, Nmesh=Nmesh, BoxSize=BoxSize, dtype=dtype, comm=source.comm)

        self.source = source
        self.weight = weight
        self.position = position
        self.value = value
        self.selection = selection
        self.interlaced = interlaced
        self.compensated = compensated
        self.resampler = resampler
        self.dtype = dtype

        self.species = source.species

    def __iter__(self):
        return iter(self.species)

    def __getitem__(self, key):
        """
        If indexed by a species name, return a CatalogMesh object holding
        only the data columns for that species with the same parameters as
        the current object.

        If not a species name, this has the same behavior as
        :func:`CatalogSource.__getitem__`.
        """
        from nbodykit.source.mesh import CatalogMesh

        if key in self.source.species:

            cat = self.source[key]

            # view as a catalog mesh
            mesh = CatalogMesh(cat,
                    BoxSize=self.attrs['BoxSize'],
                    Nmesh=self.attrs['Nmesh'],
                    dtype=self.dtype,
                    Weight=cat[self.weight],
                    Value=cat[self.value],
                    Selection=cat[self.selection],
                    Position=cat[self.position],
                    interlaced=self.interlaced,
                    compensated=self.compensated,
                    resampler=self.resampler,
                )

            # attach attributes from self
            return mesh.__finalize__(self)
        else:
            raise KeyError("%s is not a species defined in the source" % key)

    def to_real_field(self, normalize=True):
        r"""
        Paint the density field holding the sum of all particle species,
        returning a :class:`~pmesh.pm.RealField` object.

        Meta-data computed for each particle is stored in the :attr:`attrs`
        attribute of the returned RealField, with keys that are prefixed by
        the species name. In particular, the total shot noise for the
        mesh is defined as:

        .. math::

            P_\mathrm{shot} = \sum_i (W_i/W_\mathrm{tot})^2 P_{\mathrm{shot},i},

        where the sum is over all species in the catalog, ``W_i`` is the
        sum of the ``Weight`` column for the :math:`i^\mathrm{th}` species,
        and :math:`W_\mathrm{tot}` is the sum of :math:`W_i` across all species.

        Parameters
        ----------
        normalize : bool, optional
            if ``True``, normalize the density field as :math:`1+\delta`,
            dividing by the total mean number of objects per cell, as given
            by the ``num_per_cell`` meta-data value in :attr:`attrs`

        Returns
        -------
        RealField :
            the RealField holding the painted density field, with a
            :attr:`attrs` dictionary attribute holding the meta-data
        """
        # track the sum of the mean number of objects per cell across species
        attrs = {'num_per_cell':0., 'N':0}

        # initialize an empty real field
        real = self.pm.create(mode='real', value=0)

        # loop over each species
        for name in self.source.species:

            if self.pm.comm.rank == 0:
                self.logger.info("painting the '%s' species" %name)

            # get a CatalogMesh for this species
            species_mesh = self[name]

            # paint (in-place) the un-normalized density field for this species
            species_mesh.to_real_field(out=real, normalize=False)

            # add to the mean number of objects per cell and total number
            attrs['num_per_cell'] += real.attrs['num_per_cell']
            attrs['N'] += real.attrs['N']

            # store the meta-data for this species, with a prefix
            attrs.update(attrs_to_dict(real, name+'.'))

        # # normalize the field by nbar -> this is now 1+delta
        if normalize:
            real[:] /= attrs['num_per_cell']

        # update the meta-data
        real.attrs.clear()
        real.attrs.update(attrs)

        # compute total shot noise by summing of shot noise each species
        real.attrs['shotnoise'] = 0
        total_weight = sum(real.attrs['%s.W' %name] for name in self.source.species)
        for name in self.source.species:
            this_Pshot = real.attrs['%s.shotnoise' %name]
            this_weight = real.attrs['%s.W' %name]
            real.attrs['shotnoise'] += (this_weight/total_weight)**2 * this_Pshot

        return real

    def _get_compensation(self):
        from nbodykit.source.mesh.catalog import get_compensation
        return get_compensation(self.interlaced, self.resampler)

