from nbodykit.source.mesh import MultipleSpeciesCatalogMesh
from nbodykit.source.mesh import CatalogMesh
from nbodykit.utils import attrs_to_dict
import logging
import numpy

class FKPCatalogMesh(MultipleSpeciesCatalogMesh):
    """
    A subclass of
    :class:`~nbodykit.source.catalogmesh.species.MultipleSpeciesCatalogMesh`
    designed to paint a :class:`~nbodykit.source.catalog.fkp.FKPCatalog` to
    a mesh.

    The multiple species here are ``data`` and ``randoms`` CatalogSource
    objects, where ``randoms`` is a catalog of randomly distributed objects
    with no instrinsic clustering that defines the survey volume.

    The position of the catalogs are re-centered to the ``[-L/2, L/2]``
    where ``L`` is the size of the Cartesian box.

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
    selection : str
        column in ``source`` that selects the subset of particles to grid
        to the mesh
    comp_weight : str
        the completeness weight column name
    fkp_weight : str
        the FKP weight column name
    nbar : str
        the n(z) column name
    position : str, optional
        column in ``source`` specifying the position coordinates; default
        is ``Position``
    """
    logger = logging.getLogger('FKPCatalogMesh')

    def __init__(self, source, BoxSize, BoxCenter, Nmesh, dtype, selection,
                    comp_weight, fkp_weight, nbar, value='Value',
                    position='Position', interlaced=False,
                    compensated=False, resampler='cic'):

        from .catalog import FKPCatalog
        if not isinstance(source, FKPCatalog):
            raise TypeError("the input source for FKPCatalogMesh must be a FKPCatalog")

        uncentered_position = position
        position = '_RecenteredPosition'
        weight = '_TotalWeight'

        self.attrs.update(source.attrs)

        self.recenter_box(BoxSize, BoxCenter)

        MultipleSpeciesCatalogMesh.__init__(self, source=source,
                        BoxSize=BoxSize, Nmesh=Nmesh,
                        dtype=dtype, weight=weight, value=value, selection=selection, position=position,
                        interlaced=interlaced, compensated=compensated, resampler=resampler)

        self._uncentered_position = uncentered_position
        self.comp_weight = comp_weight
        self.fkp_weight = fkp_weight
        self.nbar = nbar

    def __getitem__(self, key):
        """
        If indexed by a species name, return a CatalogMesh object holding
        only the data columns for that species with the same parameters as
        the current object.

        If not a species name, this has the same behavior as
        :func:`CatalogSource.__getitem__`.
        """
        assert key in self.source.species, "the species is not defined in the source"

        # CatalogSource holding only requested species
        cat = self.source[key]

        assert cat.comm is self.comm

        # view as a catalog mesh
        mesh = CatalogMesh(cat,
                BoxSize=self.attrs['BoxSize'],
                Nmesh=self.attrs['Nmesh'],
                dtype=self.dtype,
                Weight=self.TotalWeight(key),
                Value=cat[self.value],
                Selection=cat[self.selection],
                Position=self.RecenteredPosition(key),
                interlaced=self.interlaced,
                compensated=self.compensated,
                resampler=self.resampler,
            )

        return mesh

    def recenter_box(self, BoxSize, BoxCenter):
        """
        Re-center the box by applying the new box center to the column specified
        by :attr:`position`.

        This ensures that the position column is always re-centered to
        ``[-L/2,L/2]`` where ``L`` is the BoxSize.
        """
        # check input type
        BoxSize = numpy.ones(3) * BoxSize
        BoxCenter = numpy.ones(3) * BoxCenter

        # update meta-data
        for val, name in zip([BoxSize, BoxCenter], ['BoxSize', 'BoxCenter']):
            self.attrs[name] = val


    def to_real_field(self):
        r"""
        Paint the FKP density field, returning a ``RealField``.

        Given the ``data`` and ``randoms`` catalogs, this paints:

        .. math::

            F(x) = w_\mathrm{fkp}(x) * [w_\mathrm{comp}(x)*n_\mathrm{data}(x) -
                        \alpha * w_\mathrm{comp}(x)*n_\mathrm{randoms}(x)]


        This computes the following meta-data attributes in the process of
        painting, returned in the :attr:`attrs` attributes of the returned
        RealField object:

        - randoms.W, data.W :
            the weighted sum of randoms and data objects; see
            :func:`weighted_total`
        - alpha : float
            the ratio of ``data.W`` to ``randoms.W``
        - randoms.norm, data.norm : float
            the power spectrum normalization; see :func:`normalization`
        - randoms.shotnoise, data.shotnoise: float
            the shot noise for each sample; see :func:`shotnoise`
        - shotnoise : float
            the total shot noise, equal to the sum of ``randoms.shotnoise``
            and ``data.shotnoise``
        - randoms.num_per_cell, data.num_per_cell : float
            the mean number of weighted objects per cell for each sample
        - num_per_cell : float
            the mean number of weighted objects per cell

        For further details on the meta-data, see
        :ref:`the documentation <fkp-meta-data>`.

        Returns
        -------
        :class:`~pmesh.pm.RealField` :
            the field object holding the FKP density field in real space
        """

        attrs = {}

        # determine alpha, the weighted number ratio
        for name in self.source.species:
            attrs[name+'.W'] = self.weighted_total(name)

        attrs['alpha'] = attrs['data.W'] / attrs['randoms.W']

        # paint the data
        real = self['data'].to_real_field(normalize=False)
        real.attrs.update(attrs_to_dict(real, 'data.'))
        if self.comm.rank == 0:
            self.logger.info("data painted.")

        if self.source['randoms'].csize > 0:

            # paint the randoms
            real2 = self['randoms'].to_real_field(normalize=False)

            # normalize the randoms by alpha
            real2[:] *= -1. * attrs['alpha']

            if self.comm.rank == 0:
                self.logger.info("randoms painted.")

            real[:] += real2[:]
            real.attrs.update(attrs_to_dict(real2, 'randoms.'))

        # divide by volume per cell to go from number to number density
        vol_per_cell = (self.pm.BoxSize/self.pm.Nmesh).prod()
        real[:] /= vol_per_cell

        if self.comm.rank == 0:
            self.logger.info("volume per cell is %g" % vol_per_cell)

        # remove shot noise estimates (they are inaccurate in this case)
        real.attrs.update(attrs)
        real.attrs.pop('data.shotnoise', None)
        real.attrs.pop('randoms.shotnoise', None)

        return real

    def RecenteredPosition(self, name):
        """
        The Position of the objects, re-centered on the mesh to
        the range ``[-BoxSize/2, BoxSize/2]``.

        This subtracts ``BoxCenter`` from :attr:`attrs` from the original
        position array.
        """
        assert name in ['data', 'randoms']
        return self.source[name][self._uncentered_position] - self.attrs['BoxCenter']

    def TotalWeight(self, name):
        """
        The total weight for the mesh is the completenes weight times
        the FKP weight.
        """
        assert name in ['data', 'randoms']
        return self.source[name][self.comp_weight] * self.source[name][self.fkp_weight]

    def weighted_total(self, name):
        r"""
        Compute the weighted total number of objects, using either the
        ``data`` or ``randoms`` source:

        This is the sum of the completeness weights:

        .. math::

            W = \sum w_\mathrm{comp}
        """
        # the selection
        sel = self.source.compute(self.source[name][self.selection])

        # the selected mesh for "name"
        selected = self.source[name][sel]

        # sum up completeness weights
        wsum = self.source.compute(selected[self.comp_weight].sum())
        return self.comm.allreduce(wsum)
