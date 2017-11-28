from nbodykit.source.catalogmesh.species import MultipleSpeciesCatalogMesh
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

    def __new__(cls, source, BoxSize, Nmesh, dtype, selection,
                    comp_weight, fkp_weight, nbar, value='Value',
                    position='Position', interlaced=False,
                    compensated=False, window='cic'):

        from nbodykit.source.catalog import FKPCatalog
        if not isinstance(source, FKPCatalog):
            raise TypeError("the input source for FKPCatalogMesh must be a FKPCatalog")

        uncentered_position = position
        position = '_RecenteredPosition'
        weight = '_TotalWeight'

        obj = super(FKPCatalogMesh, cls).__new__(cls, source, BoxSize, Nmesh,
                        dtype, weight, value, selection, position=position,
                        interlaced=interlaced, compensated=compensated, window=window)

        obj._uncentered_position = uncentered_position
        obj.comp_weight = comp_weight
        obj.fkp_weight = fkp_weight
        obj.nbar = nbar

        return obj

    def recenter_box(self, BoxSize, BoxCenter):
        """
        Re-center the box by applying the new box center to the column specified
        by :attr:`position`.

        This ensures that the position column is always re-centered to
        ``[-L/2,L/2]`` where ``L`` is the BoxSize.
        """
        # check input type
        for val in [BoxSize, BoxCenter]:
            if not isinstance(val, (list, numpy.ndarray)) or len(val) != 3:
                raise ValueError("recenter_box arguments should be a vector of length 3")

        # update meta-data
        for val, name in zip([BoxSize, BoxCenter], ['BoxSize', 'BoxCenter']):
            self.attrs[name] = val
            self.base.attrs[name] = val


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
        # add necessary FKP columns for INTERNAL use
        for name in self.base.species:

            # a total weight for the mesh is completeness weight x FKP weight
            self[name]['_TotalWeight'] = self.TotalWeight(name)

            # position on the mesh is re-centered to [-BoxSize/2, BoxSize/2]
            self[name]['_RecenteredPosition'] = self.RecenteredPosition(name)

        attrs = {}

        # determine alpha, the weighted number ratio
        for name in self.base.species:
            attrs[name+'.W'] = self.weighted_total(name)
        attrs['alpha'] = attrs['data.W'] / attrs['randoms.W']

        # paint the randoms
        real = self['randoms'].to_real_field(normalize=False)
        real.attrs.update(attrs_to_dict(real, 'randoms.'))

        # normalize the randoms by alpha
        real[:] *= -1. * attrs['alpha']

        # paint the data
        real2 = self['data'].to_real_field(normalize=False)
        real[:] += real2[:]
        real.attrs.update(attrs_to_dict(real2, 'data.'))

        # divide by volume per cell to go from number to number density
        vol_per_cell = (self.pm.BoxSize/self.pm.Nmesh).prod()
        real[:] /= vol_per_cell

        # remove shot noise estimates (they are inaccurate in this case)
        real.attrs.update(attrs)
        real.attrs.pop('data.shotnoise', None)
        real.attrs.pop('randoms.shotnoise', None)

        # delete internal columns
        for name in self.base.species:
            del self[name+'/_RecenteredPosition']
            del self[name+'/_TotalWeight']

        return real

    def RecenteredPosition(self, name):
        """
        The Position of the objects, re-centered on the mesh to
        the range ``[-BoxSize/2, BoxSize/2]``.

        This subtracts ``BoxCenter`` from :attr:`attrs` from the original
        position array.
        """
        assert name in ['data', 'randoms']
        return self[name][self._uncentered_position] - self.attrs['BoxCenter']

    def TotalWeight(self, name):
        """
        The total weight for the mesh is the completenes weight times
        the FKP weight.
        """
        assert name in ['data', 'randoms']
        return self[name][self.comp_weight] * self[name][self.fkp_weight]

    def weighted_total(self, name):
        r"""
        Compute the weighted total number of objects, using either the
        ``data`` or ``randoms`` source:

        This is the sum of the completeness weights:

        .. math::

            W = \sum w_\mathrm{comp}
        """
        # the selection
        sel = self.compute(self[name][self.selection])

        # the selected mesh for "name"
        mesh = self[name][sel]

        # sum up completeness weights
        wsum = self.compute(mesh[self.comp_weight].sum())
        return self.comm.allreduce(wsum)
