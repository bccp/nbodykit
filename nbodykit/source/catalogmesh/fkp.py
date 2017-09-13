from nbodykit.source.catalogmesh.species import MultipleSpeciesCatalogMesh
from nbodykit.base.catalog import column
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

    def __init__(self, source, BoxSize, Nmesh, dtype, selection,
                    comp_weight, fkp_weight, nbar, weight, position='Position'):

        from nbodykit.source.catalog import FKPCatalog
        if not isinstance(source, FKPCatalog):
            raise TypeError("the input source for FKPCatalogMesh must be a FKPCatalog")

        self.source  = source
        self.comp_weight = comp_weight
        self.fkp_weight = fkp_weight
        self.nbar = nbar

        # init the base
        # NOTE: FKP must paint the Position recentered to [-L/2, L/2]
        # so we store that in an internal column "_RecenteredPosition"
        MultipleSpeciesCatalogMesh.__init__(self, source, BoxSize, Nmesh, dtype,
                                            weight, 'Value', selection,
                                            position=position)

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

        # update the position coordinates
        for name in self.source.species:

            # add the old box center offset
            self[name][self.position] += self.attrs['BoxCenter']

            # subtract the new box center offset
            self[name][self.position] -= BoxCenter

        # update meta-data
        for val, name in zip([BoxSize, BoxCenter], ['BoxSize', 'BoxCenter']):
            self.attrs[name] = val
            self.source.attrs[name] = val


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

        # randoms get an additional weight of -alpha
        self['randoms'][self.weight] *= -1.0 * attrs['alpha']

        # paint w_data*n_data - alpha*w_randoms*n_randoms
        real = MultipleSpeciesCatalogMesh.to_real_field(self, normalize=False)

        # divide by volume per cell to go from number to number density
        vol_per_cell = (self.pm.BoxSize/self.pm.Nmesh).prod()
        real[:] /= vol_per_cell

        # remove shot noise estimates (they are inaccurate in this case)
        real.attrs.update(attrs)
        real.attrs.pop('data.shotnoise', None)
        real.attrs.pop('randoms.shotnoise', None)

        return real

    def weighted_total(self, name):
        r"""
        Compute the weighted total number of objects, using either the
        ``data`` or ``randoms`` source:

        This is the sum of the completeness weights:

        .. math::

            W = \sum w_\mathrm{comp}
        """
        sel = self.source.compute(self[name+'/'+self.selection])
        comp_weight = self[name+'/'+self.comp_weight][sel]

        wsum = self.source.compute(comp_weight.sum())
        return self.comm.allreduce(wsum)
