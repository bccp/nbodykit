from nbodykit.source.catalogmesh.species import MultipleSpeciesCatalogMesh
import logging

class FKPCatalogMesh(MultipleSpeciesCatalogMesh):
    """
    A subclass of :class:`~nbodykit.base.catalogmesh.CatalogMesh`
    designed to paint a :class:`~nbodykit.source.catalog.fkp.FKPCatalog` to
    a mesh.

    This requires ``data`` and ``randoms`` CatalogSource objects.

    Internally, all of the columns in ``data`` and ``randoms`` are stored,
    with names prefixed by ``data.`` or ``randoms.``

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
                    comp_weight, fkp_weight, nbar, position='Position'):

        from nbodykit.source.catalog import FKPCatalog
        if not isinstance(source, FKPCatalog):
            raise TypeError("the input source for FKPCatalogMesh must be a FKPCatalog")

        self.source  = source
        self.comp_weight = comp_weight
        self.fkp_weight = fkp_weight
        self.nbar = nbar

        MultipleSpeciesCatalogMesh.__init__(self, source, BoxSize, Nmesh, dtype,
                                            'TotalWeight', 'Value', selection, position=position)

    def to_real_field(self):
        """
        Paint the FKP density field, returning a ``RealField``.

        Given the ``data`` and ``randoms`` catalogs, this paints:

        .. math::

            F(x) = w_fkp(x) * [w_comp(x)*n_data(x) - alpha * w_comp(x)*n_randoms(x)]
        """
        attrs = {}

        # determine alpha, the weighted number ratio
        for name in self.source.species:
            attrs[name+'.W'] = self._weighted_total(name)
        attrs['alpha'] = attrs['data.W'] / attrs['randoms.W']

        # randoms get an additional weight of -alpha
        self['randoms/TotalWeight'] *= -1.0 * attrs['alpha']

        # paint w_data*n_data - alpha*w_randoms*n_randoms
        real = MultipleSpeciesCatalogMesh.to_real_field(self, normalize=False)

        # the rest of the meta-data
        for name in self.source.species:
            attrs[name + '.norm'] = self._normalization(name)
            attrs[name + '.shotnoise'] = self._shotnoise(name)

        # finish the statistics
        attrs['randoms.norm'] *= attrs['alpha']
        attrs['randoms.shotnoise'] *= attrs['alpha']**2 / attrs['randoms.norm']
        attrs['data.shotnoise'] /=  attrs['randoms.norm']
        attrs['shotnoise'] = attrs['data.shotnoise'] + attrs['randoms.shotnoise']

        # divide by volume per cell to go from number to number density
        vol_per_cell = (self.pm.BoxSize/self.pm.Nmesh).prod()
        real[:] /= vol_per_cell

        # update the meta-data
        real.attrs.update(attrs)

        return real

    def _normalization(self, name):
        """
        Compute the power spectrum normalization, using either the
        `data` or `randoms` source

        This computes

        .. math::

            A = \sum \bar{n} w_\mathrm[comp] w_\mathrm{fkp}^2

        References
        ----------
        see Eqs. 13,14 of Beutler et al. 2014, "The clustering of galaxies in the
        SDSS-III Baryon Oscillation Spectroscopic Survey: testing gravity with redshift
        space distortions using the power spectrum multipoles"
        """
        sel = self.source.compute(self[name+'/'+self.selection])

        nbar        = self[name+'/'+self.nbar][sel]
        comp_weight = self[name+'/'+self.comp_weight][sel]
        fkp_weight  = self[name+'/'+self.fkp_weight][sel]
        A           = nbar*comp_weight*fkp_weight**2

        A = self.source.compute(A.sum())
        return self.comm.allreduce(A)

    def _shotnoise(self, name):
        """
        Compute the power spectrum shot noise, using either the
        `data` or `randoms` source

        This computes

        .. math::

            S = \sum (w_\mathrm[comp] w_\mathrm{fkp})^2

        References
        ----------
        see Eq. 15 of Beutler et al. 2014, "The clustering of galaxies in the
        SDSS-III Baryon Oscillation Spectroscopic Survey: testing gravity with redshift
        space distortions using the power spectrum multipoles"
        """
        sel = self.source.compute(self[name+'/'+self.selection])

        comp_weight = self[name+'/'+self.comp_weight][sel]
        fkp_weight  = self[name+'/'+self.fkp_weight][sel]
        S           = (comp_weight*fkp_weight)**2

        S = self.source.compute(S.sum())
        return self.comm.allreduce(S)

    def _weighted_total(self, name):
        """
        Compute the weighted total number of objects, using either the
        `data` or `randoms` source

        This is just the sum of the completeness weights:

        .. math::

            W = \sum w_\mathrm{comp}
        """
        sel = self.source.compute(self[name+'/'+self.selection])
        comp_weight = self[name+'/'+self.comp_weight][sel]

        wsum = self.source.compute(comp_weight.sum())
        return self.comm.allreduce(wsum)
