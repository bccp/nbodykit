from nbodykit.source.catalog.species import MultipleSpeciesCatalog
from nbodykit.transform import ConstantArray

import numpy
import logging


class FKPCatalog(MultipleSpeciesCatalog):
    """
    An interface for simultaneous modeling of a ``data`` CatalogSource and a
    ``randoms`` CatalogSource, in the spirit of
    `Feldman, Kaiser, and Peacock, 1994 <https://arxiv.org/abs/astro-ph/9304022>`_.

    This main functionality of this class is:

    *   provide a uniform interface to accessing columns from the
        ``data`` CatalogSource and ``randoms`` CatalogSource, using
        column names prefixed with "data/" or "randoms/"
    *   compute the shared :attr:`BoxSize` of the source, by
        finding the maximum Cartesian extent of the ``randoms``
    *   provide an interface to a mesh object, which knows how to paint the
        FKP density field from the ``data`` and ``randoms``

    Parameters
    ----------
    data : CatalogSource
        the CatalogSource of particles representing the `data` catalog
    randoms : CatalogSource
        the CatalogSource of particles representing the `randoms` catalog
    BoxSize : float, 3-vector, optional
        the size of the Cartesian box to use for the unified `data` and
        `randoms`; if not provided, the maximum Cartesian extent of the
        `randoms` defines the box
    BoxPad : float, 3-vector, optional
        optionally apply this additional buffer to the extent of the
        Cartesian box

    References
    ----------
    - `Feldman, Kaiser, and Peacock, 1994 <https://arxiv.org/abs/astro-ph/9304022>`__
    """
    logger = logging.getLogger('FKPCatalog')

    def __repr__(self):
        return "FKPCatalog(species=%s)" %str(self.attrs['species'])

    def __init__(self, data, randoms, BoxSize=None, BoxPad=0.02):

        # init the base class
        MultipleSpeciesCatalog.__init__(self, ['data', 'randoms'], data, randoms)

        # add a default FKP weight columns, if it doesnt exist
        for i, name in enumerate(self.species):
            if 'FKPWeight' not in self[name]:
                self[name]['FKPWeight'] = 1.0 # unity by default

        # determine the BoxSize
        if numpy.isscalar(BoxSize):
            BoxSize = numpy.ones(3)*BoxSize
        self.attrs['BoxSize'] = BoxSize
        if numpy.isscalar(BoxPad):
            BoxPad = numpy.ones(3)*BoxPad
        self.attrs['BoxPad'] = BoxPad

    def _define_cartesian_box(self, position, selection):
        """
        Internal function to put the :attr:`randoms` CatalogSource in a
        Cartesian box.

        This function add two necessary attribues:

        #. :attr:`BoxSize` : array_like, (3,)
            if not provided, the BoxSize in each direction is computed from
            the maximum extent of the Cartesian coordinates of the :attr:`randoms`
            Source, with an optional, additional padding
        #. :attr:`BoxCenter`: array_like, (3,)
            the mean coordinate value in each direction; this is used to re-center
            the Cartesian coordinates of the :attr:`data` and :attr:`randoms`
            to the range of ``[-BoxSize/2, BoxSize/2]``
        """
        from nbodykit.utils import get_data_bounds

        # compute the min/max of the position data
        pos, sel = self['randoms'].read([position, selection])
        pos_min, pos_max = get_data_bounds(pos, self.comm, selection=sel)

        # used to center the data in the first cartesian quadrant
        delta = abs(pos_max - pos_min)
        self.attrs['BoxCenter'] = 0.5 * (pos_min + pos_max)

        # BoxSize is padded diff of min/max coordinates
        if self.attrs['BoxSize'] is None:
            delta *= 1.0 + self.attrs['BoxPad']
            self.attrs['BoxSize'] = numpy.ceil(delta) # round up to nearest integer

        # log some info
        if self.comm.rank == 0:
            self.logger.info("BoxSize = %s" %str(self.attrs['BoxSize']))
            self.logger.info("cartesian coordinate range: %s : %s" %(str(pos_min), str(pos_max)))
            self.logger.info("BoxCenter = %s" %str(self.attrs['BoxCenter']))

    def to_mesh(self, Nmesh=None, BoxSize=None, dtype='f4', interlaced=False,
                compensated=False, window='cic', fkp_weight='FKPWeight',
                comp_weight='Weight', nbar='NZ', selection='Selection',
                position='Position'):

        """
        Convert the FKPCatalog to a mesh, which knows how to "paint" the
        FKP density field.

        Additional keywords to the :func:`to_mesh` function include the
        FKP weight column, completeness weight column, and the column
        specifying the number density as a function of redshift.

        Parameters
        ----------
        Nmesh : int, 3-vector, optional
            the number of cells per box side; if not specified in `attrs`, this
            must be provided
        BoxSize : float, 3-vector, optional
            the size of the box; if provided, this will use the default value
            in `attrs`
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
            particles to the mesh; see ``pmesh.window.methods`` for choices
        fkp_weight : str, optional
            the name of the column in the source specifying the FKP weight;
            this weight is applied to the FKP density field:
            ``n_data - alpha*n_randoms``
        comp_weight : str, optional
            the name of the column in the source specifying the completeness
            weight; this weight is applied to the individual fields, either
            ``n_data``  or ``n_random``
        selection : str, optional
            the name of the column used to select a subset of the source when
            painting
        nbar : str, optional
            the name of the column specifying the number density as a function
            of redshift
        position : str, optional
            the name of the column that specifies the position data of the
            objects in the catalog
        """
        from nbodykit.source.catalogmesh import FKPCatalogMesh

        # verify that all of the required columns exist
        for name in self.species:
            for col in [fkp_weight, comp_weight, nbar]:
                if col not in self[name]:
                    raise ValueError("the '%s' species is missing the '%s' column" %(name, col))

        if Nmesh is None:
            try:
                Nmesh = self.attrs['Nmesh']
            except KeyError:
                raise ValueError("cannot convert FKP source to a mesh; 'Nmesh' keyword is not "
                                 "supplied and the FKP source does not define one in 'attrs'.")

        # first, define the Cartesian box
        self._define_cartesian_box(position, selection)

        if BoxSize is None:
            BoxSize = self.attrs['BoxSize']

        # initialize the FKP mesh
        kws = {'Nmesh':Nmesh, 'BoxSize':BoxSize, 'dtype':dtype, 'selection':selection}
        return FKPCatalogMesh(self,
                              nbar=nbar,
                              comp_weight=comp_weight,
                              fkp_weight=fkp_weight,
                              position=position,
                              value='Value',
                              interlaced=interlaced,
                              compensated=compensated,
                              window=window,
                              **kws)
