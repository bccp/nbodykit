from nbodykit.source.catalog.species import MultipleSpeciesCatalog
from nbodykit.transform import ConstantArray

import numpy
import logging

def FKPWeightFromNbar(P0, nbar):
    """ Create FKPWeight from nbar, the number density of objects per redshift.

        Parameters
        ----------
        P0 : float
            the FKP normalization, when P0 == 0, returns 1.0, ignoring size / shape of nbar.

        nbar : array_like
            the number density of objects per redshift

        Returns
        -------
        FKPWeight : the FKPWeight, can be assigned to a catalog as a column
        to be consumed :class:`ConvolvedFFTPower`

    """
    if P0 != 0:
        return 1.0 / (1. + P0 * nbar)

    return 1.0


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
    randoms : CatalogSource, or None
        the CatalogSource of particles representing the `randoms` catalog
        if None is given an empty catalog is used.
    BoxSize : float, 3-vector, optional
        the size of the Cartesian box to use for the unified `data` and
        `randoms`; if not provided, the maximum Cartesian extent of the
        `randoms` defines the box
    BoxPad : float, 3-vector, optional
        optionally apply this additional buffer to the extent of the
        Cartesian box
    nbar : str, optional
        the name of the column specifying the number density as a function
        of redshift. default is NZ.
    P0 : float or None
        if not None, a column named FKPWeight is added to data and random based on nbar.

    References
    ----------
    - `Feldman, Kaiser, and Peacock, 1994 <https://arxiv.org/abs/astro-ph/9304022>`__
    """
    logger = logging.getLogger('FKPCatalog')

    def __repr__(self):
        return "FKPCatalog(species=%s)" %str(self.attrs['species'])

    def __init__(self, data, randoms, BoxSize=None, BoxPad=0.02, P0=None, nbar='NZ'):

        if randoms is None:
            # create an empty catalog.
            randoms = data[:0]

        # init the base class
        MultipleSpeciesCatalog.__init__(self, ['data', 'randoms'], data, randoms)

        for i, name in enumerate(self.species):
            if nbar not in self[name]:
                raise ValueError("Column `%s` is not defined in `%s`" % (nbar, name))

        self.nbar = nbar

        if P0 is not None:
            # create a default FKP weight column, based on nbar
            for i, name in enumerate(self.species):
                self[name]['FKPWeight'] = FKPWeightFromNbar(P0, self[name][self.nbar])
        else:
            # add a default FKP weight columns, based on nbar
            for i, name in enumerate(self.species):
                if 'FKPWeight' not in self[name]:
                    self[name]['FKPWeight'] = 1.0

        # determine the BoxSize
        if numpy.isscalar(BoxSize):
            BoxSize = numpy.ones(3)*BoxSize
        self.attrs['BoxSize'] = BoxSize
        if numpy.isscalar(BoxPad):
            BoxPad = numpy.ones(3)*BoxPad
        self.attrs['BoxPad'] = BoxPad

    def _define_bbox(self, position, selection, species):
        """
        Internal function to put the :attr:`randoms` CatalogSource in a
        Cartesian bounding box, using the positions of the given species.

        This function computings the size and center of the bounding box.

        #. `BoxSize` : array_like, (3,)
            if not provided, the BoxSize in each direction is computed from
            the maximum extent of the Cartesian coordinates of the :attr:`randoms`
            Source, with an optional, additional padding
        #. `BoxCenter`: array_like, (3,)
            the mean coordinate value in each direction; this is used to re-center
            the Cartesian coordinates of the :attr:`data` and :attr:`randoms`
            to the range of ``[-BoxSize/2, BoxSize/2]``

        """
        from nbodykit.utils import get_data_bounds

        # compute the min/max of the position data
        pos, sel = self[species].read([position, selection])
        pos_min, pos_max = get_data_bounds(pos, self.comm, selection=sel)

        self.logger.info("cartesian coordinate range: %s : %s" %(str(pos_min), str(pos_max)))

        if numpy.isinf(pos_min).any() or numpy.isinf(pos_max).any():
            raise ValueError("Range of positions from `%s` is infinite;"
                    "try to use the other species with (bbox_from_species='data'." % species)

        # used to center the data in the first cartesian quadrant
        delta = abs(pos_max - pos_min)
        BoxCenter = 0.5 * (pos_min + pos_max)

        # BoxSize is padded diff of min/max coordinates
        if self.attrs['BoxSize'] is None:
            delta *= 1.0 + self.attrs['BoxPad']
            BoxSize = numpy.ceil(delta) # round up to nearest integer
        else:
            BoxSize = self.attrs['BoxSize']

        return BoxSize, BoxCenter

    def to_mesh(self, Nmesh=None, BoxSize=None, BoxCenter=None, dtype='f4', interlaced=False,
                compensated=False, resampler='cic', fkp_weight='FKPWeight',
                comp_weight='Weight', selection='Selection',
                position='Position', bbox_from_species=None, window=None, nbar=None):

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
        dtype : str, dtype, optional
            the data type of the mesh when painting
        interlaced : bool, optional
            whether to use interlacing to reduce aliasing when painting the
            particles on the mesh
        compensated : bool, optional
            whether to apply a Fourier-space transfer function to account for
            the effects of the gridding + aliasing
        resampler : str, optional
            the string name of the resampler to use when interpolating the
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
        position : str, optional
            the name of the column that specifies the position data of the
            objects in the catalog
        bbox_from_species: str, optional
            if given, use the species to infer a bbox.
            if not give, will try random, then data (if random is empty)
        window : deprecated.
            use resampler=
        nbar: deprecated.
            deprecated. set nbar in the call to FKPCatalog()
        """
        from .catalogmesh import FKPCatalogMesh

        if window is not None:
            import warnings
            resampler = window
            warnings.warn("the window argument is deprecated. Use resampler= instead", DeprecationWarning)

        # verify that all of the required columns exist
        for name in self.species:
            for col in [fkp_weight, comp_weight]:
                if col not in self[name]:
                    raise ValueError("the '%s' species is missing the '%s' column" %(name, col))

        if Nmesh is None:
            try:
                Nmesh = self.attrs['Nmesh']
            except KeyError:
                raise ValueError("cannot convert FKP source to a mesh; 'Nmesh' keyword is not "
                                 "supplied and the FKP source does not define one in 'attrs'.")

        # first, define the Cartesian box
        if bbox_from_species is not None:
            BoxSize1, BoxCenter1 = self._define_bbox(position, selection, bbox_from_species)
        else:
            if self['randoms'].csize > 0:
                BoxSize1, BoxCenter1 = self._define_bbox(position, selection, "randoms")
            else:
                BoxSize1, BoxCenter1 = self._define_bbox(position, selection, "data")

        if BoxSize is None:
            BoxSize = BoxSize1

        if BoxCenter is None:
            BoxCenter = BoxCenter1

        # log some info
        if self.comm.rank == 0:
            self.logger.info("BoxSize = %s" %str(BoxSize))
            self.logger.info("BoxCenter = %s" %str(BoxCenter))

        # initialize the FKP mesh
        kws = {'Nmesh':Nmesh, 'BoxSize':BoxSize, 'BoxCenter' : BoxCenter, 'dtype':dtype, 'selection':selection}
        return FKPCatalogMesh(self,
                              nbar=self.nbar,
                              comp_weight=comp_weight,
                              fkp_weight=fkp_weight,
                              position=position,
                              value='Value',
                              interlaced=interlaced,
                              compensated=compensated,
                              resampler=resampler,
                              **kws)
