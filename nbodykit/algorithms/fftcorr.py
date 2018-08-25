import os
import numpy
import logging

from nbodykit import CurrentMPIComm
from nbodykit.binned_statistic import BinnedStatistic
from nbodykit.meshtools import SlabIterator
from nbodykit.base.catalog import CatalogSourceBase
from nbodykit.base.mesh import MeshSource

from .fftpower import FFTBase
from .fftpower import project_to_basis
from .fftpower import _find_unique_edges

class FFTCorr(FFTBase):
    r"""
    Algorithm to compute the 1d or 2d correlation and/or multipoles
    in a periodic box, using a Fast Fourier Transform (FFT).

    This computes the power spectrum as the square of the Fourier modes of the
    density field, which are computed via a FFT. Then it is transformed back
    to obtain the correlation function.

    Results are computed when the object is inititalized. See the documenation
    of :func:`~FFTCorr.run` for the attributes storing the results.

    .. note::
        This is very similar to :class:`~nbodykit.algorithms.fftpower.FFTPower`. 

    Parameters
    ----------
    first : CatalogSource, MeshSource
        the source for the first field; if a CatalogSource is provided, it
        is automatically converted to MeshSource using the default painting
        parameters (via :func:`~nbodykit.base.catalogmesh.CatalogMesh.to_mesh`)
    mode : {'1d', '2d'}
        compute either 1d or 2d power spectra
    Nmesh : int, optional
        the number of cells per side in the particle mesh used to paint the source
    BoxSize : int, 3-vector, optional
        the size of the box
    second : CatalogSource, MeshSource, optional
        the second source for cross-correlations
    los : array_like , optional
        the direction to use as the line-of-sight; must be a unit vector
    Nmu : int, optional
        the number of mu bins to use from :math:`\mu=[0,1]`;
        if `mode = 1d`, then ``Nmu`` is set to 1
    dr : float, optional
        the linear spacing of ``r`` bins to use; if not provided, the
        fundamental mode  of the box is used; if `dr=0`, the bins are tight, such
        that each bin has a unique r value.
    rmin : float, optional
        the lower edge of the first ``r`` bin to use
    poles : list of int, optional
        a list of multipole numbers ``ell`` to compute :math:`\xi_\ell(r)`
        from :math:`\xi(r,\mu)`
    """
    logger = logging.getLogger('FFTCorr')

    def __init__(self, first, mode, Nmesh=None, BoxSize=None, second=None,
                    los=[0, 0, 1], Nmu=5, dr=None, rmin=0., poles=[]):

        # mode is either '1d' or '2d'
        if mode not in ['1d', '2d']:
            raise ValueError("`mode` should be either '1d' or '2d'")

        if poles is None:
            poles = []

        # check los
        if numpy.isscalar(los) or len(los) != 3:
            raise ValueError("line-of-sight ``los`` should be vector with length 3")
        if not numpy.allclose(numpy.einsum('i,i', los, los), 1.0, rtol=1e-5):
            raise ValueError("line-of-sight ``los`` must be a unit vector")

        FFTBase.__init__(self, first, second, Nmesh, BoxSize)

        # save meta-data
        self.attrs['mode'] = mode
        self.attrs['los'] = los
        self.attrs['Nmu'] = Nmu
        self.attrs['poles'] = poles

        if dr is None:
            dr = self.attrs['BoxSize'].min() / self.attrs['Nmesh'].max()

        self.attrs['dr'] = dr
        self.attrs['rmin'] = rmin

        self.corr, self.poles = self.run()

        # compatability
        self.attrs.update(self.corr.attrs)

    def run(self):
        r"""
        Compute the correlation function in a periodic box, using FFTs.

        returns
        -------
        corr : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            a BinnedStatistic object that holds the measured :math:`\xi(r)` or
            :math:`\xi(r,\mu)`. It stores the following variables:

            - r :
                the mean value for each ``r`` bin
            - mu : ``mode=2d`` only
                the mean value for each ``mu`` bin
            - corr :
                real array storing the correlation function
            - modes :
                the number of modes averaged together in each bin

        poles : :class:`~nbodykit.binned_statistic.BinnedStatistic` or ``None``
            a BinnedStatistic object to hold the multipole results
            :math:`\xi_\ell(r)`; if no multipoles were requested by the user,
            this is ``None``. It stores the following variables:

            - r :
                the mean value for each ``r`` bin
            - power_L :
                complex array storing the real and imaginary components for
                the :math:`\ell=L` multipole
            - modes :
                the number of modes averaged together in each bin

        corr.attrs, poles.attrs : dict
            dictionary of meta-data; in addition to storing the input parameters,
            it includes the following fields computed during the algorithm
            execution:

            - shotnoise : float
                the power Poisson shot noise, equal to :math:`V/N`, where
                :math:`V` is the volume of the box and `N` is the total
                number of objects; if a cross-correlation is computed, this
                will be equal to zero
            - N1 : int
                the total number of objects in the first source
            - N2 : int
                the total number of objects in the second source
        """

        # only need one mu bin if 1d case is requested
        if self.attrs['mode'] == "1d": self.attrs['Nmu'] = 1

        # measure the 3D power (y3d is a ComplexField)
        y3d, attrs = self._compute_3d_power(self.first, self.second)

        # measure the 3D correlation (y3d is a RealField)
        y3d = y3d.c2r(out=Ellipsis)

        # correlation is dimensionless
        # Note that L^3 cancels with dk^3.
        y3d[...] *= 1.0 / y3d.BoxSize.prod()

        # binning in k out to the minimum nyquist frequency
        # (accounting for possibly anisotropic box)
        dr = self.attrs['dr']
        rmin = self.attrs['rmin']
        if dr > 0:
            redges = numpy.arange(rmin, 0.5 * y3d.BoxSize.min() + dr/2, dr)
            rcenters = None
        else:
            redges, rcenters = _find_unique_edges(y3d.x, y3d.BoxSize / y3d.Nmesh, self.comm)

        # project on to the desired basis
        muedges = numpy.linspace(0, 1, self.attrs['Nmu']+1, endpoint=True)
        edges = [redges, muedges]
        coords = [rcenters, None]
        result, pole_result = project_to_basis(y3d, edges,
                                               poles=self.attrs['poles'],
                                               los=self.attrs['los'])

        # format the corr results into structured array
        if self.attrs['mode'] == "1d":
            cols = ['r', 'corr', 'modes']
            icols = [0, 2, 3]
            edges = edges[0:1]
            coords = coords[0:1]
        else:
            cols = ['r', 'mu', 'corr', 'modes']
            icols = [0, 1, 2, 3]

        # corr results as a structured array
        dtype = numpy.dtype([(name, result[icol].dtype.str) for icol,name in zip(icols,cols)])
        corr = numpy.squeeze(numpy.empty(result[0].shape, dtype=dtype))
        for icol, col in zip(icols, cols):
            corr[col][:] = numpy.squeeze(result[icol])

        # multipole results as a structured array
        poles = None
        if pole_result is not None:
            r, poles, N = pole_result
            cols = ['r'] + ['corr_%d' %l for l in self.attrs['poles']] + ['modes']
            result = [r] + [pole for pole in poles] + [N]

            dtype = numpy.dtype([(name, result[icol].dtype.str) for icol,name in enumerate(cols)])
            poles = numpy.empty(result[0].shape, dtype=dtype)
            for icol, col in enumerate(cols):
                poles[col][:] = result[icol]

        # set all the necessary results
        return self._make_datasets(edges, poles, corr, coords, attrs)

    def __getstate__(self):
        state = dict(
                    corr=self.corr.__getstate__(),
                    poles=self.poles.__getstate__() if self.poles is not None else None,
                    attrs=self.attrs)
        return state

    def __setstate__(self, state):
        self.attrs = state['attrs']
        self.corr = BinnedStatistic.from_state(state['corr'])
        if state['poles'] is not None:
            self.poles = BinnedStatistic.from_state(state['poles'])

    def _make_datasets(self, edges, poles, corr, coords, attrs):

        if self.attrs['mode'] == '1d':
            corr = BinnedStatistic(['r'], edges, corr, fields_to_sum=['modes'], coords=coords, **attrs)
        else:
            corr = BinnedStatistic(['r', 'mu'], edges, corr, fields_to_sum=['modes'], coords=coords, **attrs)
        if poles is not None:
            poles = BinnedStatistic(['r'], [corr.edges['r']], poles, fields_to_sum=['modes'], coords=coords, **attrs)

        return corr, poles
