import os
import numpy
import logging

from nbodykit import CurrentMPIComm
from nbodykit.binned_statistic import BinnedStatistic
from nbodykit.meshtools import SlabIterator
from nbodykit.base.catalog import CatalogSourceBase
from nbodykit.base.mesh import MeshSource

class FFTBase(object):
    """
    Base class provides functions for periodic FFT based Power spectrum code.

    Parameters
    ----------
    first : CatalogSource
        the first catalog source
    second : CatalogSource, None
        the second source, or None for auto-correlations
    Nmesh : int, 3-vector
        the number of cells per mesh size
    BoxSize : 3-vector
        the size of the box
    """
    def __init__(self, first, second, Nmesh, BoxSize):
        from pmesh.pm import ParticleMesh

        first = _cast_source(first, Nmesh=Nmesh, BoxSize=BoxSize)
        if second is not None:
            second = _cast_source(second, Nmesh=Nmesh, BoxSize=BoxSize)
        else:
            second = first

        self.first = first
        self.second = second

        # grab comm from first source
        self.comm = first.comm

        # check for comm mismatch
        assert second.comm is first.comm, "communicator mismatch between input sources"

        # check box size
        if not numpy.array_equal(first.attrs['BoxSize'], second.attrs['BoxSize']):
            raise ValueError("'BoxSize' mismatch between sources in FFTPower")

        # save meta-data
        self.attrs = {}
        self.attrs['Nmesh'] = first.attrs['Nmesh'].copy()
        self.attrs['BoxSize'] = first.attrs['BoxSize'].copy()

        self.attrs.update(zip(['Lx', 'Ly', 'Lz'], self.attrs['BoxSize']))
        self.attrs.update({'volume':self.attrs['BoxSize'].prod()})

    def save(self, output):
        """
        Save the result to disk. The format is currently JSON.
        """
        import json
        from nbodykit.utils import JSONEncoder

        # only the master rank writes
        if self.comm.rank == 0:
            self.logger.info('measurement done; saving result to %s' % output)

            with open(output, 'w') as ff:
                json.dump(self.__getstate__(), ff, cls=JSONEncoder)

    @classmethod
    @CurrentMPIComm.enable
    def load(cls, output, comm=None):
        """
        Load a saved result. The result has been saved to disk with :func:`save`.
        """
        import json
        from nbodykit.utils import JSONDecoder
        if comm.rank == 0:
            with open(output, 'r') as ff:
                state = json.load(ff, cls=JSONDecoder)
        else:
            state = None
        state = comm.bcast(state)
        self = object.__new__(cls)
        self.__setstate__(state)
        self.comm = comm

        return self

    def _compute_3d_power(self, first, second):
        """
        Compute and return the power as a function of k vector, for two input sources

        Returns
        -------
        p3d : array_like (complex)
            the 3D complex array holding the power spectrum
        attrs : dict
            meta data of the 3d power
        """
        attrs = {}
        # add self.attrs
        attrs.update(self.attrs)

        c1 = first.compute(mode='complex', Nmesh=self.attrs['Nmesh'])

        # compute the auto power of single supplied field
        if first is second:
            c2 = c1
        else:
            c2 = second.compute(mode='complex', Nmesh=self.attrs['Nmesh'])

        # calculate the 3d power spectrum, slab-by-slab to save memory
        p3d = c1
        for (s0, s1, s2) in zip(p3d.slabs, c1.slabs, c2.slabs):
            s0[...] = s1 * s2.conj()

        for i, s0 in zip(p3d.slabs.i, p3d.slabs):
            # clear the zero mode.
            mask = True
            for i1 in i:
                mask = mask & (i1 == 0)
            s0[mask] = 0

        # the complex field is dimensionless; power is L^3
        # ref to http://icc.dur.ac.uk/~tt/Lectures/UA/L4/cosmology.pdf
        p3d[...] *= self.attrs['BoxSize'].prod()

        # get the number of objects (in a safe manner)
        N1 = c1.attrs.get('N', 0)
        N2 = c2.attrs.get('N', 0)
        attrs.update({'N1':N1, 'N2':N2})

        # add shotnoise (nonzero only for auto-spectra)
        Pshot = 0
        if self.first is self.second:
            if 'shotnoise' in c1.attrs:
                Pshot = c1.attrs['shotnoise']
        attrs['shotnoise'] = Pshot


        return p3d, attrs


class FFTPower(FFTBase):
    """
    Algorithm to compute the 1d or 2d power spectrum and/or multipoles
    in a periodic box, using a Fast Fourier Transform (FFT).

    This computes the power spectrum as the square of the Fourier modes of the
    density field, which are computed via a FFT.

    Results are computed when the object is inititalized. See the documenation
    of :func:`~FFTPower.run` for the attributes storing the results.

    .. note::
        A full tutorial on the class is available in the documentation
        :ref:`here <fftpower>`.

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
    dk : float, optional
        the linear spacing of ``k`` bins to use; if not provided, the
        fundamental mode  of the box is used; if `dk=0` is set, use fine bins
        such that the modes contributing to the bin has identical modulus.
    kmin : float, optional
        the lower edge of the first ``k`` bin to use
    poles : list of int, optional
        a list of multipole numbers ``ell`` to compute :math:`P_\ell(k)`
        from :math:`P(k,\mu)`
    """
    logger = logging.getLogger('FFTPower')

    def __init__(self, first, mode, Nmesh=None, BoxSize=None, second=None,
                    los=[0, 0, 1], Nmu=5, dk=None, kmin=0., poles=[]):

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

        if dk is None:
            dk = 2 * numpy.pi / self.attrs['BoxSize'].min()

        self.attrs['dk'] = dk
        self.attrs['kmin'] = kmin

        self.power, self.poles = self.run()

        # for compatibility, copy power's attrs into self.
        self.attrs.update(self.power.attrs)

    def run(self):
        """
        Compute the power spectrum in a periodic box, using FFTs.

        Returns 
        -------
        power : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            a BinnedStatistic object that holds the measured :math:`P(k)` or
            :math:`P(k,\mu)`. It stores the following variables:

            - k :
                the mean value for each ``k`` bin
            - mu : ``mode=2d`` only
                the mean value for each ``mu`` bin
            - power :
                complex array storing the real and imaginary components of the power
            - modes :
                the number of Fourier modes averaged together in each bin

        poles : :class:`~nbodykit.binned_statistic.BinnedStatistic` or ``None``
            a BinnedStatistic object to hold the multipole results
            :math:`P_\ell(k)`; if no multipoles were requested by the user,
            this is ``None``. It stores the following variables:

            - k :
                the mean value for each ``k`` bin
            - power_L :
                complex array storing the real and imaginary components for
                the :math:`\ell=L` multipole
            - modes :
                the number of Fourier modes averaged together in each bin

        power.attrs, poles.attrs : dict
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

        # binning in k out to the minimum nyquist frequency
        # (accounting for possibly anisotropic box)
        dk = self.attrs['dk']
        kmin = self.attrs['kmin']
        if dk > 0:
            kedges = numpy.arange(kmin, numpy.pi*y3d.Nmesh.min()/y3d.BoxSize.max() + dk/2, dk)
            kcoords = None
        else:
            kedges, kcoords = _find_unique_edges(y3d.x, 2 * numpy.pi / y3d.BoxSize, y3d.pm.comm)

        # project on to the desired basis
        muedges = numpy.linspace(0, 1, self.attrs['Nmu']+1, endpoint=True)
        edges = [kedges, muedges]
        coords = [kcoords, None]
        result, pole_result = project_to_basis(y3d, edges,
                                               poles=self.attrs['poles'],
                                               los=self.attrs['los'])

        # format the power results into structured array
        if self.attrs['mode'] == "1d":
            cols = ['k', 'power', 'modes']
            icols = [0, 2, 3]
            edges = edges[0:1]
            coords = coords[0:1]
        else:
            cols = ['k', 'mu', 'power', 'modes']
            icols = [0, 1, 2, 3]

        # power results as a structured array
        dtype = numpy.dtype([(name, result[icol].dtype.str) for icol,name in zip(icols,cols)])
        power = numpy.squeeze(numpy.empty(result[0].shape, dtype=dtype))
        for icol, col in zip(icols, cols):
            power[col][:] = numpy.squeeze(result[icol])

        # multipole results as a structured array
        poles = None
        if pole_result is not None:
            k, poles, N = pole_result
            cols = ['k'] + ['power_%d' %l for l in self.attrs['poles']] + ['modes']
            result = [k] + [pole for pole in poles] + [N]

            dtype = numpy.dtype([(name, result[icol].dtype.str) for icol,name in enumerate(cols)])
            poles = numpy.empty(result[0].shape, dtype=dtype)
            for icol, col in enumerate(cols):
                poles[col][:] = result[icol]

        return self._make_datasets(edges, poles, power, coords, attrs)

    def __getstate__(self):
        state = dict(
                    power=self.power.__getstate__(),
                    poles=self.poles.__getstate__() if self.poles is not None else None,
                    attrs=self.attrs)
        return state

    def __setstate__(self, state):
        self.attrs = state['attrs']
        self.power = BinnedStatistic.from_state(state['power'])
        if state['poles'] is not None:
            self.poles = BinnedStatistic.from_state(state['poles'])

    def _make_datasets(self, edges, poles, power, coords, attrs):

        if self.attrs['mode'] == '1d':
            power = BinnedStatistic(['k'], edges, power, fields_to_sum=['modes'], coords=coords, **attrs)
        else:
            power = BinnedStatistic(['k', 'mu'], edges, power, fields_to_sum=['modes'], coords=coords, **attrs)

        if poles is not None:
            poles = BinnedStatistic(['k'], [power.edges['k']], poles, fields_to_sum=['modes'], coords=[power.coords['k']], **attrs)

        return power, poles

class ProjectedFFTPower(FFTBase):
    """
    The power spectrum of a field in a periodic box, projected over certain axes.

    This is not really always physically meaningful, but convenient for
    making sense of Lyman-Alpha forest or lensing maps.

    This is usually called the 1d power spectrum or 2d power spectrum.

    Results are computed when the object is inititalized. See the documenation
    of :func:`~ProjectedFFTPower.run` for the attributes storing the results.

    Parameters
    ----------
    first : CatalogSource, MeshSource
        the source for the first field; if a CatalogSource is provided, it
        is automatically converted to MeshSource using the default painting
        parameters (via :func:`~nbodykit.base.catalogmesh.CatalogMesh.to_mesh`)
    Nmesh : int, optional
        the number of cells per side in the particle mesh used to paint the source
    BoxSize : int, 3-vector, optional
        the size of the box
    second : CatalogSource, MeshSource, optional
        the second source for cross-correlations
    axes : tuple
        axes to measure the power on. The axes not in the list will be averaged out.
        For example:
        - (0, 1) : project to x,y and measure power
        - (0) : project to x and measure power.
    dk : float, optional
        the linear spacing of ``k`` bins to use; if not provided, the
        fundamental mode  of the box is used
    kmin : float, optional
        the lower edge of the first ``k`` bin to use
    """
    logger = logging.getLogger('ProjectedFFTPower')

    def __init__(self, first, Nmesh=None, BoxSize=None, second=None,
                    axes=(0, 1), dk=None, kmin=0.):

        FFTBase.__init__(self, first, second, Nmesh, BoxSize)

        # only deal with 1d and 2d projections.
        assert len(axes) in (1, 2), "length of ``axes`` in ProjectedFFTPower should be 1 or 2"

        if dk is None:
            dk = 2 * numpy.pi / self.attrs['BoxSize'].min()

        self.attrs['dk'] = dk
        self.attrs['kmin'] = kmin

        self.attrs['axes'] = axes
        self.run()

    def run(self):
        """
        Run the algorithm. This attaches the following attributes to the class:

        - :attr:`edges`
        - :attr:`power`

        Attributes
        ----------
        edges : array_like
            the edges of the wavenumber bins
        power : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            a BinnedStatistic object that holds the projected power.
            It stores the following variables:

            - k :
                the mean value for each ``k`` bin
            - power :
                complex array holding the real and imaginary components of the
                projected power
            - modes :
                the number of Fourier modes averaged together in each bin
        """
        c1 = self.first.compute(Nmesh=self.attrs['Nmesh'], mode='complex')
        r1 = c1.preview(self.attrs['Nmesh'], axes=self.attrs['axes'])
        # average along projected axes;
        # part of product is the rfftn vs r2c (for axes)
        # the rest is for the mean (Nmesh - axes)
        c1 = numpy.fft.rfftn(r1) / self.attrs['Nmesh'].prod()

        # compute the auto power of single supplied field
        if self.first is self.second:
            c2 = c1
        else:
            c2 = self.second.compute(Nmesh=self.attrs['Nmesh'], mode='complex')
            r2 = c2.preview(self.attrs['Nmesh'], axes=self.attrs['axes'])
            c2 = numpy.fft.rfftn(r2) / self.attrs['Nmesh'].prod() # average along projected axes

        pk = c1 * c2.conj()
        # clear the zero mode
        pk.flat[0] = 0

        shape = numpy.array([self.attrs['Nmesh'][i] for i in self.attrs['axes']], dtype='int')
        boxsize = numpy.array([self.attrs['BoxSize'][i] for i in self.attrs['axes']])
        I = numpy.eye(len(shape), dtype='int') * -2 + 1

        k = [numpy.fft.fftfreq(N, 1. / (N * 2 * numpy.pi / L))[:pkshape].reshape(kshape) for N, L, kshape, pkshape in zip(shape, boxsize, I, pk.shape)]

        kmag = sum(ki ** 2 for ki in k) ** 0.5
        W = numpy.empty(pk.shape, dtype='f4')
        W[...] = 2.0
        W[..., 0] = 1.0
        W[..., -1] = 1.0

        dk = self.attrs['dk']
        kmin = self.attrs['kmin']
        axes = list(self.attrs['axes'])
        kedges = numpy.arange(kmin, numpy.pi * self.attrs['Nmesh'][axes].min() / self.attrs['BoxSize'][axes].max() + dk/2, dk)

        xsum = numpy.zeros(len(kedges) + 1)
        Psum = numpy.zeros(len(kedges) + 1, dtype='complex128')
        Nsum = numpy.zeros(len(kedges) + 1)

        dig = numpy.digitize(kmag.flat, kedges)
        xsum.flat += numpy.bincount(dig, weights=(W * kmag).flat, minlength=xsum.size)
        Psum.real.flat += numpy.bincount(dig, weights=(W * pk.real).flat, minlength=xsum.size)
        Psum.imag.flat += numpy.bincount(dig, weights=(W * pk.imag).flat, minlength=xsum.size)
        Nsum.flat += numpy.bincount(dig, weights=W.flat, minlength=xsum.size)

        self.power = numpy.empty(len(kedges) - 1,
                dtype=[('k', 'f8'), ('power', 'c16'), ('modes', 'f8')])

        with numpy.errstate(invalid='ignore', divide='ignore'):
            self.power['k'] = (xsum / Nsum)[1:-1]
            self.power['power'] = (Psum / Nsum)[1:-1] * boxsize.prod() # dimension is 'volume'
            self.power['modes'] = Nsum[1:-1]

        self.edges = kedges

        self.power = BinnedStatistic(['k'], [self.edges], self.power)

    def __getstate__(self):
        state = dict(
                     edges=self.edges,
                     power=self.power.data,
                     attrs=self.attrs)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.power = BinnedStatistic(['k'], [self.edges], self.power)

def project_to_basis(y3d, edges, los=[0, 0, 1], poles=[]):
    """
    Project a 3D statistic on to the specified basis. The basis will be one
    of:

    - 2D (`x`, `mu`) bins: `mu` is the cosine of the angle to the line-of-sight
    - 2D (`x`, `ell`) bins: `ell` is the multipole number, which specifies
      the Legendre polynomial when weighting different `mu` bins

    .. note::

        The 2D (`x`, `mu`) bins will be computed only if `poles` is specified.
        See return types for further details.

    Notes
    -----
    *   the `mu` range extends from 0.0 to 1.0
    *   the `mu` bins are half-inclusive half-exclusive, except the last bin
        is inclusive on both ends (to include `mu = 1.0`)

    Parameters
    ----------
    y3d : RealField or ComplexField
        the 3D array holding the statistic to be projected to the specified basis
    edges : list of arrays, (2,)
        list of arrays specifying the edges of the desired `x` bins and `mu` bins
    los : array_like,
        the line-of-sight direction to use, which `mu` is defined with
        respect to; default is [0, 0, 1] for z.
    poles : list of int, optional
        if provided, a list of integers specifying multipole numbers to
        project the 2d `(x, mu)` bins on to
    hermitian_symmetric : bool, optional
        Whether the input array `y3d` is Hermitian-symmetric, i.e., the negative
        frequency terms are just the complex conjugates of the corresponding
        positive-frequency terms; if ``True``, the positive frequency terms
        will be explicitly double-counted to account for this symmetry

    Returns
    -------
    result : tuple
        the 2D binned results; a tuple of ``(xmean_2d, mumean_2d, y2d, N_2d)``, where:

        - xmean_2d : array_like, (Nx, Nmu)
            the mean `x` value in each 2D bin
        - mumean_2d : array_like, (Nx, Nmu)
            the mean `mu` value in each 2D bin
        - y2d : array_like, (Nx, Nmu)
            the mean `y3d` value in each 2D bin
        - N_2d : array_like, (Nx, Nmu)
            the number of values averaged in each 2D bin

    pole_result : tuple or `None`
        the multipole results; if `poles` supplied it is a tuple of ``(xmean_1d, poles, N_1d)``,
        where:

        - xmean_1d : array_like, (Nx,)
            the mean `x` value in each 1D multipole bin
        - poles : array_like, (Nell, Nx)
            the mean multipoles value in each 1D bin
        - N_1d : array_like, (Nx,)
            the number of values averaged in each 1D bin
    """
    comm = y3d.pm.comm
    x3d = y3d.x
    hermitian_symmetric = numpy.iscomplexobj(y3d)

    from scipy.special import legendre

    # setup the bin edges and number of bins
    xedges, muedges = edges
    x2edges = xedges**2
    Nx = len(xedges) - 1
    Nmu = len(muedges) - 1

    # always make sure first ell value is monopole, which
    # is just (x, mu) projection since legendre of ell=0 is 1
    do_poles = len(poles) > 0
    _poles = [0]+sorted(poles) if 0 not in poles else sorted(poles)
    legpoly = [legendre(l) for l in _poles]
    ell_idx = [_poles.index(l) for l in poles]
    Nell = len(_poles)

    # valid ell values
    if any(ell < 0 for ell in _poles):
        raise ValueError("in `project_to_basis`, multipole numbers must be non-negative integers")

    # initialize the binning arrays
    musum = numpy.zeros((Nx+2, Nmu+2))
    xsum = numpy.zeros((Nx+2, Nmu+2))
    ysum = numpy.zeros((Nell, Nx+2, Nmu+2), dtype=y3d.dtype) # extra dimension for multipoles
    Nsum = numpy.zeros((Nx+2, Nmu+2), dtype='i8')

    # if input array is Hermitian symmetric, only half of the last
    # axis is stored in `y3d`
    symmetry_axis = -1 if hermitian_symmetric else None

    # iterate over y-z planes of the coordinate mesh
    for slab in SlabIterator(x3d, axis=0, symmetry_axis=symmetry_axis):

        # the square of coordinate mesh norm
        # (either Fourier space k or configuraton space x)
        xslab = slab.norm2()

        # if empty, do nothing
        if len(xslab.flat) == 0: continue

        # get the bin indices for x on the slab
        dig_x = numpy.digitize(xslab.flat, x2edges)

        # make xslab just x
        xslab **= 0.5

        # get the bin indices for mu on the slab
        mu = slab.mu(los) # defined with respect to specified LOS
        dig_mu = numpy.digitize(abs(mu).flat, muedges)

        # make the multi-index
        multi_index = numpy.ravel_multi_index([dig_x, dig_mu], (Nx+2,Nmu+2))

        # sum up x in each bin (accounting for negative freqs)
        xslab[:] *= slab.hermitian_weights
        xsum.flat += numpy.bincount(multi_index, weights=xslab.flat, minlength=xsum.size)

        # count number of modes in each bin (accounting for negative freqs)
        Nslab = numpy.ones_like(xslab) * slab.hermitian_weights
        Nsum.flat += numpy.bincount(multi_index, weights=Nslab.flat, minlength=Nsum.size)

        # compute multipoles by weighting by Legendre(ell, mu)
        for iell, ell in enumerate(_poles):

            # weight the input 3D array by the appropriate Legendre polynomial
            weighted_y3d = legpoly[iell](mu) * y3d[slab.index]

            # add conjugate for this kx, ky, kz, corresponding to
            # the (-kx, -ky, -kz) --> need to make mu negative for conjugate
            # Below is identical to the sum of
            # Leg(ell)(+mu) * y3d[:, nonsingular]    (kx, ky, kz)
            # Leg(ell)(-mu) * y3d[:, nonsingular].conj()  (-kx, -ky, -kz)
            # or
            # weighted_y3d[:, nonsingular] += (-1)**ell * weighted_y3d[:, nonsingular].conj()
            # but numerically more accurate.
            if hermitian_symmetric:

                if ell % 2: # odd, real part cancels
                    weighted_y3d.real[slab.nonsingular] = 0.
                    weighted_y3d.imag[slab.nonsingular] *= 2.
                else:  # even, imag part cancels
                    weighted_y3d.real[slab.nonsingular] *= 2.
                    weighted_y3d.imag[slab.nonsingular] = 0.

            # sum up the weighted y in each bin
            weighted_y3d *= (2.*ell + 1.)
            ysum[iell,...].real.flat += numpy.bincount(multi_index, weights=weighted_y3d.real.flat, minlength=Nsum.size)
            if numpy.iscomplexobj(ysum):
                ysum[iell,...].imag.flat += numpy.bincount(multi_index, weights=weighted_y3d.imag.flat, minlength=Nsum.size)

        # sum up the absolute mag of mu in each bin (accounting for negative freqs)
        mu[:] *= slab.hermitian_weights
        musum.flat += numpy.bincount(multi_index, weights=abs(mu).flat, minlength=musum.size)

    # sum binning arrays across all ranks
    xsum  = comm.allreduce(xsum)
    musum = comm.allreduce(musum)
    ysum  = comm.allreduce(ysum)
    Nsum  = comm.allreduce(Nsum)

    # add the last 'internal' mu bin (mu == 1) to the last visible mu bin
    # this makes the last visible mu bin inclusive on both ends.
    ysum[..., -2] += ysum[..., -1]
    musum[:, -2]  += musum[:, -1]
    xsum[:, -2]   += xsum[:, -1]
    Nsum[:, -2]   += Nsum[:, -1]

    # reshape and slice to remove out of bounds points
    sl = slice(1, -1)
    with numpy.errstate(invalid='ignore', divide='ignore'):

        # 2D binned results
        y2d       = (ysum[0,...] / Nsum)[sl,sl] # ell=0 is first index
        xmean_2d  = (xsum / Nsum)[sl,sl]
        mumean_2d = (musum / Nsum)[sl, sl]
        N_2d      = Nsum[sl,sl]

        # 1D multipole results (summing over mu (last) axis)
        if do_poles:
            N_1d     = Nsum[sl,sl].sum(axis=-1)
            xmean_1d = xsum[sl,sl].sum(axis=-1) / N_1d
            poles    = ysum[:, sl,sl].sum(axis=-1) / N_1d
            poles    = poles[ell_idx,...]

    # return y(x,mu) + (possibly empty) multipoles
    result = (xmean_2d, mumean_2d, y2d, N_2d)
    pole_result = (xmean_1d, poles, N_1d) if do_poles else None
    return result, pole_result

def _cast_source(source, BoxSize, Nmesh):
    """
    Cast an object to a MeshSource. BoxSize and Nmesh is used
    only on CatalogSource
    """
    from pmesh.pm import Field
    from nbodykit.source.mesh import FieldMesh

    if isinstance(source, Field):
        # if input is a Field object, wrap it as a MeshSource.
        source = FieldMesh(source)
    elif isinstance(source, CatalogSourceBase):
        # if input is CatalogSource, use defaults to make it into a mesh
        if not isinstance(source, MeshSource):
            source = source.to_mesh(BoxSize=BoxSize, Nmesh=Nmesh, dtype='f8', compensated=True)

    # now we are having a MeshSource
    # paint the density field to the mesh
    if not isinstance(source, MeshSource):
        raise TypeError("Unknown type of source in FFTPower: %s" % str(type(source)))
    if BoxSize is not None and any(source.attrs['BoxSize'] != BoxSize):
        raise ValueError("Mismatched Boxsize between __init__ and source.attrs")
    if Nmesh is not None and any(source.attrs['Nmesh'] != Nmesh):
        raise ValueError(("Mismatched Nmesh between __init__ and source.attrs; "
                          "if trying to re-sample with a different mesh, specify "
                          "`Nmesh` as keyword of to_mesh()"))

    return source

def _find_unique_edges(x, x0, comm):
    """ Construct unique edges based on x0.

        The modes along each direction are assumed to be multiples of x0

        Returns edges and the true centers
    """
    def find_unique(x, x0):
        fx2 = 0
        for xi, x0i in zip(x, x0):
            fx2 = fx2 + xi ** 2

        fx2 = numpy.ravel(fx2)
        ix2 = numpy.int64(fx2 / (x0.min() * 0.5) ** 2 + 0.5)
        ix2, ind = numpy.unique(ix2, return_index=True)
        fx2 = fx2[ind]
        return fx2 ** 0.5

    fx = find_unique(x, x0)

    fx = numpy.concatenate(comm.allgather(fx), axis=0)
    # may have duplicates after allgather
    fx = numpy.unique(fx)
    fx.sort()

    # now make some reasonable bins.
    width = numpy.diff(fx)
    edges = fx.copy()
    edges[1:] -= width * 0.5
    edges = numpy.append(edges, [fx[-1] + width[-1] * 0.5])
    edges[0] = 0

    # fx is the 'true' centers, up to round-off errors.
    return edges, fx
