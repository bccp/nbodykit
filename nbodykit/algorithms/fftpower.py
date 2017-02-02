import os
import numpy
import logging

from nbodykit import CurrentMPIComm
from nbodykit.dataset import DataSet
from nbodykit.meshtools import SlabIterator
from pmesh.pm import ComplexField

class FFTPower(object):
    """
    Algorithm to compute the 1d or 2d power spectrum and/or multipoles
    in a periodic box, using a Fast Fourier Transform (FFT)
    
    Notes
    -----
    The algorithm saves the power spectrum results to a plaintext file, 
    as well as the meta-data associted with the algorithm. The names of the
    columns saved to file are:
    
        - k : 
            the mean value for each `k` bin
        - mu : 2D power only
            the mean value for each `mu` bin
        - power.real, power.imag : 1D/2D power only
            the real and imaginary components of 1D power
        - power_X.real, power_X.imag : multipoles only
            the real and imaginary components for the `X` multipole
        - modes : 
            the number of Fourier modes averaged together in each bin
    
    The plaintext files also include meta-data associated with the algorithm:
    
        - Lx, Ly, Lz : 
            the length of each side of the box used when computing FFTs
        - volumne : 
            the volume of the box; equal to ``Lx*Ly*Lz``
        - N1 : 
            the number of objects in the 1st catalog
        - N2 : 
            the number of objects in the 2nd catalog; equal to `N1`
            if the power spectrum is an auto spectrum
    
    See :func:`nbodykit.files.Read1DPlainText`, :func:`nbodykit.files.Read2DPlainText`
    and :func:`nbodykit.dataset.Power1dDataSet.from_nbkit`
    :func:`nbodykit.dataset.Power2dDataSet.from_nbkit` for examples on how to read the
    the plaintext file.
    """
    logger = logging.getLogger('FFTPower')
    
    def __init__(self, first, mode, Nmesh=None, BoxSize=None, second=None, los=[0, 0, 1], Nmu=5, dk=None, kmin=0., poles=[]):
        """
        Parameters
        ----------
        comm : 
            the MPI communicator
        first : ParticleSource, MeshSource
            the source for the first field. ParticleSource is automatically converted
        mode : {'1d', '2d'}
            compute either 1d or 2d power spectra
        Nmesh : int
            the number of cells per side in the particle mesh used to paint the source
        second : ParticleSource, MeshSource; optional
            the second source for cross-correlations
        los : array_like ; optional
            the direction to use as the line-of-sight
        Nmu : int; optional
            the number of mu bins to use from mu=[0,1]; if `mode = 1d`, then `Nmu` is set to 1
        dk : float; optional
            the spacing of k bins to use; if not provided, the fundamental mode of the box is used
        kmin : float, optional
            the lower edge of the first ``k`` bin to use
        poles : list of int; optional
            a list of multipole numbers ``ell`` to compute :math:`P_\ell(k)` from :math:`P(k,\mu)`
        """
        from pmesh.pm import ParticleMesh
        from nbodykit.base.mesh import MeshSource
        
        # mode is either '1d' or '2d'
        if mode not in ['1d', '2d']:
            raise ValueError("`mode` should be either '1d' or '2d'")

        # grab comm from first source
        self.comm = first.comm 
        
        # if input is ParticleSource, use defaults to make it into a mesh
        if not hasattr(first, 'paint'):
            first = first.to_mesh(BoxSize=BoxSize, Nmesh=Nmesh, dtype='f8', compensated=True)
            
        # handle the second input source
        if second is None:
            second = first
        else:
            # make the second input a mesh if we need to
            if not hasattr(second, 'paint'):
                second = second.to_mesh(BoxSize=BoxSize, Nmesh=Nmesh, dtype='f8', compensated=True)
                
        # check for comm mismatch
        assert second.comm is first.comm, "communicator mismatch between input sources"
        
        self.sources = [first, second]
        assert all([isinstance(src, MeshSource) for src in self.sources]), 'error converting input sources to meshes'
        
        # using Nmesh from source
        if Nmesh is None:
            Nmesh = self.sources[0].attrs['Nmesh']

        _BoxSize = self.sources[0].attrs['BoxSize'].copy()
        if BoxSize is not None:
            _BoxSize[:] = BoxSize

        _Nmesh = self.sources[0].attrs['Nmesh'].copy()
        if _Nmesh is not None:
            _Nmesh[:] = Nmesh

        # check box sizes
        if len(self.sources) == 2:
            if not numpy.array_equal(self.sources[0].attrs['BoxSize'], self.sources[1].attrs['BoxSize']):
                raise ValueError("BoxSize mismatch between cross-correlation sources")
            if not numpy.array_equal(self.sources[0].attrs['BoxSize'], _BoxSize):
                raise ValueError("BoxSize mismatch between sources and the algorithm.")


        # setup the particle mesh object
        self.pm = ParticleMesh(BoxSize=_BoxSize, Nmesh=_Nmesh, dtype='f4', comm=self.comm)

        self.attrs = {}

        # save meta-data
        self.attrs['mode']    = mode
        self.attrs['los']     = los
        self.attrs['Nmu']     = Nmu
        self.attrs['dk']      = dk
        self.attrs['kmin']    = kmin 
        self.attrs['poles']   = poles
        self.attrs['Nmesh']   = self.pm.Nmesh.copy()
        self.attrs['BoxSize'] = self.pm.BoxSize.copy()

        # update the meta-data to return
        self.attrs.update(zip(['Lx', 'Ly', 'Lz'], _BoxSize))

        self.attrs.update({'volume':_BoxSize.prod()})

        self.run()

    def run(self):
        """
        Compute the power spectrum in a periodic box, using FFTs. This
        function returns nothing, but attaches several attributes
        to the class (see below).
        
        Attributes
        ----------
        edges : array_like
            the edges of the wavenumber bins
        power : :class:`~nbodykit.dataset.DataSet`
            a DataSet object that behaves similar to a structured array, with
            fancy slicing and re-indexing; it holds the measured :math:`P(k)` or 
            :math:`P(k,\mu)`
        poles : :class:`~nbodykit.dataset.DataSet` or ``None``
            a DataSet object to hold the multipole results :math:`P_\ell(k)`;
            if no multipoles were requested by the user, this is ``None``
        """

        # only need one mu bin if 1d case is requested
        if self.attrs['mode'] == "1d": self.attrs['Nmu'] = 1

        # measure the 3D power (y3d is a ComplexField)
        y3d = self._compute_3d_power()

        # binning in k out to the minimum nyquist frequency 
        # (accounting for possibly anisotropic box)
        dk = 2*numpy.pi/y3d.BoxSize.min() if self.attrs['dk'] is None else self.attrs['dk']
        kedges = numpy.arange(self.attrs['kmin'], numpy.pi*y3d.Nmesh.min()/y3d.BoxSize.max() + dk/2, dk)

        # project on to the desired basis
        muedges = numpy.linspace(0, 1, self.attrs['Nmu']+1, endpoint=True)
        edges = [kedges, muedges]
        result, pole_result = project_to_basis(y3d, edges, 
                                               poles=self.attrs['poles'], 
                                               los=self.attrs['los'])

        # format the power results into structured array
        if self.attrs['mode'] == "1d":
            cols = ['k', 'power', 'modes']
            icols = [0, 2, 3]
            edges = edges[0]
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

        # set all the necessary results
        self.edges = edges
        self.poles = poles
        self.power = power
        
        self._make_datasets()
    
    def __getstate__(self):
        state = dict(
                     edges=self.edges,
                     power=self.power.data,
                     poles=getattr(self.poles, 'data', None),
                     attrs=self.attrs)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._make_datasets()
        
    def save(self, output):
        """
        Save the FFTPower result to disk.

        The format is currently JSON.
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
        Load a saved FFTPower result.

        The result has been saved to disk with :func:`FFTPower.save`.
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

        return self

    def _make_datasets(self):
        
        if self.attrs['mode'] == '1d':
            self.power = DataSet(['k'], [self.edges], self.power, fields_to_sum=['modes'])
        else:
            self.power = DataSet(['k', 'mu'], self.edges, self.power, fields_to_sum=['modes'])
        if self.poles is not None:
            self.poles = DataSet(['k'], [self.power.edges['k']], self.poles, fields_to_sum=['modes'])
            
    def _source2field(self, source):

        # step 1: paint the density field to the mesh
        c = source.paint(mode='complex')

        if self.comm.rank == 0: self.logger.info('field: %s painting done' % str(source))

        if any(c.pm.Nmesh != self.pm.Nmesh):
            cnew = ComplexField(self.pm)
            c = c.resample(out=cnew)

            if self.comm.rank == 0: self.logger.info('field: %s resampling done' % str(source))

        return c

    def _compute_3d_power(self):
        """
        Compute and return the 3D power from two input sources

        Parameters
        ----------
        sources : list of ParticleSource or MeshSource
            the list of sources which the 3D power will be computed
        pm : ParticleMesh
            the particle mesh object that handles the painting and FFTs
        comm : MPI.Communicator, optional
            the communicator to pass to the ParticleMesh object. If not
            provided, ``MPI.COMM_WORLD`` is used

        Returns
        -------
        p3d : array_like (complex)
            the 3D complex array holding the power spectrum
        """
        sources = self.sources
        pm = self.pm
        comm = self.comm

        rank = comm.rank

        c1 = self._source2field(self.sources[0])

        # compute the auto power of single supplied field
        if sources[0] is sources[1]:
            c2 = c1
        else:
            c2 = self._source2field(self.sources[1])

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
        self.attrs.update({'N1':N1, 'N2':N2})
        
        # add shotnoise (nonzero only for auto-spectra)
        Pshot = 0
        if sources[0] is sources[1] and N1 > 0:
            Pshot = self.attrs['BoxSize'].prod() / N1
        self.attrs['shotnoise'] = Pshot

        return p3d

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
        
        xmean_2d : array_like, (Nx, Nmu)
            the mean `x` value in each 2D bin
        mumean_2d : array_like, (Nx, Nmu)
            the mean `mu` value in each 2D bin
        y2d : array_like, (Nx, Nmu)
            the mean `y3d` value in each 2D bin
        N_2d : array_like, (Nx, Nmu)
            the number of values averaged in each 2D bin
    
    pole_result : tuple or `None`
        the multipole results; if `poles` supplied it is a tuple of ``(xmean_1d, poles, N_1d)``, 
        where:
    
        xmean_1d : array_like, (Nx,)
            the mean `x` value in each 1D multipole bin
        poles : array_like, (Nell, Nx)
            the mean multipoles value in each 1D bin
        N_1d : array_like, (Nx,)
            the number of values averaged in each 1D bin
    """
    comm = y3d.pm.comm
    x3d = y3d.x
    hermitian_symmetric = isinstance(y3d, ComplexField)

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
    with numpy.errstate(invalid='ignore'):
        
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

