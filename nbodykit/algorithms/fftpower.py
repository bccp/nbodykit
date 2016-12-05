import os
import numpy
import logging

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
    
    def __init__(self, comm, first, mode, Nmesh, second=None, los='z', Nmu=5, dk=None, kmin=0., poles=[]):
        """
        Parameters
        ----------
        comm : 
            the MPI communicator
        first : ParticleSource, GridSource
            the source for the first field
        mode : {'1d', '2d'}
            compute either 1d or 2d power spectra
        Nmesh : int
            the number of cells per side in the particle mesh used to paint the source
        second : ParticleSource, GridSource; optional
            the second source for cross-correlations
        loz : {'x', 'y', 'z'}; optional
            the axis of the box to use as the line-of-sight
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
        
        # check inputs
        if mode not in ['1d', '2d']:
            raise ValueError("`mode` should be either '1d' or '2d'")
        if los not in 'xyz':
            raise ValueError("`los` should be one of 'x','y','z'")
        
        self.comm = comm 
                   
        # save meta-data
        self.attrs['mode']  = mode
        self.attrs['Nmesh'] = Nmesh
        self.attrs['los']   = los
        self.attrs['Nmu']   = Nmu
        self.attrs['dk']    = dk
        self.attrs['kmin']  = kmin 
        self.attrs['poles'] = poles
        
        # combine the two sources
        self.fields = [first]
        if second is not None:
            self.fields.append(second)

        # check box sizes
        if len(self.fields) == 2:
            if not numpy.array_equal(self.fields[0].BoxSize , self.fields[1].BoxSize):
                raise ValueError("BoxSize mismatch between cross-correlation sources")

        # setup the particle mesh object, taking BoxSize from the painters
        self.pm = ParticleMesh(BoxSize=self.fields[0].BoxSize, Nmesh=[self.attrs['Nmesh']]*3,
                                dtype='f4', comm=self.comm)

    @property
    def attrs(self):
        """
        Dictionary storing relevant meta-data
        """
        try:
            return self._attrs
        except AttributeError:
            self._attrs = {}
            return self._attrs
            
    @property
    def transfers(self):
        """
        A list of callables to apply to the Fourier transform of the real-space
        density field for each source in :attr:`fields`
        
        The function signature of these functions should be
        ``fk(k, kx, ky, kz)``, where ``k`` is the norm of the wavenumber
        and ``kx``, ``ky``, and ``kz`` are the individual components
        """
        try:
            return self._transfers
        except AttributeError:
            from nbodykit import transfers as tf
            
            self._transfers = []
            for i, field in enumerate(self.fields):
                t = [tf.NormalizeDC, tf.RemoveDC]
                brush = field.painter.paintbrush.upper()
                if brush in ['TSC', 'CIC']:
                    if not field.painter.interlaced:
                        t.append(getattr(tf, "%sAliasingWindow" %brush))
                    else:
                        t.append(getattr(tf, "%sWindow" %brush))
                self._transfers.append(t)
                
            return self._transfers
            
    def set_transfers(self, transfers):
        """
        Set the transfer functions applied to the Fourier-space field
        """
        self._transfers = transfers
    
    def run(self):
        """
        Run the algorithm, which computes and returns the power spectrum
        """
        from nbodykit import measurestats

        # only need one mu bin if 1d case is requested
        if self.attrs['mode'] == "1d": self.attrs['Nmu'] = 1

        # measure the 3D power (y3d is a ComplexField)
        y3d = measurestats.compute_3d_power(self.fields, self.pm, transfers=self.transfers, comm=self.comm)

        # get the number of objects (in a safe manner)
        N1 = len(self.fields[0])
        N2 = len(self.fields[1]) if len(self.fields) == 2 else N1

        # binning in k out to the minimum nyquist frequency 
        # (accounting for possibly anisotropic box)
        dk = 2*numpy.pi/y3d.BoxSize.min() if self.attrs['dk'] is None else self.attrs['dk']
        kedges = numpy.arange(self.attrs['kmin'], numpy.pi*y3d.Nmesh.min()/y3d.BoxSize.max() + dk/2, dk)

        # project on to the desired basis
        muedges = numpy.linspace(0, 1, self.attrs['Nmu']+1, endpoint=True)
        edges = [kedges, muedges]
        result, pole_result = measurestats.project_to_basis(self.comm, y3d.x, y3d, edges, 
                                                            poles=self.attrs['poles'], 
                                                            los=self.attrs['los'],
                                                            hermitian_symmetric=True)

        # update the meta-data to return
        Lx, Ly, Lz = y3d.BoxSize
        self.attrs.update({'Lx':Lx, 'Ly':Ly, 'Lz':Lz, 'volume':Lx*Ly*Lz, 'N1':N1, 'N2':N2})

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
        power = numpy.empty(result[0].shape, dtype=dtype)
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
    
        # return all the necessary results
        return edges, power, poles

    def save(self, output, **results):
        """
        Save the power spectrum results to the specified output file

        Parameters
        ----------
        output : str
            the string specifying the file to save
        data : array_like
            the tuple returned by `run()` -- first argument specifies the bin
            edges and the second is a dictionary holding the data results
        """
        # only the master rank writes
        if self.comm.rank == 0:
            import pickle
            
            self.logger.info('measurement done; saving result to %s' %output)
            if 'attrs'not in results:
                results['attrs'] = self.attrs
            
            with open(output, 'wb') as ff:
                pickle.dump(results, ff) 