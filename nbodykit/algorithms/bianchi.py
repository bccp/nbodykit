import os
import numpy
import logging
import time

from nbodykit import CurrentMPIComm
from .fftpower import project_to_basis
from pmesh.pm import ComplexField

def timer(start, end):
    """
    Utility function to return a string representing the elapsed time, 
    as computed from the input start and end times
    
    Parameters
    ----------
    start : int
        the start time in seconds
    end : int
        the end time in seconds
    
    Returns
    -------
    str : 
        the elapsed time as a string, using the format `hours:minutes:seconds`
    """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

def BianchiKernel(x, v, i, j, k=None, offset=[0.,0.,0.]):
    """
    Apply coordinate kernels to ``data`` necessary to compute the 
    power spectrum multipoles via FFTs using the algorithm 
    detailed in Bianchi et al. 2015.
    
    This multiplies by one of two kernels:
    
        1. x_i * x_j / x**2 * data, if `k` is None
        2. x_i**2 * x_j * x_k / x**4 * data, if `k` is not None
    
    See equations 10 (for quadrupole) and 12 (for hexadecapole)
    of Bianchi et al 2015.
    
    Parameters
    ----------
    data : array_like
        the array to rescale -- either the configuration-space 
        `pm.real` or the Fourier-space `pm.complex`
    x : array_like
        the coordinate array -- either `pm.r` or `pm.k`
    i, j, k : int
        the integers specifying the coordinate axes; see the 
        above description 
    """   
    # add any offsets to the input coordinate array
    x = [xx.copy() + offset[kk] for kk,xx in enumerate(x)] 
    
    # normalization is norm squared of coordinate mesh
    norm = sum(xx**2 for xx in x)

    # get coordinate arrays for indices i, j         
    xi = x[i]
    if j == i: xj = xi
    else: xj = x[j]
        
    # handle third index j
    if k is not None:
        
        # get coordinate array for index k
        if k == i: xk = xi
        elif k == j: xk = xj
        else: xk = x[k]

        # weight data by xi**2 * xj * xj 
        with numpy.errstate(invalid='ignore'):
            v = v * xi**2 * xj * xk / norm**2
        v[norm==0] = 0.
    else:
        # weight data by xi * xj
        with numpy.errstate(invalid='ignore'):
            v = v * xi * xj / norm
        v[norm==0] = 0.
        
    return v
     
class BianchiFFTPower(object):
    """
    Algorithm to compute the power spectrum multipoles using FFTs
    for a data survey with non-trivial geometry
    
    The algorithm used to compute the multipoles is detailed
    in Bianchi et al. 2015 (http://adsabs.harvard.edu/abs/2015MNRAS.453L..11B)
    """
    logger = logging.getLogger('BianchiFFTPower')

    def __init__(self, source, max_ell, Nmesh=None, 
                    kmin=0., 
                    dk=None, 
                    use_fkp_weights=False, 
                    P0_fkp=None, 
                    factor_hexadecapole=False,
                    paintbrush='cic'):
        """
        Parameters
        ----------
        
        """
        if not hasattr(source, 'paint'):
            source = source.to_mesh(Nmesh=Nmesh)
        self.source  = source
        self.Nmesh   = Nmesh
        self.comm = self.source.comm
        
        if max_ell not in [0, 2, 4]:
            raise ValueError("valid values for the maximum multipole number are [0, 2, 4]")
        
        self.attrs = {}
        self.attrs['max_ell']             = max_ell
        self.attrs['dk']                  = dk
        self.attrs['kmin']                = kmin
        self.attrs['use_fkp_weights']     = use_fkp_weights
        self.attrs['P0_fkp']              = P0_fkp
        self.attrs['factor_hexadecapole'] = factor_hexadecapole
        self.attrs.update(self.source.attrs)

        
    def _compute_multipoles(self):
        """
        Use the algorithm detailed in Bianchi et al. 2015 to compute and return the 3D 
        power spectrum multipoles (`ell = [0, 2, 4]`) from one input field, which contains 
        non-trivial survey geometry.
    
        The estimator uses the FFT algorithm outlined in Bianchi et al. 2015
        (http://adsabs.harvard.edu/abs/2015arXiv150505341B) to compute
        the monopole, quadrupole, and hexadecapole

        Returns
        -------
        pm : ParticleMesh
            the mesh object used to do painting, FFTs, etc
        result : list of arrays
            list of 3D complex arrays holding power spectrum multipoles; respectively, 
            if `ell_max=0,2,4`, the list holds the monopole only, monopole and quadrupole, 
            or the monopole, quadrupole, and hexadecapole
        stats : dict
            dict holding the statistics of the input fields, as returned
            by the `FKPPainter` painter
    
        References
        ----------
        * Bianchi, Davide et al., `Measuring line-of-sight-dependent Fourier-space clustering using FFTs`,
          MNRAS, 2015
        * Scoccimarro, Roman, `Fast estimators for redshift-space clustering`, Phys. Review D, 2015
        """
        rank                = self.comm.rank
        max_ell             = self.attrs['max_ell']
        factor_hexadecapole = self.attrs['factor_hexadecapole']
        
        pm     = self.source.pm
        offset = self.attrs['BoxCenter'] + 0.5*pm.BoxSize / pm.Nmesh
            
        # initialize the compensation transfer
        compensation = None
        try:
            compensation = self.source._get_compensation()
            self.logger.info('using compensation function %s' %compensation[0][1].__name__)
        except ValueError as e:
            self.logger.warning('no compensation applied: %s' %str(e))
            
        # determine kernels needed to compute ell=2,4
        bianchi_transfers = {}
        for ell in range(2, max_ell+1, 2):
            
            if ell == 2:
                integers = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

                amps = [1.]*3 + [2.]*3
            elif ell == 4 and not factor_hexadecapole:
                integers = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 0, 1), (0, 0, 2),
                            (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 1), (0, 1, 1),
                            (0, 2, 2), (1, 2, 2), (0, 1, 2), (1, 0, 2), (2, 0, 1)]
                amps = [1.]*3 + [4.]*6 + [6.]*3 + [12.]*3
            bianchi_transfers[ell] = (amps, integers)
                
        # paint the FKP density field to the mesh (paints: data - alpha*randoms, essentially)
        rfield = self.source.paint(mode='real')

        # save the painted density field for later
        density = rfield.copy()
        if rank == 0: self.logger.info('%s painting done' %self.source.window)
    
        # FFT density field and apply the paintbrush window transfer kernel
        cfield = rfield.r2c()
        if compensation is not None:
            cfield.apply(func=compensation[0][1], kind=compensation[0][2], out=Ellipsis)
        if rank == 0: self.logger.info('ell = 0 done; 1 r2c completed')
        
        # monopole A0 is just the FFT of the FKP density field
        volume = pm.BoxSize.prod()
        A0 = ComplexField(pm)
        A0[:] = cfield[:] * volume # normalize with a factor of volume
    
        # store the A0, A2, A4 arrays here
        result = []
        result.append(A0); del A0
    
        xgrid = [x + offset[ii] for ii, x in enumerate(pm.x)]
        
        # loop over the higher order multipoles (ell > 0)
        start = time.time()
        for ell in range(2, max_ell+1, 2):
        
            # temporary array to hold sum of all of the terms in Fourier space
            Aell_sum = ComplexField(pm)
            Aell_sum[:] = 0.
                                
            # loop over each kernel term for this multipole
            for amp, integers in zip(*bianchi_transfers[ell]):
                
                # reset the realspace mesh to the original FKP density
                rfield[:] = density[:]        
                
                # apply the config-space kernel
                func = lambda x,v: BianchiKernel(x, v, *integers, offset=offset)
                rfield.apply(func=func, kind='relative', out=Ellipsis)

                # real to complex
                rfield.r2c(out=cfield)

                # apply the Fourier-space kernel
                func = lambda x,v: BianchiKernel(x, v, *integers)
                cfield.apply(func=func, kind='wavenumber', out=Ellipsis)        
                
                # and this contribution to the total sum
                Aell_sum[:] += amp*volume*cfield[:]
                if rank == 0:
                    self.logger.debug("done Bianchi term for %s..." %str(integers))

            # apply the window transfer function and save
            if compensation is not None:
                Aell_sum.apply(func=compensation[0][1], kind=compensation[0][2], out=Ellipsis)
            result.append(Aell_sum); del Aell_sum # delete temp array since appending to list makes copy
        
            # log the total number of FFTs computed for each ell
            if rank == 0: 
                args = (ell, len(bianchi_transfers[ell][0]))
                self.logger.info('ell = %d done; %s r2c completed' %args)
        
        # density array no longer needed
        del density
    
        # summarize how long it took
        stop = time.time()
        if rank == 0:
            self.logger.info("higher order multipoles computed in elapsed time %s" %timer(start, stop))
            if factor_hexadecapole:
                self.logger.info("using factorized hexadecapole estimator for ell=4")
    
        # proper normalization: same as equation 49 of Scoccimarro et al. 2015 
        self.logger.info("normalizing power spectrum with A_ran = %.6f" %self.source.attrs['A_ran'])
        norm = 1.0 / self.source.attrs['A_ran']
    
        # reuse memory for output arrays
        P0 = result[0]
        if max_ell > 0: 
            P2 = result[1]
        if max_ell > 2:
            P4 = ComplexField(pm) if factor_hexadecapole else result[2]
        
        # calculate the power spectrum multipoles, slab-by-slab to save memory
        for islab in range(P0.shape[0]):

            # save arrays for reuse
            P0_star = (P0[islab]).conj()
            if max_ell > 0: P2_star = (P2[islab]).conj()

            # hexadecapole
            if max_ell > 2:
            
                # see equation 8 of Bianchi et al. 2015
                if not factor_hexadecapole:
                    P4[islab, ...] = norm * 9./8. * P0[islab] * (35.*(P4[islab]).conj() - 30.*P2_star + 3.*P0_star)
                # see equation 48 of Scoccimarro et al; 2015
                else:
                    P4[islab, ...] = norm * 9./8. * ( 35.*P2[islab]*P2_star + 3.*P0[islab]*P0_star - 5./3.*(11.*P0[islab]*P2_star + 7.*P2[islab]*P0_star) )
        
            # quadrupole: equation 7 of Bianchi et al. 2015
            if max_ell > 0:
                P2[islab, ...] = norm * 5./2. * P0[islab] * (3.*P2_star - P0_star)

            # monopole: equation 6 of Bianchi et al. 2015
            P0[islab, ...] = norm * P0[islab] * P0_star
        
        return result
        
    def run(self):
        """
        Compute the power spectrum multipoles
        
        This function does not return anything, but addes 
        """
        pm = self.source.pm

        # measure the 3D multipoles in Fourier space
        poles = self._compute_multipoles()
        k3d = pm.k

        # setup the binning in k out to the minimum nyquist frequency
        dk = 2*numpy.pi/pm.BoxSize.min() if self.attrs['dk'] is None else self.attrs['dk']
        kedges = numpy.arange(self.attrs['kmin'], numpy.pi*pm.Nmesh.min()/pm.BoxSize.max() + dk/2, dk)

        # project on to 1d k-basis (averaging over mu=[0,1])
        muedges = numpy.linspace(0, 1, 2, endpoint=True)
        edges = [kedges, muedges]
        poles_final = []
        for p in poles:
            result, _ = project_to_basis(p, edges)
            poles_final.append(numpy.squeeze(result[2]))

        # format (k, poles, modes)
        poles = numpy.vstack(poles_final)
        k = numpy.squeeze(result[0])
        N = numpy.squeeze(result[-1])
        
        # make a structured array holding the results
        cols = ['k'] + ['power_%d' %l for l in range(0, self.attrs['max_ell']+1,2)] + ['modes']
        result = [k] + [pole for pole in poles] + [N]
        dtype = numpy.dtype([(name, result[icol].dtype.str) for icol,name in enumerate(cols)])
        poles = numpy.empty(result[0].shape, dtype=dtype)
        for icol, col in enumerate(cols):
            poles[col][:] = result[icol]
        
        # set all the necessary results
        self.edges = kedges
        self.poles = poles # structured array holding the result
    
    def __getstate__(self):
        state = dict(edges=self.edges,
                     poles=self.poles,
                     attrs=self.attrs)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, output):
        """ 
        Save the BianchiFFTPower result to disk.

        The format is currently json.
        
        Parameters
        ----------
        output : str
            the name of the file to dump the JSON results to
        """
        import json
        from nbodykit.utils import JSONEncoder
        
        # only the master rank writes
        if self.comm.rank == 0:
            self.logger.info('saving BianchiFFTPower result to %s' %output)

            with open(output, 'w') as ff:
                json.dump(self.__getstate__(), ff, cls=JSONEncoder)
    
    @classmethod
    @CurrentMPIComm.enable
    def load(cls, output, comm=None):
        """
        Load a saved BianchiFFTPower result, which has been saved to 
        disk with :func:`BianchiFFTPower.save`
        
        The current MPI communicator is automatically used
        if the ``comm`` keyword is ``None``
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
