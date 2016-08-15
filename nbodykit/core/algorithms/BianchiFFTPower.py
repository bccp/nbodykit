from nbodykit.core import Algorithm, DataSource
from nbodykit.fkp import FKPCatalog
import numpy

class BianchiPowerAlgorithm(Algorithm):
    """
    Algorithm to compute the power spectrum multipoles using FFTs
    for a data survey with non-trivial geometry
    
    The algorithm used to compute the multipoles is detailed
    in Bianchi et al. 2015 (http://adsabs.harvard.edu/abs/2015MNRAS.453L..11B)
    
    
    Notes
    -----
    The algorithm saves the power spectrum result to a plaintext file, 
    as well as the meta-data associted with the algorithm.
    
    The columns names are:
    
        - k : 
            the mean value for each `k` bin
        - power_X.real, power_X.imag : multipoles only
            the real and imaginary components for the `X` multipole
        - modes : 
            the number of Fourier modes averaged together in each bin
    
    The plaintext files also include meta-data associated with the algorithm:
    
        - Lx, Ly, Lz : 
            the length of each side of the box used when computing FFTs
        - volumne : 
            the volume of the box; equal to ``Lx*Ly*Lz``
        - N_data : 
            the number of objects in the "data" catalog
        - N_ran : 
            the number of objects in the "randoms" catalog
        - alpha : 
            the ratio of data to randoms; equal to ``N_data/N_ran``
        - S_data : 
            the unnormalized shot noise for the "data" catalog; see
            equations 13-15 of `Beutler et al. 2014 <http://arxiv.org/abs/1312.4611>`_
        - S_ran : 
            the same as `S_data`, but for the "randoms" catalog
        - A_data : 
            the power spectrum normalization, as computed from the "data" catalog; 
            see equations 13-15 of `Beutler et al. 2014 <http://arxiv.org/abs/1312.4611>`_
            for further details
        - A_ran : 
            the same as `A_data`, but for the "randoms" catalog; this is the 
            actual value used to normalize the power spectrum, but its value
            should be very close to `A_data`
        - shot_noise : 
            the final shot noise for the monopole; equal to ``(S_ran + S_data)/A_ran``
    
    See :func:`nbodykit.files.Read1DPlainText` and 
    :func:`nbodykit.dataset.Power1dDataSet.from_nbkit` for examples on how to read the
    the plaintext file.
    """
    plugin_name = "BianchiFFTPower"

    def __init__(self, data, randoms, Nmesh, max_ell, 
                    paintbrush='cic',
                    dk=None, 
                    kmin=0., 
                    BoxSize=None, 
                    BoxPad=0.02, 
                    compute_fkp_weights=False, 
                    P0_fkp=None, 
                    nbar=None, 
                    fsky=None,
                    factor_hexadecapole=False,
                    keep_cache=False):
        
        # positional arguments
        self.data    = data
        self.randoms = randoms
        self.Nmesh   = Nmesh
        self.max_ell = max_ell
        
        # keyword arguments
        self.paintbrush          = paintbrush
        self.dk                  = dk
        self.kmin                = kmin
        self.BoxSize             = BoxSize
        self.BoxPad              = BoxPad
        self.compute_fkp_weights = compute_fkp_weights
        self.P0_fkp              = P0_fkp
        self.nbar                = nbar
        self.fsky                = fsky
        self.factor_hexadecapole = factor_hexadecapole
        self.keep_cache          = keep_cache
        
        # initialize the FKP catalog (unopened)
        kws = {}
        kws['BoxSize'] = self.BoxSize
        kws['BoxPad'] = self.BoxPad
        kws['compute_fkp_weights'] = self.compute_fkp_weights
        kws['P0_fkp'] = self.P0_fkp
        kws['nbar'] = self.nbar
        kws['fsky'] = self.fsky
        self.catalog = FKPCatalog(self.data, self.randoms, **kws)
        
    @classmethod
    def fill_schema(cls):
        
        s = cls.schema
        s.description = "power spectrum multipoles using FFTs for a data survey with \n"
        s.description += "non-trivial geometry, as detailed in Bianchi et al. 2015 (1505.05341)"

        # the required arguments
        s.add_argument("data", type=DataSource.from_config,
            help="DataSource representing the `data` catalog")
        s.add_argument("randoms", type=DataSource.from_config,
            help="DataSource representing the `randoms` catalog")        
        s.add_argument("Nmesh", type=int,
            help='the number of cells in the gridded mesh (per axis)')
        s.add_argument('max_ell', type=int, choices=[0,2,4],
            help='compute multipoles up to and including this ell value')

        # the optional arguments
        s.add_argument("BoxSize", type=DataSource.BoxSizeParser,
            help="the size of the box; if not provided, automatically computed from the `randoms` catalog")
        s.add_argument('BoxPad', type=float,
            help='when setting the box size automatically, apply this additional buffer')
        s.add_argument('compute_fkp_weights', type=bool,
            help='if set, use FKP weights, computed from `P0_fkp` and the provided `nbar`') 
        s.add_argument('P0_fkp', type=float,
            help='the fiducial power value `P0` used to compute FKP weights')
        s.add_argument('nbar', type=str,
            help='read `nbar(z)` from this file, which provides two columns (z, nbar)')
        s.add_argument('fsky', type=float, 
            help='the sky area fraction of the tracer catalog, used in the volume calculation of `nbar`')
        
        s.add_argument('paintbrush', type=lambda x: x.lower(), choices=['cic', 'tsc'],
            help='the density assignment kernel to use when painting; '
                 'CIC (2nd order) or TSC (3rd order)')
        s.add_argument("dk", type=float,
            help='the spacing of k bins to use; if not provided, '
                 'the fundamental mode of the box is used')
        s.add_argument("kmin", type=float,
            help='the edge of the first `k` bin to use; default is 0')
        s.add_argument('factor_hexadecapole', type=bool, 
            help="use the factored expression for the hexadecapole (ell=4) from "
                 "eq. 27 of Scoccimarro 2015 (1506.02729)")
        s.add_argument('keep_cache', type=bool, 
            help='if `True`, force the data cache to persist while the algorithm instance is valid')
                                
    def run(self):
        """
        Run the algorithm, which computes and returns the power spectrum
        """
        from nbodykit import measurestats
        if self.comm.rank == 0: self.logger.info('importing done')
        
        # explicity store an open stream
        # this prevents the cache from being destroyed while the 
        # algorithm instance is active
        if self.keep_cache:
            self._datacache = self.catalog.data.keep_cache()
            self._rancache = self.catalog.randoms.keep_cache()
        
        # measure
        kws = {'factor_hexadecapole': self.factor_hexadecapole, 'paintbrush':self.paintbrush}
        pm, poles, meta = measurestats.compute_bianchi_poles(self.comm, self.max_ell, self.catalog, self.Nmesh, **kws)
        k3d = pm.k

        # binning in k out to the minimum nyquist frequency
        # (accounting for possibly anisotropic box)
        dk = 2*numpy.pi/pm.BoxSize.min() if self.dk is None else self.dk
        kedges = numpy.arange(self.kmin, numpy.pi*pm.Nmesh.min()/pm.BoxSize.max() + dk/2, dk)

        # project on to 1d k basis
        muedges = numpy.linspace(0, 1, 2, endpoint=True)
        edges = [kedges, muedges]
        poles_final = []
        for p in poles:
            
            # result is (k, mu, power, modes)
            result, _ = measurestats.project_to_basis(pm.comm, k3d, p, edges, hermitian_symmetric=True)
            poles_final.append(numpy.squeeze(result[2]))
            
        # return (k, poles, modes)
        poles_final = numpy.vstack(poles_final)
        k = numpy.squeeze(result[0])
        modes = numpy.squeeze(result[-1])
        result = k, poles_final, modes

        # compute the metadata to return
        Lx, Ly, Lz = pm.BoxSize
        meta.update({'Lx':Lx, 'Ly':Ly, 'Lz':Lz, 'volume':Lx*Ly*Lz})

        # return all the necessary results
        return kedges, result, meta

    def save(self, output, result):
        """
        Save the power spectrum results to the specified output file

        Parameters
        ----------
        output : str
            the string specifying the file to save
        result : tuple
            the tuple returned by `run()` -- first argument specifies the bin
            edges and the second is a dictionary holding the data results
        """
        from nbodykit.storage import MeasurementStorage
        
        # only the master rank writes
        if self.comm.rank == 0:
            
            kedges, result, meta = result
            k, poles, N = result
            ells = range(0, self.max_ell+1, 2)
            
            # write binned statistic
            args = (",".join(map(str, ells)), output)
            self.logger.info('measurement done; saving ell = %s multipole(s) to %s' %args)
            cols = ['k'] + ['power_%d' %l for l in ells] + ['modes']
            pole_result = [k] + [pole for pole in poles] + [N]
            
            storage = MeasurementStorage.create('1d', output)
            storage.write(kedges, cols, pole_result, **meta)

