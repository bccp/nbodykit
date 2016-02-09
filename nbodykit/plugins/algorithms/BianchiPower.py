from nbodykit.extensionpoints import Algorithm, plugin_isinstance
from nbodykit.extensionpoints import DataSource, Transfer, Painter
from nbodykit.plugins import add_plugin_list_argument

import numpy
import logging
    
class BianchiPowerAlgorithm(Algorithm):
    """
    Algorithm to compute the power spectrum multipoles using FFTs 
    for a data survey with non-trivial geometry
    """
    plugin_name = "BianchiPower"
    logger = logging.getLogger(plugin_name)
    
    @classmethod
    def register(kls):
        p = kls.parser
        p.description = "galaxy survey power spectrum multipoles calculator via FFT"

        # the required arguments
        p.add_argument('poles', type=lambda s: [int(i) for i in s.split()],
            help='the multipoles to compute; values must be in [0,2,4]')
        p.add_argument("Nmesh", type=int, 
            help='the number of cells in the gridded mesh')
        p.add_argument("data", type=DataSource.fromstring,
            help='the `DataSource` specifiying the data catalog to read')
        p.add_argument("randoms", type=DataSource.fromstring,
            help='the `DataSource` specifiying the randoms catalog to read')
        
        # the optional arguments
        p.add_argument("--dk", type=float,
            help='the spacing of k bins to use; if not provided, the fundamental mode of the box is used')
        p.add_argument("--kmin", type=float, default=0.,
            help='the edge of the first `k` bin to use; default is 0')
        p.add_argument('-q', '--quiet', action="store_const", dest="log_level", 
            default=logging.DEBUG, help="silence the logging output", const=logging.ERROR)

            
    def run(self):
        """
        Run the algorithm, which computes and returns the power spectrum
        """
        from nbodykit import measurestats
        from pmesh.particlemesh import ParticleMesh
        
        self.logger.setLevel(self.log_level)
        if self.comm.rank == 0: self.logger.info('importing done')
        
        # setup the particle mesh object, taking BoxSize from the painters
        pm = ParticleMesh(self.fields[0][0].BoxSize, self.Nmesh, dtype='f4', comm=self.comm)

        # only need one mu bin if 1d case is requested
        if self.mode == "1d": self.Nmu = 1
    
        # measure
        y3d, N1, N2 = measurestats.compute_3d_power(self.fields, pm, comm=self.comm, log_level=self.log_level)
        x3d = pm.k
    
        # binning in k out to the minimum nyquist frequency 
        # (accounting for possibly anisotropic box)
        dk = 2*numpy.pi/pm.BoxSize.min() if self.dk is None else self.dk
        kedges = numpy.arange(self.kmin, numpy.pi*pm.Nmesh/pm.BoxSize.min() + dk/2, dk)
    
        # project on to the desired basis
        muedges = numpy.linspace(0, 1, self.Nmu+1, endpoint=True)
        edges = [kedges, muedges]
        result, pole_result = measurestats.project_to_basis(pm.comm, x3d, y3d, edges, 
                                                            poles=self.poles, 
                                                            los=self.los, 
                                                            symmetric=True)
                                                            
        # compute the metadata to return
        Lx, Ly, Lz = pm.BoxSize
        meta = {'Lx':Lx, 'Ly':Ly, 'Lz':Lz, 'volume':Lx*Ly*Lz, 'N1':N1, 'N2':N2}                                                    
        
        # return all the necessary results
        return edges, result, pole_result, meta

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
        from nbodykit.extensionpoints import MeasurementStorage
        
        # only the master rank writes        
        if self.comm.rank == 0:
            
            edges, result, pole_result, meta = result
            if self.mode == "1d":
                cols = ['k', 'power', 'modes']
                result = [numpy.squeeze(result[i]) for i in [0, 2, 3]]
                edges_ = edges[0]
            else:
                edges_ = edges
                cols = ['k', 'mu', 'power', 'modes']
                
            # write binned statistic
            self.logger.info('measurement done; saving result to %s' %output)
            storage = MeasurementStorage.new(self.mode, output)
            storage.write(edges_, cols, result, **meta)
        
            # write multipoles
            if len(self.poles):
                filename, ext = os.path.splitext(output)
                pole_output = filename + '_poles' + ext
            
                # format is k pole_0, pole_1, ...., modes_1d
                self.logger.info('saving ell = %s multipoles to %s' %(",".join(map(str,self.poles)), pole_output))
                storage = MeasurementStorage.new('1d', pole_output)
            
                k, poles, N = pole_result
                cols = ['k'] + ['power_%d' %l for l in self.poles] + ['modes']
                pole_result = [k] + [pole for pole in poles] + [N]
                storage.write(edges[0], cols, pole_result, **meta)
                

class FFTCorrelationAlgorithm(Algorithm):
    """
    Algorithm to compute the 1d or 2d correlation function and multipoles
    in a periodic box. This FFTs the measured power spectrum to compute
    the correlation function
    """
    plugin_name = "FFTCorrelation"
    logger = logging.getLogger(plugin_name)

    @classmethod
    def register(kls):
        import copy
        
        # copy the FFTPower parser
        kls.parser = copy.copy(FFTPowerAlgorithm.parser)
        kls.parser.description = "correlation spectrum calculator via FFT in a periodic box"
        kls.parser.prog = 'FFTCorrelation'
            
    def run(self):
        """
        Run the algorithm, which computes and returns the correlation function
        """
        from nbodykit import measurestats
        from pmesh.particlemesh import ParticleMesh

        self.logger.setLevel(self.log_level)
        if self.comm.rank == 0: self.logger.info('importing done')

        # setup the particle mesh object, taking BoxSize from the painters
        pm = ParticleMesh(self.fields[0][0].BoxSize, self.Nmesh, dtype='f4', comm=self.comm)

        # only need one mu bin if 1d case is requested
        if self.mode == "1d": self.Nmu = 1

        # measure
        y3d, N1, N2 = measurestats.compute_3d_corr(self.fields, pm, comm=self.comm, log_level=self.log_level)
        x3d = pm.x

        # make the bin edges
        dr = pm.BoxSize[0] / pm.Nmesh
        redges = numpy.arange(0, pm.BoxSize[0] + dr * 0.5, dr)

        # project on to the desired basis
        muedges = numpy.linspace(0, 1, self.Nmu+1, endpoint=True)
        edges = [redges, muedges]
        result, pole_result = measurestats.project_to_basis(pm.comm, x3d, y3d, edges,
                                                            poles=self.poles,
                                                            los=self.los,
                                                            symmetric=False)

        # compute the metadata to return
        Lx, Ly, Lz = pm.BoxSize
        meta = {'Lx':Lx, 'Ly':Ly, 'Lz':Lz, 'volume':Lx*Ly*Lz, 'N1':N1, 'N2':N2}

        # return all the necessary results
        return edges, result, pole_result, meta

    def save(self, output, result):
        """
        Save the correlation function results to the specified output file

        Parameters
        ----------
        output : str
            the string specifying the file to save
        result : tuple
            the tuple returned by `run()` -- first argument specifies the bin
            edges and the second is a dictionary holding the data results
        """
        from nbodykit.extensionpoints import MeasurementStorage

        # only the master rank writes
        if self.comm.rank == 0:

            edges, result, pole_result, meta = result
            if self.mode == "1d":
                cols = ['r', 'corr', 'modes']
                result = [numpy.squeeze(result[i]) for i in [0, 2, 3]]
                edges_ = edges[0]
            else:
                edges_ = edges
                cols = ['r', 'mu', 'corr', 'modes']

            # write binned statistic
            self.logger.info('measurement done; saving result to %s' %output)
            storage = MeasurementStorage.new(self.mode, output)
            storage.write(edges_, cols, result, **meta)

            # write multipoles
            if len(self.poles):
                filename, ext = os.path.splitext(output)
                pole_output = filename + '_poles' + ext

                # format is k pole_0, pole_1, ...., modes_1d
                self.logger.info('saving ell = %s multipoles to %s' %(",".join(map(str,self.poles)), pole_output))
                storage = MeasurementStorage.new('1d', pole_output)

                k, poles, N = pole_result
                cols = ['k'] + ['power_%d' %l for l in self.poles] + ['modes']
                pole_result = [k] + [pole for pole in poles] + [N]
                storage.write(edges[0], cols, pole_result, **meta)
    


