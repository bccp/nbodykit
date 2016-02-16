from nbodykit.extensionpoints import Algorithm, plugin_isinstance
from nbodykit.extensionpoints import DataSource, Transfer, Painter
from nbodykit.plugins import add_plugin_list_argument

import numpy
import logging

def FieldsType(input_fields):
    """
    Construct a list of `Fields`, which are tuples of 
    (`DataSource`, `Painter`, `Transfer`)
    
    Parameters
    ----------
    input_fields : list
        a list of string representations of 
        (DataSource, Painter, Transfer) plugins 
        
    Returns
    -------
    fields : list
        list of instantiated (DataSource, Painter, Transfer) tuples
    """     
    fields = []
    i = 0
    N = len(input_fields)
    
    default_painter = Painter.fromstring("DefaultPainter")
    default_transfer = [Transfer.fromstring(x) for x in ['NormalizeDC', 'RemoveDC', 'AnisotropicCIC']]
    
    while i < N:
        
        # start with a default option for (DataSource, Painter, Transfer)
        field = [None, default_painter, []]
        
        # should be a DataSource here, or break
        if plugin_isinstance(input_fields[i], DataSource):
            
            # set data source
            field[0] = DataSource.fromstring(input_fields[i])
            
            # loop until out of values or another DataSource found
            i += 1
            while i < N and not plugin_isinstance(input_fields[i], DataSource):
                s = input_fields[i]
                
                # set Painter
                if plugin_isinstance(s, Painter):
                    field[1] = Painter.fromstring(s)
                # add one Transfer
                elif plugin_isinstance(s, Transfer):
                    field[2].append(Transfer.fromstring(s))
                # add list of Transfers
                elif isinstance(s, list):
                    field[2] += [Transfer.fromstring(x) for x in s]
                else:
                    raise ValueError("failure to parse line `%s` for `fields` key" %str(s))                    
                i += 1
            if not len(field[2]): field[2] = default_transfer
            fields.append(tuple(field))
        else: # failure
            raise ValueError("parsing error constructing input `fields` -- see documentation for proper syntax")

    return fields
    
class FFTPowerAlgorithm(Algorithm):
    """
    Algorithm to compute the 1d or 2d power spectrum and multipoles
    in a periodic box, using an FFT
    """
    plugin_name = "FFTPower"
    logger = logging.getLogger(plugin_name)
    
    @classmethod
    def register(kls):
        p = kls.parser
        p.description = "periodic power spectrum calculator via FFT"

        # the required arguments
        p.add_argument("mode", choices=["2d", "1d"], 
            help='compute the power as a function of `k` or `k` and `mu`') 
        p.add_argument("Nmesh", type=int, 
            help='the number of cells in the gridded mesh')
        add_plugin_list_argument(p, "fields", type=FieldsType,
            help="strings specifying the input data sources, painters, and transfers, in that order, respectively -- "+
                "use --list-datasource and --list-painters for further documentation",
            metavar="DataSource [Painter] [Transfer] [DataSource [Painter] [Transfer]]")
        
        # the optional arguments
        p.add_argument("--los", choices="xyz", default='z',
            help="the line-of-sight direction -- the angle `mu` is defined with respect to")
        p.add_argument("--Nmu", type=int, default=5, 
            help='the number of mu bins to use from mu=[0,1]; if `mode = 1d`, then `Nmu` is set to 1' )
        p.add_argument("--dk", type=float,
            help='the spacing of k bins to use; if not provided, the fundamental mode of the box is used')
        p.add_argument("--kmin", type=float, default=0.,
            help='the edge of the first `k` bin to use; default is 0')
        p.add_argument('-q', '--quiet', action="store_const", dest="log_level", default=logging.DEBUG,
            help="silence the logging output", const=logging.ERROR)
        p.add_argument('--poles', type=lambda s: [int(i) for i in s.split()], default=[],
            help='if specified, also compute these multipoles from P(k,mu)')
            
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
        y3d, stats1, stats2 = measurestats.compute_3d_power(self.fields, pm, comm=self.comm, log_level=self.log_level)
        x3d = pm.k
    
        # get the number of objects (in a safe manner)
        N1 = stats1.get('Ntot', -1)
        N2 = stats2.get('Ntot', -1)
        
        # binning in k out to the minimum nyquist frequency 
        # (accounting for possibly anisotropic box)
        dk = 2*numpy.pi/pm.BoxSize.min() if self.dk is None else self.dk
        kedges = numpy.arange(self.kmin, numpy.pi*pm.Nmesh/pm.BoxSize.max() + dk/2, dk)
    
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
        y3d, stats1, stats2 = measurestats.compute_3d_corr(self.fields, pm, comm=self.comm, log_level=self.log_level)
        x3d = pm.x
        
        # get the number of objects (in a safe manner)
        N1 = stats1.get('Ntot', -1)
        N2 = stats2.get('Ntot', -1)

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
    


