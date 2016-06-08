from nbodykit.extensionpoints import Algorithm
from nbodykit.extensionpoints import DataSource, Transfer, Painter

import os
import numpy
import logging

def FieldType(ns):
    """
    Construct and return a `Field`:
    a tuple of (`DataSource`, `Painter`, `Transfer`)
    
    Notes
    -----
    *   the default Painter is set to `DefaultPainter`
    *   the default Transfer chain is set to 
        [`NormalizeDC`, `RemoveDC`, `AnisotropicCIC`]
    
    Parameters
    ----------
    fields_dict : OrderedDict
        an ordered dictionary where the keys are Plugin names
        and the values are instantiated Plugins
        
    Returns
    -------
    field : list
        list of (DataSource, Painter, Transfer)
    """         
    # define the default Painter and Transfer
    default_painter = Painter.create("DefaultPainter")
    default_transfer = [Transfer.create(x) for x in ['NormalizeDC', 'RemoveDC', 'AnisotropicCIC']]
    
    # start with a default option for (DataSource, Painter, Transfer)
    field = [None, default_painter, default_transfer]
    
    # set the DataSource
    if 'DataSource' not in ns:
        raise ValueError("exactly one `DataSource` per field must be specified")    
    field[0] = getattr(ns, 'DataSource')
    
    # set the Painter
    if 'Painter' in ns:
        field[1] = getattr(ns, 'Painter')
    
    # set the Transfer
    if 'Transfer' in ns:
        field[2] = getattr(ns, 'Transfer')
        
    return field
    
class FFTPowerAlgorithm(Algorithm):
    """
    Algorithm to compute the 1d or 2d power spectrum and multipoles
    in a periodic box, using an FFT
    """
    plugin_name = "FFTPower"
    logger = logging.getLogger(plugin_name)
    
    def __init__(self, mode, Nmesh, field, other=None, los='z', Nmu=5, 
                    dk=None, kmin=0., quiet=False, poles=[], paintbrush='cic'):
                      
        # combine the two fields
        self.fields = [self.field]
        if self.other is not None:
            self.fields.append(self.other)
        
    @classmethod
    def register(cls):
        
        s = cls.schema
        s.description = "periodic power spectrum calculator via FFT"

        # required
        s.add_argument("mode", type=str, choices=['1d', '2d'], 
            help='compute the power as a function of `k` or `k` and `mu`') 
        s.add_argument("Nmesh", type=int, 
            help='the number of cells in the gridded mesh')
        
        # the first field
        s.add_argument('field', type=FieldType,
            help="first data field; a tuple of (DataSource, Painter, Transfer)")
        s.add_argument('field.DataSource', type=DataSource.from_config, required=True,
            help='the 1st DataSource; run `nbkit.py --list-datasources` for all options')
        s.add_argument('field.Painter', type=Painter.from_config, required=False, 
            help='the 1st Painter; run `nbkit.py --list-painters` for all options')
        s.add_argument('field.Transfer', nargs='*', type=Transfer.from_config, required=False,
            help='the 1st Transfer chain; run `nbkit.py --list-transfers` for all options')
        
        # the other field
        s.add_argument('other', type=FieldType, required=False,
            help="the other data field; a tuple of (DataSource, Painter, Transfer)")
        s.add_argument('other.DataSource', type=DataSource.from_config, required=False,
            help='the 2nd DataSource; run `nbkit.py --list-datasources` for all options')
        s.add_argument('other.Painter',  type=Painter.from_config, required=False, 
            help='the 2nd Painter; run `nbkit.py --list-painters` for all options')
        s.add_argument('other.Transfer', nargs='*', type=Transfer.from_config, required=False,
            help='the 2nd Transfer chain; run `nbkit.py --list-transfers` for all options')
            
        # optional
        s.add_argument("los", type=str, choices="xyz",
            help="the line-of-sight direction -- the angle `mu` is defined with respect to")
        s.add_argument("Nmu", type=int,
            help='the number of mu bins to use from mu=[0,1]; if `mode = 1d`, then `Nmu` is set to 1' )
        s.add_argument("dk", type=float,
            help='the spacing of k bins to use; if not provided, the fundamental mode of the box is used')
        s.add_argument("kmin", type=float,
            help='the edge of the first `k` bin to use; default is 0')
        s.add_argument('quiet', type=bool,
            help="silence the logging output")
        s.add_argument('poles', type=int, nargs='*',
            help='if specified, also compute these multipoles from P(k,mu)')
        s.add_argument('paintbrush', type=lambda x: x.lower(), choices=['cic', 'tsc'],
            help='the density assignment kernel to use when painting; '
                 'CIC (2nd order) or TSC (3rd order)')
            
    def run(self):
        """
        Run the algorithm, which computes and returns the power spectrum
        """
        from nbodykit import measurestats
        from pmesh.particlemesh import ParticleMesh
        
        if self.comm.rank == 0: self.logger.info('importing done')
        
        # setup the particle mesh object, taking BoxSize from the painters
        pm = ParticleMesh(self.fields[0][0].BoxSize, self.Nmesh, 
                            paintbrush=self.paintbrush, dtype='f4', comm=self.comm)

        # only need one mu bin if 1d case is requested
        if self.mode == "1d": self.Nmu = 1
    
        # measure
        y3d, stats1, stats2 = measurestats.compute_3d_power(self.fields, pm, comm=self.comm)
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
        from nbodykit.storage import MeasurementStorage
        
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
            storage = MeasurementStorage.create(self.mode, output)
            storage.write(edges_, cols, result, **meta)
        
            # write multipoles
            if len(self.poles):
                filename, ext = os.path.splitext(output)
                pole_output = filename + '_poles' + ext
            
                # format is k pole_0, pole_1, ...., modes_1d
                self.logger.info('saving ell = %s multipoles to %s' %(",".join(map(str,self.poles)), pole_output))
                storage = MeasurementStorage.create('1d', pole_output)
            
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
    
    def __init__(self, mode, Nmesh, field, other=None, los='z', Nmu=5, 
                    dk=None, kmin=0., quiet=False, poles=[], paintbrush='cic'):
            
        # combine the two fields
        self.fields = [self.field]
        if self.other is not None:
            self.fields.append(self.other)

    @classmethod
    def register(cls):
        
        cls.schema.description = "correlation spectrum calculator via FFT in a periodic box"
        for name in FFTPowerAlgorithm.schema:
            cls.schema[name] = FFTPowerAlgorithm.schema[name]
            
    def run(self):
        """
        Run the algorithm, which computes and returns the correlation function
        """
        from nbodykit import measurestats
        from pmesh.particlemesh import ParticleMesh

        if self.comm.rank == 0: self.logger.info('importing done')

        # setup the particle mesh object, taking BoxSize from the painters
        pm = ParticleMesh(self.fields[0][0].BoxSize, self.Nmesh, 
                            paintbrush=self.paintbrush, dtype='f4', comm=self.comm)

        # only need one mu bin if 1d case is requested
        if self.mode == "1d": self.Nmu = 1

        # measure
        y3d, stats1, stats2 = measurestats.compute_3d_corr(self.fields, pm, comm=self.comm)
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
        from nbodykit.storage import MeasurementStorage

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
            storage = MeasurementStorage.create(self.mode, output)
            storage.write(edges_, cols, result, **meta)

            # write multipoles
            if len(self.poles):
                filename, ext = os.path.splitext(output)
                pole_output = filename + '_poles' + ext

                # format is k pole_0, pole_1, ...., modes_1d
                self.logger.info('saving ell = %s multipoles to %s' %(",".join(map(str,self.poles)), pole_output))
                storage = MeasurementStorage.create('1d', pole_output)

                k, poles, N = pole_result
                cols = ['k'] + ['power_%d' %l for l in self.poles] + ['modes']
                pole_result = [k] + [pole for pole in poles] + [N]
                storage.write(edges[0], cols, pole_result, **meta)
    


