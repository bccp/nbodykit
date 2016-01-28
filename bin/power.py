import logging
from mpi4py import MPI
import numpy

rank = MPI.COMM_WORLD.rank
name = MPI.Get_processor_name()
logging.basicConfig(level=logging.DEBUG,
                    format='rank %d on %s: '%(rank,name) + \
                            '%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('power.py')
              
from nbodykit import measurestats
from nbodykit.extensionpoints import DataSource, Painter, Transfer
from nbodykit.extensionpoints import MeasurementStorage
from nbodykit.extensionpoints import plugin_isinstance

from nbodykit.plugins import ArgumentParser, ListPluginsAction, load
from argparse import Action, SUPPRESS
from pmesh.particlemesh import ParticleMesh

def construct_fields(input_fields, X=None):
    """
    Construct a list holding 1 or 2 tuples of (DataSource, Painter, Transfer).
    """
    # load plugin paths
    if X is not None:
        if isinstance(X, str):
            load(X)
        elif isinstance(X, list):
            for path in X: load(path)
    
    fields = []
    i = 0
    N = len(input_fields)
    
    while i < N:
        
        # start with a default option for (DataSource, Painter, Transfer)
        field = [None, Painter.create("DefaultPainter"), []]
        
        # should be a DataSource here, or break
        if plugin_isinstance(input_fields[i], DataSource):
            
            # set data source
            field[0] = DataSource.create(input_fields[i])
            
            # loop until out of values or another DataSource found
            i += 1
            while i < N and not plugin_isinstance(input_fields[i], DataSource):
                s = input_fields[i]
                
                # set Painter
                if plugin_isinstance(s, Painter):
                    field[1] = Painter.create(s)
                # add one Transfer
                elif plugin_isinstance(s, Transfer):
                    field[2].append(Transfer.create(s))
                # add list of Transfers
                elif isinstance(s, list):
                    field[2] += [Transfer.create(x) for x in s]
                else:
                    raise ValueError("failure to parse line `%s` for `fields` key" %str(s))                    
                i += 1
            fields.append(tuple(field))
        else: # failure
            raise ValueError("failure to parse `fields`")

    return fields
    
h = """
    mode: 
        help: 1d or 2d
        choices: [1d, 2d]
    Nmesh:
        help: size of calculation mesh, recommend 2 * Ngrid
    output:
        help: write power to this file. set as `-` for stdout'
    fields:
        help: Input data sources and painters. 
               Use --list-painter and --list-datasource to see a list of painters and data sources.
    los: 
        help: the line-of-sight direction, which the angle `mu` is defined with respect to
        choices: [x, y, z]
        default: z
        required: False
    Nmu: 
        help: the number of mu bins to use; if `mode = 1d`, then `Nmu` is set to 1
        default: 5
        required: False
    dk: 
        help: the spacing of k bins to use; if not provided, the fundamental mode of the box is used
    kmin:
        help: the edge of the first bin to use; default is 0
        default: 0.
        required: False
    poles:
        help: if specified, compute these multipoles from P(k,mu), saving to `pole_output`
        default: []
        required: False
    pole_output: 
        help: the name of the output file for multipoles
        required: False
    correlation:
        help: compute the correlation function instead of power spectrum
        default : False
        required: False
    """

def ReadConfigFile(parser_string):
    import yaml
    
    class ConfigFileAction(Action):
        info = yaml.load(parser_string)
            
        def __call__(self, parser, namespace, values, option_string=None):
            
            # set defaults
            for k in self.info:
                setattr(namespace, k, self.info[k].get('default', None))
        
            # read the yaml config file
            config = yaml.load(open(values, 'r'))
            
            # set the fields attribute
            namespace.fields = construct_fields(config.pop('fields'), X=config.pop('X', None))
            
            for k in config:
                v = config[k]
                if 'choices' in self.info[k]:
                    if config[k] not in self.info[k]['choices']:
                        args = (k, config[k], ", ".join(["'%s'" %a for a in self.info[k]['choices']]))
                        raise ValueError("argument %s: invalid choice '%s' (choose from %s)" %args)
                setattr(namespace, k, config[k])
                
        @classmethod
        def format_help(cls):
            positional = []
            optional = []
            for k in cls.info:
                s =  "  %-20s %s" %(k, cls.info[k]['help'])
                if cls.info[k].get('required', True):
                    positional.append(s)
                else:
                    optional.append(s)
            
            positional = "\n".join(positional)
            optional = "\n".join(optional)
            help_str = "positional arguments:\n%s\n\noptional arguments:\n%s\n" %(positional, optional)

            class ConfigFileHelp(Action):
                def __init__(self,
                             option_strings,
                             dest=SUPPRESS,
                             default=SUPPRESS,
                             help=None):
                    Action.__init__(self, 
                        option_strings=option_strings,
                        dest=dest,
                        default=default,
                        nargs=0,
                        help=help)
                           
                def __call__(self, parser, namespace, values, option_string=None):
                    parser.exit(0, help_str)
            return ConfigFileHelp
            
    return ConfigFileAction


        
        
def initialize_parser(**kwargs):
    """
    Initialize the command-line parser for ``power.py``, 
    optionally providing``args`` to be passed to the
    initializing (for, i.e., the case when this is called
    not from the command line)
    
    Parameters
    ----------
    kwargs : 
        keyword arguments to pass to the `ArgumentParser` class
    """
    parser = ArgumentParser("Parallel Power Spectrum Calculator",
            description=
         """Calculating matter power spectrum from RunPB input files. 
            Output is written to stdout, in Mpc/h units. 
            PowerSpectrum is the true one, without (2 pi) ** 3 factor. (differ from Gadget/NGenIC internal)
            This script moves all particles to the halo center.
         """,
            epilog=
         """
            This script is written by Yu Feng, as part of `nbodykit'. 
            Other contributors are: Nick Hand, Man-yat Chu
            The author would like thank Marcel Schmittfull for the explanation on cic, shotnoise, and k==0 plane errors.
         """,
            **kwargs
         )
    
    # add the input field types
    read_config = ReadConfigFile(h)
    parser.add_argument("config", type=str, action=read_config, help="the configuration file; use --config-help for format")
    parser.add_argument("--config-help", action=read_config.format_help())

    parser.add_argument('-q', '--quiet', help="silence the logging output",  
            action="store_const", dest="log_level", const=logging.ERROR, default=logging.DEBUG)
    

    parser.add_argument("--list-datasource", action=ListPluginsAction(DataSource))
    parser.add_argument("--list-painter", action=ListPluginsAction(Painter))
    parser.add_argument("--list-transfer", action=ListPluginsAction(Transfer))

    return parser

def initialize_compatible_parser():
    # add the arguments, compatible for old power.py syntax
    class InputAction(Action):
        def __init__(self, option_strings, dest, 
            nargs=None, const=None, default=None, type=None, 
            choices=None, required=False, help=None, metavar=None):
            Action.__init__(self, option_strings, dest,
                    nargs, const, default, type, choices, required, help, metavar)

        def __call__(self, parser, namespace, values, option_string=None):
            fields = []
            default_painter = Painter.create("DefaultPainter")
            transfer = [Transfer.create(x) for x in ['NormalizeDC', 'RemoveDC', 'AnisotropicCIC']]
            for string in values:
                try:
                    datasource = DataSource.create(string)
                    fields.append((datasource, default_painter, transfer))
                except KeyError: # FIXME: define a proper way to test if the string
                                 # is a datasource or redo this entire mechanism
                    datasource, painter = fields.pop()
                    painter = Painter.create(string)
                    fields.append((datasource, painter, transfer
                        ))
            namespace.fields = fields
    parser = ArgumentParser("Parallel Power Spectrum Calculator")

    parser.add_argument("mode", choices=["2d", "1d"]) 
    parser.add_argument("Nmesh", type=int, help='size of calculation mesh, recommend 2 * Ngrid')
    parser.add_argument("output", help='write power to this file. set as `-` for stdout') 

    # add the input field types
    h = "one or two input fields, specified as:\n\n"
    parser.add_argument("fields", nargs="+", 
            action=InputAction,
            help="Input data sources and painters. Use --list-painter and --list-datasource to see a list of painters and data sources.", 
            metavar="DataSource [Painter] [DataSource [Painter]]")
    parser.add_argument("--los", choices="xyz", default='z',
            help="the line-of-sight direction, which the angle `mu` is defined with respect to")
    parser.add_argument("--Nmu", type=int, default=5,
            help='the number of mu bins to use; if `mode = 1d`, then `Nmu` is set to 1' )
    parser.add_argument("--dk", type=float,
            help='the spacing of k bins to use; if not provided, the fundamental mode of the box is used')
    parser.add_argument("--kmin", type=float, default=0,
            help='the edge of the first bin to use; default is 0')
    parser.add_argument('-q', '--quiet', help="silence the logging output",
            action="store_const", dest="log_level", const=logging.ERROR, default=logging.DEBUG)
    parser.add_argument('--poles', type=lambda s: [int(i) for i in s.split()], default=[],
            help='if specified, compute these multipoles from P(k,mu), saving to `pole_output`')
    parser.add_argument('--pole_output', type=str, help='the name of the output file for multipoles')

    parser.add_argument("--correlation", action='store_true', default=False,
        help='Calculate correlation function instead of power spectrum.')

    def error(message): 
        raise ValueError(x)
    parser.error = error
    return parser

def compute_power(ns, comm=None):
    """
    Compute the power spectrum. Given a `Namespace`, this is the function,
    that computes and saves the power spectrum. It does all the work.
    
    Parameters
    ----------
    ns : argparse.Namespace
        the parser namespace corresponding to the ``initialize_parser``
        function
    comm : MPI.Communicator
        the communicator to pass to the ``ParticleMesh`` object
    """
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    logger.setLevel(ns.log_level)
    if rank == 0: logger.info('importing done')

    # setup the particle mesh object, taking BoxSize from the painters
    pm = ParticleMesh(ns.fields[0][0].BoxSize, ns.Nmesh, dtype='f4', comm=comm)

    # only need one mu bin if 1d case is requested
    if ns.mode == "1d": ns.Nmu = 1
    
    # correlation function
    if ns.correlation:

        # measure
        y3d, N1, N2 = measurestats.compute_3d_corr(ns.fields, pm, comm=comm, log_level=ns.log_level)
        x3d = pm.x
                
        # make the bin edges
        dx = pm.BoxSize[0] / pm.Nmesh
        xedges = numpy.arange(0, pm.BoxSize[0] + dx * 0.5, dx)
        
        # correlation needs all modes
        symmetric = False
        
        # col names
        x_str, y_str = 'r', 'corr'
                
    # power spectrum
    else:
        
        # measure
        y3d, N1, N2 = measurestats.compute_3d_power(ns.fields, pm, comm=comm, log_level=ns.log_level)
        x3d = pm.k
        
        # binning in k out to the minimum nyquist frequency 
        # (accounting for possibly anisotropic box)
        dx = 2*numpy.pi/pm.BoxSize.min() if ns.dk is None else ns.dk
        xedges = numpy.arange(ns.kmin, numpy.pi*pm.Nmesh/pm.BoxSize.min() + dx/2, dx)
        
        # power spectrum doesnt need negative z wavenumbers
        symmetric = True
        
        # col names
        x_str, y_str = 'k', 'power'
    
    # project on to the desired basis
    muedges = numpy.linspace(0, 1, ns.Nmu+1, endpoint=True)
    edges = [xedges, muedges]
    result, pole_result = measurestats.project_to_basis(pm.comm, x3d, y3d, edges, 
                                                        poles=ns.poles, los=ns.los, 
                                                        symmetric=symmetric)

    # now output    
    if ns.mode == "1d":
        cols = [x_str, y_str, 'modes']
        result = [numpy.squeeze(result[i]) for i in [0, 2, 3]]
        edges = edges[0]
    else:
        cols = [x_str, 'mu', y_str, 'modes']
        
    if rank == 0:
        
        # metadata
        Lx, Ly, Lz = pm.BoxSize
        meta = {'Lx':Lx, 'Ly':Ly, 'Lz':Lz, 'volume':Lx*Ly*Lz, 'N1':N1, 'N2':N2}
        
        # write binned statistic
        logger.info('measurement done; saving result to %s' %ns.output)
        storage = MeasurementStorage.new(ns.mode, ns.output)
        storage.write(edges, cols, result, **meta)
        
        # write multipoles
        if pole_result is not None:
            if ns.pole_output is None:
                raise RuntimeError("you specified multipoles to compute, but did not provide an output file name")
            
            # format is k pole_0, pole_1, ...., modes_1d
            logger.info('saving ell = %s multipoles to %s' %(",".join(map(str,ns.poles)), ns.pole_output))
            storage = MeasurementStorage.new('1d', ns.pole_output)
            
            x, poles, N = pole_result
            cols = [x_str] + [y_str+'_%d' %l for l in ns.poles] + ['modes']
            pole_result = [x] + [pole for pole in poles] + [N]
            storage.write(xedges, cols, pole_result, **meta)
            
            
def main():
    # parse
    parser = initialize_parser()
    oldparser = initialize_compatible_parser()

    try:
        ns = oldparser.parse_args()
    except:
        ns = parser.parse_args()
        
    # do the work
    compute_power(ns)

if __name__ == '__main__':
    main()
