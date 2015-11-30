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
              
from nbodykit import plugins, measurestats
from nbodykit.utils.pluginargparse import PluginArgumentParser
from pmesh.particlemesh import ParticleMesh
from pmesh.transfer import TransferFunction

#--------------------------------------------------
# setup the parser
#--------------------------------------------------
def initialize_parser(**kwargs):
    """
    Initialize the command-line parser for ``power.py``, 
    optionally providing``args`` to be passed to the
    initializing (for, i.e., the case when this is called
    not from the command line)
    
    Parameters
    ----------
    kwargs : 
        keyword arguments to pass to the `PluginArgumentParser` class
    """
    parser = PluginArgumentParser("Parallel Power Spectrum Calculator",
            loader=plugins.load,
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

    # add the positional arguments
    parser.add_argument("mode", choices=["2d", "1d"]) 
    parser.add_argument("Nmesh", type=int, help='size of calculation mesh, recommend 2 * Ngrid')
    parser.add_argument("output", help='write power to this file. set as `-` for stdout') 

    # add the input field types
    h = "one or two input fields, specified as:\n\n"
    parser.add_argument("inputs", nargs="+", type=plugins.DataSource.open, 
                        help=h+plugins.DataSource.format_help())

    # add the optional arguments
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
    
    return parser

#--------------------------------------------------
# computation tools
#--------------------------------------------------
def AnisotropicCIC(comm, complex, w):
    for wi in w:
        tmp = (1 - 2. / 3 * numpy.sin(0.5 * wi) ** 2) ** 0.5
        complex[:] /= tmp

def compute_power(ns, comm=None, transfer=None, painter=None):
    """
    Compute the power spectrum. Given a `Namespace`, this is the function,
    that computes and saves the power spectrum. It does all the work.
    
    Parameters
    ----------
    ns : argparse.Namespace
        the parser namespace corresponding to the ``initialize_parser``
        functions
    comm : MPI.Communicator
        the communicator to pass to the ``ParticleMesh`` object
    transfer : list, optional
        list of transfer functions to apply that will be
        passed to ``compute_3d_power``. If `None`, then
        the default chain ``TransferFunction.NormalizeDC``, 
        ``TransferFunction.RemoveDC``, and ``AnisotropicCIC``
        will be applied
    painter : callable, optional
        the painter function(s) to pass to ``compute_3d_power``. 
        Only passed if not `None`
    """    
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    
    # handle default measurement keywords
    measure_kw = {'comm':comm, 'log_level':ns.log_level}
    
    # transfer chain
    default_chain = [TransferFunction.NormalizeDC, TransferFunction.RemoveDC, AnisotropicCIC]
    measure_kw.setdefault('transfer', default_chain)
    if transfer is not None:
        measure_kw['transfer'] = transfer
    
    # painter
    if painter is not None:
        measure_kw['painter'] = painter
    
    # set logging level
    logger.setLevel(ns.log_level)
    
    if rank == 0: logger.info('importing done')

    # setup the particle mesh object, taking BoxSize from the painters
    pm = ParticleMesh(ns.inputs[0].BoxSize, ns.Nmesh, dtype='f4', comm=comm)

    # only need one mu bin if 1d case is requested
    if ns.mode == "1d": ns.Nmu = 1

    # binning keywords
    binning_kw = {'poles':ns.poles, 'los':ns.los}
    
    # correlation function
    if ns.correlation:

        # measure
        y3d, N1, N2 = measurestats.compute_3d_corr(ns.inputs, pm, **measure_kw)
        x3d = pm.x
                
        # make the bin edges
        dx = pm.BoxSize[0] / pm.Nmesh
        xedges = numpy.arange(0, pm.BoxSize[0] + dx * 0.5, dx)
        
        # correlation needs all modes
        binning_kw['symmetric'] = False
        
        # col names
        x_str, y_str = 'r', 'corr'
                
    # power spectrum
    else:
        
        # measure
        y3d, N1, N2 = measurestats.compute_3d_power(ns.inputs, pm, **measure_kw)
        x3d = pm.k
        
        # binning in k out to the minimum nyquist frequency 
        # (accounting for possibly anisotropic box)
        dx = 2*numpy.pi/pm.BoxSize.min() if ns.dk is None else ns.dk
        xedges = numpy.arange(ns.kmin, numpy.pi*pm.Nmesh/pm.BoxSize.min() + dx/2, dx)
        
        # power spectrum doesnt need negative z wavenumbers
        binning_kw['symmetric'] = True
        
        # col names
        x_str, y_str = 'k', 'power'
    
    # project on to the desired basis
    muedges = numpy.linspace(0, 1, ns.Nmu+1, endpoint=True)
    edges = [xedges, muedges]
    result, pole_result = measurestats.project_to_basis(pm.comm, x3d, y3d, edges, **binning_kw)

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
        storage = plugins.MeasurementStorage.new(ns.mode, ns.output)
        storage.write(edges, cols, result, **meta)
        
        # write multipoles
        if pole_result is not None:
            if ns.pole_output is None:
                raise RuntimeError("you specified multipoles to compute, but did not provide an output file name")
            
            # format is k pole_0, pole_1, ...., modes_1d
            logger.info('saving ell = %s multipoles to %s' %(",".join(map(str,ns.poles)), ns.pole_output))
            storage = plugins.MeasurementStorage.new('1d', ns.pole_output)
            
            cols = [x_str] + [y_str+'_%d' %l for l in ns.poles] + ['modes']
            storage.write(xedges, cols, pole_result, **meta)
            
            
def main():
    """
    The main function to initialize the parser and do the work
    """
    # parse
    ns = initialize_parser().parse_args()
        
    # do the work
    compute_power(ns)

if __name__ == '__main__':
    main()
