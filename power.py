import logging
import warnings
from mpi4py import MPI
import numpy

rank = MPI.COMM_WORLD.rank
name = MPI.Get_processor_name()
logging.basicConfig(level=logging.DEBUG,
                    format='rank %d on %s: '%(rank,name) + \
                            '%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger('power.py')
              
import nbodykit
from nbodykit import plugins
from nbodykit.utils.pluginargparse import PluginArgumentParser
from nbodykit.measurepower import measurepower
from pypm.particlemesh import ParticleMesh
from pypm.transfer import TransferFunction

#--------------------------------------------------
# setup the parser
#--------------------------------------------------
def initialize_power_parser(**kwargs):
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
    parser.add_argument("--bunchsize", type=int, default=1024*1024*4,
        help='Number of particles to read per rank. A larger number usually means faster IO, but less memory for the FFT mesh. This is not respected by some data sources.')
    parser.add_argument("--binshift", type=float, default=0.0,
            help='Shift the bin center by this fraction of the bin width. Default is 0.0. Marcel uses 0.5. this shall rarely be changed.' )
    parser.add_argument("--remove-cic", default='anisotropic', choices=["anisotropic","isotropic", "none"],
            help='deconvolve cic, anisotropic is the proper way, see http://www.personal.psu.edu/duj13/dissertation/djeong_diss.pdf')
    parser.add_argument("--remove-shotnoise", action='store_true', default=False,
            help='Remove shotnoise')
    parser.add_argument("--Nmu", type=int, default=5,
            help='the number of mu bins to use; if `mode = 1d`, then `Nmu` is set to 1' )
    parser.add_argument("--los", choices="xyz", default='z',
            help="the line-of-sight direction, which the angle `mu` is defined with respect to")
    parser.add_argument("--dk", type=float,
            help='the spacing of k bins to use; if not provided, the fundamental mode of the box is used')
    parser.add_argument("--kmin", type=float, default=0,
            help='the edge of the first bin to use; default is 0')
    parser.add_argument('-q', '--quiet', help="silence the logging output",
            action="store_const", dest="log_level", const=logging.ERROR, default=logging.DEBUG)
    parser.add_argument('--poles', type=lambda s: map(int, s.split()), default=[],
            help='if specified, compute these multipoles from P(k,mu), saving to `pole_output`')
    parser.add_argument('--pole_output', type=str, help='the name of the output file for multipoles')

    parser.add_argument("--correlation", action='store_true', default=False,
        help='Calculate correaltion function instead of power spectrum.')
    
    return parser

#--------------------------------------------------
# computation tools
#--------------------------------------------------
def AnisotropicCIC(comm, complex, w):
    for wi in w:
        tmp = (1 - 2. / 3 * numpy.sin(0.5 * wi) ** 2) ** 0.5
        complex[:] /= tmp

def IsotropicCIC(comm, complex, w):
    for row in range(complex.shape[0]):
        scratch = numpy.float64(w[0][row] ** 2)
        for wi in w[1:]:
            scratch = scratch + wi[0] ** 2

        tmp = (1.0 - 0.666666667 * numpy.sin(scratch * 0.5) ** 2) ** 0.5
        complex[row] *= tmp

def compute_power(ns, comm=None):
    """
    Compute the power spectrum. Given a `Namespace`, this is the function,
    that computes and saves the power spectrum. It does all the work.
    
    Parameters
    ----------
    ns : argparse.Namespace
        the parser namespace corresponding to the ``initialize_power_parser``
        functions
    comm : MPI.Communicator
        the communicator to pass to the ``ParticleMesh`` object
    """
    rank = comm.rank if comm is not None else MPI.COMM_WORLD.rank
    
    # set logging level
    logger.setLevel(ns.log_level)
    
    if rank == 0:
        logger.info('importing done')

    chain = [TransferFunction.NormalizeDC, TransferFunction.RemoveDC]
    if ns.remove_cic == 'anisotropic':
        chain.append(AnisotropicCIC)
    if ns.remove_cic == 'isotropic':
        chain.append(IsotropicCIC)
        
    # setup the particle mesh object, taking BoxSize from the painters
    pm = ParticleMesh(ns.inputs[0].BoxSize, ns.Nmesh, dtype='f4', comm=comm)

    # paint first input
    Ntot1 = paint(ns.inputs[0], pm, ns)

    # painting
    if rank == 0:
        logger.info('painting done')
    pm.r2c()
    if rank == 0:
        logger.info('r2c done')

    # filter the field 
    pm.transfer(chain)

    # do the cross power
    do_cross = len(ns.inputs) > 1 and ns.inputs[0] != ns.inputs[1]

    if do_cross:
        
        # crash if box size isn't the same
        if not numpy.all(ns.inputs[0].BoxSize == ns.inputs[1].BoxSize):
            raise ValueError("mismatch in box sizes for cross power measurement")
        
        c1 = pm.complex.copy()
        Ntot2 = paint(ns.inputs[1], pm, ns)

        if rank == 0:
            logger.info('painting 2 done')

        pm.r2c()
        if rank == 0:
            logger.info('r2c 2 done')

        # filter the field 
        pm.transfer(chain)
        c2 = pm.complex
  
    # do the auto power
    else:
        c1 = pm.complex
        c2 = pm.complex
        Ntot2 = Ntot1 

    shotnoise =  pm.BoxSize.prod() / (1.0*Ntot1)

    # reuse the memory in c1.real for the 3d power spectrum
    p3d = c1.real

    # calculate the 3d power spectrum, row by row to save memory
    for row in range(len(c1)):
        # the power P(k,mu)
        p3d[row, ...] = c1[row].real * c2[row].real + c1[row].imag * c2[row].imag

    if ns.remove_shotnoise and not do_cross:
        p3d[...] -= shotnoise / pm.BoxSize.prod()

    if ns.correlation:
        pm.complex[:] = p3d.copy()
        # direct transform dimensionless p3d
        pm.c2r()
        p3d = pm.real
        k = pm.r
        dk = pm.BoxSize[0] / pm.Nmesh
        kedges = numpy.arange(0, pm.BoxSize[0] + dk * 0.5, dk)
        print pm.r[0].shape, pm.r[1].shape, pm.r[1].shape
    else: 
        # each complex field has units of L^3, so power is L^6
        # YF: FIXME: what does this comment mean? 
        # ref to http://icc.dur.ac.uk/~tt/Lectures/UA/L4/cosmology.pdf
        p3d[...] *= pm.BoxSize.prod() 
        k = pm.k
        # kedges out to the minimum nyquist frequency (accounting for possibly anisotropic box)
        BoxSize_min = numpy.amin(pm.BoxSize)
        w_to_k = pm.Nmesh / BoxSize_min
        if ns.dk is None: 
            dk = 2*numpy.pi/BoxSize_min
        else:
            dk = ns.dk
        kedges = numpy.arange(ns.kmin, numpy.pi*w_to_k + dk/2, dk)
        kedges += ns.binshift * dk

    # only need one mu bin if 1d case is requested
    if ns.mode == "1d": ns.Nmu = 1 

    # do the calculation
    Lx, Ly, Lz = pm.BoxSize
    meta = {'Lx':Lx, 'Ly':Ly, 'Lz':Lz, 'volume':Lx*Ly*Lz, 
            'N1':Ntot1, 'N2':Ntot2, 'shot_noise': shotnoise}
    

    # now project the 3d power spectrum to a desired basis

    result = measurepower(pm.comm, k, p3d, kedges, ns.Nmu, 
                            los=ns.los, poles=ns.poles)
    
    # format the output appropriately
    if len(ns.poles):
        pole_result, pkmu_result, edges = result
        result = dict(zip(['k','mu','power','modes','edges'], pkmu_result+(edges,)))
    elif ns.mode == "1d":
        # this writes out 0 -> mean k, 2 -> mean power, 3 -> number of modes
        meta['edges'] = result[-1][0] # write out kedges as metadata
        result = map(numpy.ravel, (result[i] for i in [0, 2, 3]))
    elif ns.mode == "2d":
        result = dict(zip(['k','mu','power','modes','edges'], result))
        
    if rank == 0:
        # save the power
        logger.info('measurement done; saving power to %s' %ns.output)
        storage = plugins.PowerSpectrumStorage.new(ns.mode, ns.output)
        storage.write(result, **meta)
        
        # save the multipoles
        if len(ns.poles):
            if ns.pole_output is None:
                raise RuntimeError("you specified multipoles to compute, but did not provide an output file name")
            meta['edges'] = edges[0]
            
            # format is k pole_0, pole_1, ...., modes_1d
            logger.info('saving ell = %s multipoles to %s' %(",".join(map(str,ns.poles)), ns.pole_output))
            result = [x for x in numpy.vstack(pole_result)]
            storage = plugins.PowerSpectrumStorage.new('1d', ns.pole_output)
            storage.write(result, **meta)
            
            
 
def paint(input, pm, ns):
    """
    Paint the ``DataSource`` specified by ``input`` onto the 
    ``ParticleMesh`` specified by ``pm``
    
    Parameters
    ----------
    input : ``DataSource``
        the data source object that handles reading of fields
    pm : ``ParticleMesh``
        particle mesh object that does the painting
    ns : argparse.Namespace
        the namespace holding the command-line options
        
    Returns
    -------
    Ntot : int
        the total number of objects, as determined from painting
    """
    # compatibility with the older painters. 
    # We need to get rid of them.
    if hasattr(input, 'paint'):
        if pm.comm.rank == 0:
            warnings.warn('paint method of type %s shall be replaced with a read method'
                % type(input), DeprecationWarning)
        return input.paint(pm)

    pm.real[:] = 0
    Ntot = 0

    if pm.comm.rank == 0: 
        logger.info("BoxSize = %s", str(input.BoxSize))
    for position, weight in input.read(['Position', 'Weight'], pm.comm, ns.bunchsize):
        min = numpy.min(
            pm.comm.allgather(
                    [numpy.inf, numpy.inf, numpy.inf] 
                    if len(position) == 0 else 
                    position.min(axis=0)),
            axis=0)
        max = numpy.max(
            pm.comm.allgather(
                    [-numpy.inf, -numpy.inf, -numpy.inf] 
                    if len(position) == 0 else 
                    position.max(axis=0)),
            axis=0)
        if pm.comm.rank == 0:
            logger.info("Range of position %s:%s" % (str(min), str(max)))

        layout = pm.decompose(position)
        # Ntot shall be calculated before exchange. Issue #55.
        if weight is None:
            Ntot += len(position)
            weight = 1
        else:
            Ntot += weight.sum()
            weight = layout.exchange(weight)
           
        position = layout.exchange(position)

        pm.paint(position, weight)
    return pm.comm.allreduce(Ntot)

def main():
    """
    The main function to initialize the parser and do the work
    """
    # parse
    ns = initialize_power_parser().parse_args()
        
    # do the work
    compute_power(ns)

if __name__ == '__main__':
    main()
