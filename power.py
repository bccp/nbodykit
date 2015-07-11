from sys import argv
from sys import stdout
from sys import stderr
import logging

logging.basicConfig(level=logging.DEBUG)

import numpy
import nbodykit
from nbodykit import plugins
from nbodykit.utils.argumentparser import ArgumentParser
#--------------------------------------------------
# setup the parser
#--------------------------------------------------

# initialize the parser
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
     """
     )

# add the positional arguments
parser.add_argument("mode", choices=["2d", "1d"]) 
parser.add_argument("BoxSize", type=float, help='BoxSize in Mpc/h')
parser.add_argument("Nmesh", type=int, help='size of calculation mesh, recommend 2 * Ngrid')
parser.add_argument("output", help='write power to this file. set as `-` for stdout') 

# add the input field types
h = "one or two input fields, specified as:\n\n"
parser.add_argument("inputs", nargs="+", type=plugins.InputPainter.parse, 
                    help=h+plugins.InputPainter.format_help())

# add the optional arguments
parser.add_argument("--binshift", type=float, default=0.0,
        help='Shift the bin center by this fraction of the bin width. Default is 0.0. Marcel uses 0.5. this shall rarely be changed.' )
parser.add_argument("--bunchsize", type=int, default=1024*1024*4,
        help='Number of particles to read per rank. A larger number usually means faster IO, but less memory for the FFT mesh')
parser.add_argument("--remove-cic", default='anisotropic', choices=["anisotropic","isotropic", "none"],
        help='deconvolve cic, anisotropic is the proper way, see http://www.personal.psu.edu/duj13/dissertation/djeong_diss.pdf')
parser.add_argument("--remove-shotnoise", action='store_true', default=False,
        help='Remove shotnoise')
parser.add_argument("--Nmu", type=int, default=5,
        help='the number of mu bins to use; if `mode = 1d`, then `Nmu` is set to 1' )
parser.add_argument("--los", choices="xyz", default='z',
        help="the line-of-sight direction, which the angle `mu` is defined with respect to")

# parse
ns = parser.parse_args()

#--------------------------------------------------
# done with the parser. now do the real calculation
#--------------------------------------------------

from nbodykit.measurepower import measurepower
from pypm.particlemesh import ParticleMesh
from pypm.transfer import TransferFunction
from mpi4py import MPI

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

def main():

    if MPI.COMM_WORLD.rank == 0:
        print 'importing done'

    chain = [
        TransferFunction.NormalizeDC,
        TransferFunction.RemoveDC,
    ]

    if ns.remove_cic == 'anisotropic':
        chain.append(AnisotropicCIC)
    if ns.remove_cic == 'isotropic':
        chain.append(IsotropicCIC)

    # setup the particle mesh object
    pm = ParticleMesh(ns.BoxSize, ns.Nmesh, dtype='f4')

    # paint first input
    Ntot1 = ns.inputs[0].paint(ns, pm)

    # painting
    if MPI.COMM_WORLD.rank == 0:
        print 'painting done'
    pm.r2c()
    if MPI.COMM_WORLD.rank == 0:
        print 'r2c done'

    # filter the field 
    pm.transfer(chain)

    # do the cross power
    do_cross = len(ns.inputs) > 1 and ns.inputs[0] != ns.inputs[1]

    if do_cross:
        c1 = pm.complex.copy()

        Ntot2 = ns.inputs[1].paint(ns, pm)

        if MPI.COMM_WORLD.rank == 0:
            print 'painting 2 done'

        pm.r2c()
        if MPI.COMM_WORLD.rank == 0:
            print 'r2c 2 done'

        # filter the field 
        pm.transfer(chain)
        c2 = pm.complex
  
    # do the auto power
    else:
        c1 = pm.complex
        c2 = pm.complex

        Ntot2 = Ntot1 

    if ns.remove_shotnoise and not do_cross:
        shotnoise = pm.BoxSize ** 3 / (1.0*Ntot1)
    else:
        shotnoise = 0
 
    # only need one mu bin if 1d case is requested
    if ns.mode == "1d": ns.Nmu = 1 

    # do the calculation
    meta = {'box_size':pm.BoxSize, 'N1':Ntot1, 
            'N2':Ntot2, 'shot_noise': shotnoise}
    result = measurepower(pm, c1, c2, ns.Nmu, binshift=ns.binshift, 
                            shotnoise=shotnoise, los=ns.los)
    
    # format the output appropriately
    if ns.mode == "1d":
        # this writes out 0 -> mean k, 2 -> mean power, 3 -> number of modes
        meta['edges'] = result[-1][0] # write out kedges as metadata
        result = map(numpy.ravel, (result[i] for i in [0, 2, 3]))
    elif ns.mode == "2d":
        result = dict(zip(['k','mu','power','modes','edges'], result))
        
    if MPI.COMM_WORLD.rank == 0:
        print 'measure'
        storage = plugins.PowerSpectrumStorage.get(ns.mode, ns.output)
        storage.write(result, **meta)
 
main()
