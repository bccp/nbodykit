from sys import argv
from sys import stdout
from sys import stderr
import logging
import functools

from argparse import ArgumentParser, RawTextHelpFormatter
from power_io.field_input import InputFieldType, register_field_types

#--------------------------------------------------
# setup the parser
#--------------------------------------------------

# initialize the parser
parser = ArgumentParser("Parallel Power Spectrum Calculator",
        formatter_class=RawTextHelpFormatter,
        fromfile_prefix_chars="@",
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
        
# override file reading option to treat each space-separated word as 
# an argument and ignore comments. Can put option + value on same line
def line_reader(self, line):
    for arg in line.split():
        if not arg.strip():
            continue
        if arg[0] == '#':
            break
        yield arg
parser.convert_arg_line_to_args = functools.partial(line_reader, parser)

# add the positional arguments
parser.add_argument("mode", choices=["2d", "1d"]) 
parser.add_argument("BoxSize", type=float, help='BoxSize in Mpc/h')
parser.add_argument("Nmesh", type=int, help='size of calculation mesh, recommend 2 * Ngrid')
parser.add_argument("output", help='write power to this file') 

# add the input field types, first registering all defined field types 
register_field_types(InputFieldType) 
h = "one or two input fields, specifed as:\n\n"
parser.add_argument("inputs", nargs="+", type=InputFieldType, help=h+InputFieldType.format_help())

# add the optional arguments
parser.add_argument("--binshift", type=float, default=0.0,
        help='Shift the bin center by this fraction of the bin width. Default is 0.0. Marcel uses 0.5. this shall rarely be changed.' )
parser.add_argument("--bunchsize", type=int, default=1024*1024*4,
        help='Number of particles to read per rank. A larger number usually means faster IO, but less memory for the FFT mesh')
parser.add_argument("--remove-cic", default='anisotropic', choices=["anisotropic","isotropic", "none"],
        help='deconvolve cic, anisotropic is the proper way, see http://www.personal.psu.edu/duj13/dissertation/djeong_diss.pdf')
parser.add_argument("--Nmu", type=int, default=5,
        help='the number of mu bins to use' )

# parse
ns = parser.parse_args()

#--------------------------------------------------
# done with the parser. now do the real calculation
#--------------------------------------------------

logging.basicConfig(level=logging.DEBUG)

import numpy
import nbodykit
from nbodykit.measurepower import measure2Dpower, measurepower
from pypm.particlemesh import ParticleMesh
from mpi4py import MPI

def main():


    if MPI.COMM_WORLD.rank == 0:
        print 'importing done'

    # setup the particle mesh object
    pm = ParticleMesh(ns.BoxSize, ns.Nmesh, dtype='f4')

    # sort out if we have 1 or 2 inputs
    input1 = ns.inputs[0]
    input2 = None
    if len(ns.inputs) > 1:
        input2 = ns.inputs[1]
    Ntot1 = input1.painter(ns, input1, pm)

    # painting
    if MPI.COMM_WORLD.rank == 0:
        print 'painting done'
    pm.r2c()
    if MPI.COMM_WORLD.rank == 0:
        print 'r2c done'

    # do the cross power
    if input2 is not None and input2 != input1:
        complex = pm.complex.copy()
        numpy.conjugate(complex, out=complex)

        Ntot2 = input2.painter(ns, input2, pm)
        if MPI.COMM_WORLD.rank == 0:
            print 'painting 2 done'
        pm.r2c()
        if MPI.COMM_WORLD.rank == 0:
            print 'r2c 2 done'
        complex *= pm.complex
        complex **= 0.5

        if MPI.COMM_WORLD.rank == 0:
            print 'cross done'
    # do the auto power
    else:
        complex = pm.complex
    
    # call the appropriate function for 1d/2d cases
    if ns.mode == "1d":
        do1d(pm, complex, ns)

    if ns.mode == "2d":
        do2d(pm, complex, ns)
    
def do2d(pm, complex, ns):
    k, mu, p, N, edges = measure2Dpower(pm, complex, ns.binshift, ns.remove_cic, 0, ns.Nmu)
  
    if MPI.COMM_WORLD.rank == 0:
        print 'measure'

    if pm.comm.rank == 0:
        if ns.output != '-':
            myout = open(ns.output, 'w')
        else:
            myout = stdout
        numpy.savetxt(myout, zip(k.flat, mu.flat, p.flat, N.flat), '%0.7g')
        myout.flush()

def do1d(pm, complex, ns):
    k, p = measurepower(pm, complex, ns.binshift, ns.remove_cic, 0)

    if MPI.COMM_WORLD.rank == 0:
        print 'measure'

    if pm.comm.rank == 0:
        if ns.output != '-':
            myout = open(ns.output, 'w')
        else:
            myout = stdout
        numpy.savetxt(myout, zip(k, p), '%0.7g')
        myout.flush()

main()
