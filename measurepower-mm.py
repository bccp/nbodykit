from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser

parser = ArgumentParser("Parallel Cross Power Spectrum Calculator",
        description=
     """Calculating cross matter power spectrum from two RunPB input files. 
        Output is written to stdout, in Mpc/h units. 
        PowerSpectrum is the true one, without (2 pi) ** 3 factor. (differ from Gadget/NGenIC internal)

     """,
        epilog=
     """
        This script is written by Yu Feng, as part of `nbodykit'. 
        The author would like thank Marcel Schmittfull for the explanation on cic, shotnoise, and k==0 plane errors.
     """
        )

parser.add_argument("filename1", 
        help='basename of the input, only runpb format is supported in this script')
parser.add_argument("filename2", 
        help='basename of the input, only runpb format is supported in this script')
parser.add_argument("BoxSize", type=float, 
        help='BoxSize in Mpc/h')
parser.add_argument("Nmesh", type=int, 
        help='size of calculation mesh, recommend 2 * Ngrid')
parser.add_argument("output", help='write power to this file')
parser.add_argument("--binshift", type=float, default=0.0,
        help='Shift the bin center by this fraction of the bin width. Default is 0.0. Marcel uses 0.5. this shall rarely be changed.' )
parser.add_argument("--bunchsize", type=int, default=1024*1024*4,
        help='Number of particles to read per rank. A larger number usually means faster IO, but less memory for the FFT mesh')
parser.add_argument("--remove-cic", default='anisotropic', choices=["anisotropic","isotropic", "none"],
        help='deconvolve cic, anisotropic is the proper way, see http://www.personal.psu.edu/duj13/dissertation/djeong_diss.pdf')

ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)

import numpy
import nbodykit
from nbodykit.files import TPMSnapshotFile, read
from nbodykit.measurepower import measurepower

from pypm.particlemesh import ParticleMesh
from pypm.transfer import TransferFunction


from mpi4py import MPI

def paint_darkmatter(pm, filename, fileformat):
    pm.real[:] = 0
    Ntot = 0
    for round, P in enumerate(read(pm.comm, filename, TPMSnapshotFile, 
                columns=['Position'], bunchsize=ns.bunchsize)):
        P['Position'] *= ns.BoxSize
        layout = pm.decompose(P['Position'])
        tpos = layout.exchange(P['Position'])
        #print tpos.shape
        pm.paint(tpos)
        npaint = pm.comm.allreduce(len(tpos), op=MPI.SUM) 
        nread = pm.comm.allreduce(len(P['Position']), op=MPI.SUM) 
        if pm.comm.rank == 0:
            logging.info('round %d, npaint %d, nread %d' % (round, npaint, nread))
        Ntot = Ntot + nread
    return Ntot

def main():

    if MPI.COMM_WORLD.rank == 0:
        print 'importing done'

    pm = ParticleMesh(ns.BoxSize, ns.Nmesh, dtype='f4')

    Ntot = paint_darkmatter(pm, ns.filename1, TPMSnapshotFile)

    if MPI.COMM_WORLD.rank == 0:
        print 'painting done'
    pm.r2c()
    if MPI.COMM_WORLD.rank == 0:
        print 'r2c done'

    complex = pm.complex.copy()
    numpy.conjugate(complex, out=complex)

    Ntot = paint_darkmatter(pm, ns.filename2, TPMSnapshotFile)
    if MPI.COMM_WORLD.rank == 0:
        print 'painting 2 done'
    pm.r2c()
    if MPI.COMM_WORLD.rank == 0:
        print 'r2c 2 done'
    complex *= pm.complex
    complex **= 0.5

    if MPI.COMM_WORLD.rank == 0:
        print 'cross done'
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
