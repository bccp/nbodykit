from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser

parser = ArgumentParser("Parallel Power Spectrum Calculator",
        description=
     """Calculating matter power spectrum from RunPB input files. 
        Output is written to stdout, in Mpc/h units. 
        PowerSpectrum is the true one, without (2 pi) ** 3 factor. (differ from Gadget/NGenIC internal)

     """,
        epilog=
     """
        This script is written by Yu Feng, as part of `nbodykit'. 
        The author would like thank Marcel Schmittfull for the explanation on cic, shotnoise, and k==0 plane errors.
     """
        )

parser.add_argument("filename", 
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
parser.add_argument("--remove-shotnoise", action='store_true', default=False, 
        help='removing the shot noise term')

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
    for round, P in enumerate(read(pm.comm, ns.filename, TPMSnapshotFile, 
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
    pm = ParticleMesh(ns.BoxSize, ns.Nmesh)

    Ntot = paint_darkmatter(pm, ns.filename, TPMSnapshotFile)

    if ns.remove_shotnoise:
        shotnoise = pm.BoxSize ** 3 / Ntot
    else:
        shotnoise = 0

    pm.r2c()

    k, p = measurepower(pm, pm.complex, ns.binshift, ns.remove_cic, shotnoise)

    if pm.comm.rank == 0:
        if ns.output != '-':
            myout = open(ns.output, 'w')
        else:
            myout = stdout
        numpy.savetxt(myout, zip(k, p), '%0.7g')
        myout.flush()

main()
