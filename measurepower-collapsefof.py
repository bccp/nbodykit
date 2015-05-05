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
        This script moves all particles to the halo center.
     """,
        epilog=
     """
        This script is written by Yu Feng, as part of `nbodykit'. 
        The author would like thank Marcel Schmittfull for the explanation on cic, shotnoise, and k==0 plane errors.
     """
        )

parser.add_argument("filename", 
        help='basename of the input, only runpb format is supported in this script')
parser.add_argument("halolabel", 
        help='basename of the halo label files, only nbodykit format is supported in this script')
parser.add_argument("halocatalogue", 
        help='basename of the halocatalogue, only nbodykit format is supported in this script')
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
from nbodykit import files
from nbodykit.measurepower import measurepower

from pypm.particlemesh import ParticleMesh
from pypm.transfer import TransferFunction
from itertools import izip

from mpi4py import MPI

def main():
    pm = ParticleMesh(ns.BoxSize, ns.Nmesh)
    if pm.comm.rank == 0:
        
        hf = files.HaloFile(ns.halocatalogue)
        nhalo = hf.nhalo
        halopos = hf.read_pos()
        halopos = pm.comm.bcast(halopos)
    else:
        halopos = pm.comm.bcast(None)

    Ntot = 0
    for round, (P, PL) in enumerate(izip(
                files.read(pm.comm, ns.filename, files.TPMSnapshotFile, 
                    columns=['Position'], bunchsize=ns.bunchsize),
                files.read(pm.comm, ns.halolabel, files.HaloLabelFile, 
                    columns=['Label'], bunchsize=ns.bunchsize),
                )):
        mask = PL['Label'] != 0
        logging.info("Number of particles in halos is %d" % mask.sum())
        P['Position'][mask] = halopos[PL['Label'][mask]]

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

    if ns.remove_shotnoise:
        shotnoise = pm.BoxSize ** 3 / Ntot
    else:
        shotnoise = 0

    k, p = measurepower(pm, ns.binshift, ns.remove_cic, shotnoise)

    if pm.comm.rank == 0:
        if ns.output != '-':
            myout = open(ns.output, 'w')
        else:
            myout = stdout
        numpy.savetxt(myout, zip(k, p), '%0.7g')
        myout.flush()

        hf = files.HaloFile(ns.halocatalogue)
        mass = hf.read_mass()
        M1 = ((mass[1:] * 1.0)** 2).sum(dtype='f8') / (1.0 * Ntot) ** 2 * ns.BoxSize ** 3
        logging.info("p[-inf] = %g, M1 = %g" % (p[-1], M1))

main()
