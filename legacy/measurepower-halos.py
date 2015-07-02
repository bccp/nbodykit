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

parser.add_argument("halocatalogue", 
        help='basename of the halocatalogue, only nbodykit format is supported in this script')
parser.add_argument("massmin", type=float,
        help='min mass in log10(Msun/h) ')
parser.add_argument("massmax", type=float,
        help='max mass in log10(Msun/h) ')
parser.add_argument("m0", type=float, 
        help='mass of a single particle in Msun/h')
parser.add_argument("BoxSize", type=float, 
        help='BoxSize in Mpc/h')
parser.add_argument("Nmesh", type=int, 
        help='size of calculation mesh, recommend 2 * Ngrid')
parser.add_argument("output", help='write power to this file')

parser.add_argument("--binshift", type=float, default=0.0,
        help='Shift the bin center by this fraction of the bin width. Default is 0.0. Marcel uses 0.5. this shall rarely be changed.' )
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

def paint_halos(pm, halocatalogue, BoxSize, m0, massmin, massmax):
    if pm.comm.rank == 0:
        hf = files.HaloFile(halocatalogue)
        nhalo = hf.nhalo
        halopos = numpy.float32(hf.read_pos())
        halomass = numpy.float32(hf.read_mass() * m0)
        logmass = numpy.log10(halomass)
        mask = logmass > massmin
        mask &= logmass < massmax
        print logmass
        halopos = halopos[mask]
        logging.info("total number of halos in mass range is %d" % mask.sum())
    else:
        halopos = numpy.empty((0, 3), dtype='f4')
        halomass = numpy.empty(0, dtype='f4')

    P = {}
    P['Position'] = halopos

    Ntot = len(halopos)
    Ntot = pm.comm.bcast(Ntot)

    P['Position'] *= BoxSize

    layout = pm.decompose(P['Position'])
    tpos = layout.exchange(P['Position'])
    pm.paint(tpos)

    npaint = pm.comm.allreduce(len(tpos), op=MPI.SUM) 
    return Ntot

def main():
    pm = ParticleMesh(ns.BoxSize, ns.Nmesh)

    Ntot = paint_halos(pm, ns.halocatalogue, ns.BoxSize, ns.m0, ns.massmin, ns.massmax)

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
