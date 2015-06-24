from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser

parser = ArgumentParser("Serial Halo Painter",
        description=
     """ Paint the halo number density field to a grid.
         The output is a binary file of single precision float..
     """,
        epilog=
     """
        This script is written by Yu Feng, as part of `nbodykit'. 
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
parser.add_argument("output", help='write mesh to this file')

ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)

import numpy
import nbodykit
from nbodykit import files

from pypm import cic
from itertools import izip

def paint_halos(Nmesh, halocatalogue, BoxSize, m0, massmin, massmax):
    canvas = numpy.zeros((Nmesh, Nmesh, Nmesh), dtype='f4')

    hf = files.HaloFile(halocatalogue)
    nhalo = hf.nhalo
    halopos = numpy.float32(hf.read_pos())
    halomass = numpy.float32(hf.read_mass() * m0)
    logmass = numpy.log10(halomass)
    mask = logmass > massmin
    mask &= logmass < massmax
    halopos = halopos[mask]
    logging.info("total number of halos in mass range is %d" % mask.sum())
    
    halopos *= Nmesh

    cic.paint(halopos, canvas, period=Nmesh)
    return canvas

def main():
    canvas = paint_halos(ns.Nmesh, ns.halocatalogue, ns.BoxSize, ns.m0, ns.massmin, ns.massmax)
    canvas /= (ns.BoxSize  / ns.Nmesh) ** 3
    canvas.tofile(ns.output)

main()
