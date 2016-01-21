from __future__ import print_function

from sys import argv
import logging

from nbodykit.plugins import ArgumentParser
from nbodykit.extensionpoints import DataSource
import numpy
import h5py

parser = ArgumentParser(None,
        description=
        """
        Create a subsample from a data source. The number density is calculated,
        but if a mass per particle is given, the density is calculated.
        """,
        epilog=
        """
        This script is written by Yu Feng, as part of `nbodykit'. 
        """
        )

h = "Data source to read particle position:\n\n"
parser.add_argument("datasource", type=DataSource.create,
        help=h + DataSource.format_help())
parser.add_argument("Nmesh", type=int,
        help='Size of FFT mesh for painting')
parser.add_argument("smoothing", type=float, 
        help='Smoothing Length in distance units. 
              It has to be greater than the mesh resolution. 
              Otherwise the code will die')
parser.add_argument("output", help='output file; convention is to end with .subsample. format needs to be documented.')


ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)
from mpi4py import MPI

import nbodykit
from pmesh import particlemesh
def main():
    comm = MPI.COMM_WORLD
    pm = ParticleMesh(ns.datasource.BoxSize, ns.Nmesh, dtype='f4', comm=comm)

