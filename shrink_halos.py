from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser

parser = ArgumentParser("Shrinking profile of halos",
        description=
     """ Shrinking the halo profile. 
     """,
        epilog=
     """
        This script is written by Yu Feng, as part of `nbodykit'. 
     """
        )

parser.add_argument("snapfilename", 
        help='basename of the snapshot, only runpb format is supported in this script')
parser.add_argument("halolabel", 
        help='basename of the halo label files, only nbodykit format is supported in this script')
parser.add_argument("halocatalogue", 
        help='halocatalogue')
parser.add_argument("boxsize", 
        help='size of box', type=float)
parser.add_argument("m0", 
        help='m0, the mass of a particle', type=float)
parser.add_argument("--bunchsize", 
        help='number of particles to process per bunch', type=int, default=1024 * 1024)
parser.add_argument("output", help='write output to this file')

ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)

import numpy
import nbodykit
from nbodykit import files
from nbodykit import halos
from nbodykit.corrfrompower import corrfrompower
from kdcount import correlate
import mpsort
from mpi4py import MPI

def distp(center, pos, boxsize):
    diff = (pos - center)
    diff[diff > 0.5 * boxsize]  -= boxsize
    diff[diff < -0.5 * boxsize]  += boxsize
    return diff
def model(R, Mhalo):
    a = 0.501237530292
    R200 = a * (Mhalo / 1e13) ** 0.333333
    x = R / R200
    Rc = R * numpy.log(1 + x**0.36) * 1.15
    Rc = Rc.clip(0, R)
    return Rc

def main():
    comm = MPI.COMM_WORLD
 
    if comm.rank == 0:
        h = files.HaloFile(ns.halocatalogue)
        #print h.read_pos().shape()
        N = h.read('Mass')
        halo_pos = h.read('Position')
    else:
        N = None
        halo_pos = None
    N = comm.bcast(N)
    halo_pos = comm.bcast(halo_pos)
    halo_mass = N * ns.m0

    if comm.rank == 0:
        snapfile = files.Snapshot(ns.snapfilename, files.TPMSnapshotFile)
        labelfile = files.Snapshot(ns.halolabel, files.HaloLabelFile)
        npart = snapfile.npart
        output = files.Snapshot.create(ns.output, files.TPMSnapshotFile, npart)
        comm.bcast((snapfile, labelfile, output))
    else:
        snapfile, labelfile, output = comm.bcast(None) 
    comm.barrier()

    Ntot = sum(snapfile.npart)
    for i in range(0, Ntot, ns.bunchsize):
        start, end, junk = slice(i, i + ns.bunchsize).indices(Ntot)
        position = snapfile.read('Position', start, end)
        velocity = snapfile.read('Velocity', start, end)
        ID = snapfile.read('ID', start, end)
        label = labelfile.read('Label', start, end)

        mask = label != 0
        label = label[mask]
        position2 = position[mask]
        center = halo_pos[label]
        mass = halo_mass[label]
        dist = distp(center, position2, 1.0)
        rwrong = numpy.einsum('ij,ij->i', dist, dist) ** 0.5

        rcorrect = model(rwrong, mass)
        dist *= (rcorrect / rwrong)[:, None]
        position2 = (center + dist) % 1.0

        position[mask] = position2
        output.write('Position', start, position)
        #output.write('Velocity', start, velocity)
        output.write('ID', start, ID)

main()
