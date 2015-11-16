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
    Rc = numpy.log(1 + x**0.36) * 1.15
    Rc = Rc.clip(0, 1)
    return Rc
def model(R, Mhalo):
    a = 0.501237530292
    R200 = a * (Mhalo / 1e13) ** 0.333333
    x = R / R200
    return f(x, R)

def f(xxx, xx2):
    rt = 0.8* xxx ** 0.25
    #mask = xx2 < 0.1
    #rt[mask] = 1.0
    return rt.clip(0, 1)

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
    for i in range(0, Ntot, ns.bunchsize * comm.size):
        if comm.rank == 0:
            print i, Ntot
        start, end, junk = slice(i, i + ns.bunchsize * comm.size).indices(Ntot)
        mystart = start + (end - start) * comm.rank // comm.size
        myend = start + (end - start) * (comm.rank + 1) // comm.size
        position = snapfile.read('Position', mystart, myend)
        velocity = snapfile.read('Velocity', mystart, myend)
        ID = snapfile.read('ID', mystart, myend)
        label = labelfile.read('Label', mystart, myend)

        mask = label != 0
        label = label[mask]
        position2 = position[mask]
        center = halo_pos[label]
        mass = halo_mass[label]
        dist = distp(center, position2, 1.0)
        rwrong = numpy.einsum('ij,ij->i', dist, dist) ** 0.5

        rwrong *= ns.boxsize
        rfact = model(rwrong, mass)
        dist *= rfact[:, None]
        position2 = (center + dist) % 1.0

        position[mask] = position2
        print 'writing at', mystart
        output.write('Position', mystart, position)
        output.write('Velocity', mystart, velocity)
        output.write('ID', mystart, ID)

main()
