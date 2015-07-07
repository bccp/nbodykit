from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser

parser = ArgumentParser("Measuring 1 halo term",
        description=
     """ Measuring the paircounting of particles within a halo
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
parser.add_argument("Nmesh", 
        help='Nmesh', type=int)
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
    diff = numpy.abs(diff)
    diff[diff > 0.5 * boxsize]  -= boxsize
    return numpy.einsum('ij,ij->i', diff, diff) ** 0.5

def main():
    comm = MPI.COMM_WORLD
    SNAP, LABEL = None, None
    if comm.rank == 0:
        SNAP = files.Snapshot(ns.snapfilename, files.TPMSnapshotFile)
        LABEL = files.Snapshot(ns.halolabel, files.HaloLabelFile)

    SNAP = comm.bcast(SNAP)
    LABEL = comm.bcast(LABEL)
 
    Ntot = sum(SNAP.npart)
    assert Ntot == sum(LABEL.npart)

    h = files.HaloFile(ns.halocatalogue)
    #print h.read_pos().shape()
    N = h.read_mass()

    N0 = Ntot - sum(N[1:])
    # halos are assigned to ranks 0, 1, 2, 3 ...
    halorank = numpy.arange(len(N)) % comm.size
    # but non halos are special we will fix it later.
    halorank[0] = -1

    NonhaloStart = comm.rank * int(N0) // comm.size
    NonhaloEnd   = (comm.rank + 1)* int(N0) // comm.size

    myNtotal = numpy.sum(N[halorank == comm.rank], dtype='i8') + (NonhaloEnd - NonhaloStart)

    print("Rank %d NonhaloStart %d NonhaloEnd %d myNtotal %d" %
            (comm.rank, NonhaloStart, NonhaloEnd, myNtotal))

    data = numpy.empty(myNtotal, dtype=[
                ('Position', ('f4', 3)), 
                ('Label', ('i4')), 
                ('Rank', ('i4')), 
                ])

    allNtotal = comm.allgather(myNtotal)
    start = sum(allNtotal[:comm.rank])
    end = sum(allNtotal[:comm.rank+1])
    data['Position'] = SNAP.read("Position", start, end)
    data['Label'] = LABEL.read("Label", start, end)
    data['Rank'] = halorank[data['Label']]
    # now assign ranks to nonhalo particles
    nonhalomask = (data['Label'] == 0)

    nonhalocount = comm.allgather(nonhalomask.sum())

    data['Rank'][nonhalomask] = (sum(nonhalocount[:comm.rank]) + numpy.arange(nonhalomask.sum())) % comm.size

    mpsort.sort(data, orderby='Rank')

    arg = data['Label'].argsort()
    data = data[arg]
    
    ul = numpy.unique(data['Label'])
    nbin=ns.Nmesh
    den_prof=numpy.zeros(nbin+1)
    count_prof=numpy.zeros(nbin+1)
    bins=numpy.linspace(0,ns.boxsize,nbin)
    left=bins[0:-1]
    right=bins[1:]
    centre=(left+right)/2
    halo_pos = h.read_pos()*ns.boxsize

    for l in ul:
        if l == 0: 
            continue
        start = data['Label'].searchsorted(l, side='left')
        end = data['Label'].searchsorted(l, side='right')
        pos = data['Position'][start:end]*ns.boxsize
        dist = distp(halo_pos[l,:], pos[:,:], ns.boxsize)
        count_prof+=numpy.bincount(numpy.digitize(dist, bins),minlength=nbin+1)
        if l % 1000 == 0:
            print l
    
    shell_vol = 4*3.1416*ns.boxsize/nbin*centre**2
    den_prof = count_prof[1:-1]/shell_vol

    if comm.rank == 0:
        if ns.output != '-':
            ff = open(ns.output, 'w')
            print ff
        else:
            ff = stdout
        with ff:
            numpy.savetxt(ff, zip(centre, den_prof))



main()
