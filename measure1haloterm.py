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
parser.add_argument("boxsize", 
        help='size of box', type=float)
parser.add_argument("Nmesh", 
        help='Nmesh', type=int)
parser.add_argument("halolabel", 
        help='basename of the halo label files, only nbodykit format is supported in this script')
parser.add_argument("halocatalogue", 
        help='halocatalogue')
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

    N = h.read_mass()

    # halos are assigned to ranks 0, 1, 2, 3 ...
    halorank = numpy.arange(len(N)) % comm.size
    # but non halos are special we will fix it later.
    halorank[0] = -1

    NonhaloStart = comm.rank * numpy.int64(N[0]) // comm.size
    NonhaloEnd   = (comm.rank + 1)* numpy.int64(N[0]) // comm.size

    myNtotal = numpy.sum(N[halorank == comm.rank], dtype='i8') + (NonhaloEnd - NonhaloStart)

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

    bins = correlate.RBinning(40./ ns.boxsize, Nbins=Nmesh)
    sum1 = numpy.zeros(len(bins.centers))

    for l in ul:
        if l == 0: continue
        start = data['Label'].searchsorted(l, side='left')
        end = data['Label'].searchsorted(l, side='right')
        pos = data['Position'][start:end]
        dataset = correlate.points(pos, boxsize=1.0)
        result = correlate.paircount(dataset, dataset, bins, np=0)
        sum1 += result.sum1

    sum1 = comm.allreduce(sum1, MPI.SUM)
    Ntot = sum(SNAP.npart)
    RR = 4. / 3 * numpy.pi * numpy.diff(bins.edges**3) * (Ntot *Ntot)

    k = numpy.arange(ns.Nmesh // 2) * 2 * numpy.pi / ns.boxsize
    # asymtotically zero at r. The mean doesn't matter as 
    # we don't use zero k mode anyways.
    k, p = corrfrompower(bins.centers * ns.boxsize, sum1 / RR, R=k)
    # inverse FT factor
    p *= (2 * numpy.pi) ** 3

    if comm.rank == 0:

        if ns.output != '-':
            ff = open(ns.output, 'w')
        else:
            ff = stdout
        with ff:
    #        numpy.savetxt(ff, zip(bins.centers, sum1 / RR - 1.0))
            numpy.savetxt(ff, zip(k, p))

if __name__ == '__main__':
    main()
