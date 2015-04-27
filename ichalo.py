from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser

parser = ArgumentParser("Finding IC position of halos",
        description=
     """IC position of halos are defined as the center of mass position of
        halo particles at the IC
     """,
        epilog=
     """
        This script is written by Yu Feng, as part of `nbodykit'. 
     """
        )

parser.add_argument("icfilename", 
        help='basename of the ic, only runpb format is supported in this script')
parser.add_argument("snapfilename", 
        help='basename of the snapshot, only runpb format is supported in this script')
parser.add_argument("halolabel", 
        help='basename of the halo label files, only nbodykit format is supported in this script')
parser.add_argument("output", help='write output to this file')

ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)

import numpy
import nbodykit
from nbodykit import files
from nbodykit import halos

import mpsort
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    IC, SNAP, LABEL = None, None, None
    if comm.rank == 0:
        IC = files.Snapshot(ns.icfilename, files.TPMSnapshotFile)
        SNAP = files.Snapshot(ns.snapfilename, files.TPMSnapshotFile)
        LABEL = files.Snapshot(ns.halolabel, files.HaloLabelFile)

    IC = comm.bcast(IC)
    SNAP = comm.bcast(SNAP)
    LABEL = comm.bcast(LABEL)
 
    Ntot = sum(IC.npart)
    assert Ntot == sum(SNAP.npart)
    assert Ntot == sum(LABEL.npart)

    start = comm.rank * Ntot  // comm.size
    end   = (comm.rank + 1)* Ntot  // comm.size
    data = numpy.empty(end - start, dtype=[
                ('Label', ('i4')), 
                ('ID', ('i8')), 
                ])
    data['ID'] = SNAP.read("ID", start, end)
    data['Label'] = LABEL.read("Label", start, end)

    mpsort.sort(data, orderby='ID')

    label = data['Label'].copy()

    data = numpy.empty(end - start, dtype=[
                ('ID', ('i8')), 
                ('Position', ('f4', 3)), 
                ])
    # suppose IC is sorted by ID. This is not necessarily true.
    data['Position'][:] = IC.read("Position", start, end)
    data['ID'][:] = IC.read("ID", start, end)
    mpsort.sort(data, orderby='ID')

    pos = data['Position'].copy()
    del data

    N = halos.count(label)
    hpos = halos.centerofmass(label, pos, boxsize=1.0)
    
    if comm.rank == 0:
        logging.info("Total number of halos: %d" % len(N))
        logging.info("N %s" % str(N))
        LinkingLength = LABEL.get_file(0).linking_length

        with open(ns.output + '.ichalo', 'w') as ff:
            numpy.int32(len(N)).tofile(ff)
            numpy.float32(LinkingLength).tofile(ff)
            numpy.int32(N).tofile(ff)
            numpy.float32(hpos).tofile(ff)
        print hpos
        logging.info("Written %s" % ns.output + '.ichalo')


main()
