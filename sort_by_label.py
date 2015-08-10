from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser

parser = ArgumentParser("Sorting Particles by the halo label",
        description=
     """ Sorting particles by the Halo Label.
         Particles not in any halo is put to the end. The first halo
         offsets at 0.
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
    mystart = Ntot * comm.rank // comm.size
    myend =  Ntot * (comm.rank + 1) // comm.size

    for field in ['Position', 'Velocity', 'ID']:
        content = snapfile.read(field, mystart, myend)
        if len(content.shape) == 1:
            dtype = numpy.dtype(content.dtype)
        else:
            dtype = numpy.dtype((content.dtype, content.shape[1:]))
        data = numpy.empty(myend - mystart, dtype=[
            ('Label', 'u8'),
            ('content', dtype),
                ])
        data['content'] = content
        content = None
        data['Label'] = labelfile.read('Label', mystart, myend)
        nonhalo = data['Label'] == 0
        data['Label'][nonhalo] = numpy.iinfo('u8').max
        mpsort.sort(data, orderby='Label')
        
        output.write(field, mystart, data['content'])

main()
