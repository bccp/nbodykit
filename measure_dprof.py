from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser

parser = ArgumentParser("Measuring density profile of halos",
        description=
     """ Measuring the average halo density profile. 
     """,
        epilog=
     """
        This script is written by Man-Yat Chu, as part of `nbodykit'. 
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
parser.add_argument("nbins", 
        help='number of bins for the density profile.', type=int)
parser.add_argument("mbins", 
        help='number of bins for different masses of halos.', type=int)
parser.add_argument("m0", 
        help='m0, the mass of particle', type=float)
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
    if comm.rank == 0:
        h = files.HaloFile(ns.halocatalogue)
        #print h.read_pos().shape()
        N = h.read_mass()
        halo_pos = h.read_pos()*ns.boxsize
    else:
        N = None
        halo_pos = None
    N = comm.bcast(N)
    halo_pos = comm.bcast(halo_pos)

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
    nbin=ns.nbins
    mbin=ns.mbins
    den_prof=numpy.zeros((nbin-2, mbin+1))
    count_prof=numpy.zeros((nbin+1, mbin+1))
    bins=numpy.linspace(0,ns.boxsize * 0.1,nbin)
    mass_bins=numpy.linspace(numpy.amin(N),numpy.amax(N),mbin)
    left=bins[0:-1]
    right=bins[1:]
    centre=(left+right)/2
    mleft=mass_bins[0:-1]
    mright=mass_bins[1:]
    mcentre=(mleft+mright)/2
    m_ind = numpy.digitize(N, mass_bins)

    for l in ul:
        if l == 0: 
            continue
        start = data['Label'].searchsorted(l, side='left')
        end = data['Label'].searchsorted(l, side='right')
        pos = data['Position'][start:end]*ns.boxsize
        dist = distp(halo_pos[l,:], pos[:,:], ns.boxsize)
        count_prof[:,m_ind[l]]+=numpy.bincount(numpy.digitize(dist, bins),minlength=nbin+1)
        if l % 1000 == 0:
            print l
    
    count_prof = comm.allreduce(count_prof)
    shell_vol = 4 * 3.1416 / 3. * numpy.diff(bins ** 3)
    den_prof = count_prof[1:-1,:] /shell_vol[:,None] / len(halo_pos) / (Ntot / ns.boxsize ** 3) - 1

    if comm.rank == 0:
        if ns.output != '-':
            ff = open(ns.output, 'w')
            print ff
        else:
            ff = stdout
        with ff:
            ff.write('# centre of the mass bins %f' %mcentre) #######Still in progress of writing header
            numpy.savetxt(ff, zip(centre, den_prof))



main()
