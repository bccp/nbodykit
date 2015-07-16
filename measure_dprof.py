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
parser.add_argument("Rmax", 
        help='Rmax physical units', type=float)
parser.add_argument("--Rbins", 
        help='number of R bins for the density profile.', type=int, default=100)
parser.add_argument("--logMedges", 
        help='edges for different mass bins of halos.', 
        type=float, nargs='+', default=[12, 12.5, 13, 13.5, 14])
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
    diff = numpy.abs(diff)
    diff[diff > 0.5 * boxsize]  -= boxsize
    return numpy.einsum('ij,ij->i', diff, diff) ** 0.5

def main():
    comm = MPI.COMM_WORLD
 
    if comm.rank == 0:
        h = files.HaloFile(ns.halocatalogue)
        #print h.read_pos().shape()
        N = h.read_mass()
        halo_pos = h.read_pos()
    else:
        N = None
        halo_pos = None
    N = comm.bcast(N)
    halo_pos = comm.bcast(halo_pos)
    halo_mass = N * ns.m0

    halo_ind = numpy.digitize(numpy.log10(halo_mass), ns.logMedges)
    Nhalo_per_massbin = numpy.bincount(halo_ind, minlength=len(ns.logMedges) + 1)

    redges = numpy.linspace(0, ns.Rmax / ns.boxsize, ns.Rbins + 2, endpoint=True)[1:]
    den_prof = numpy.zeros((ns.Rbins, len(ns.logMedges) - 1))
    count_prof = numpy.zeros((ns.Rbins + 2, len(ns.logMedges) + 1))

    Ntot = 0
    for round, (P, PL) in enumerate(
            zip(files.read(comm, 
                ns.snapfilename, 
                files.TPMSnapshotFile, 
                columns=['Position', 'Velocity'], 
                bunchsize=ns.bunchsize),
            files.read(comm, 
                ns.halolabel, 
                files.HaloLabelFile, 
                columns=['Label'],
                bunchsize=ns.bunchsize),
            )):
        m_ind = halo_ind[PL['Label']]
        center = halo_pos[PL['Label']]

        dist = distp(center, P['Position'], 1.0)
        d_ind = numpy.digitize(dist, redges)
        
        ind = numpy.ravel_multi_index((d_ind, m_ind), 
            count_prof.shape)

        count_prof.flat += numpy.bincount(ind, minlength=count_prof.size)
        Ntot += len(PL['Label'])

    Ntot = comm.allreduce(Ntot)
    count_prof = comm.allreduce(count_prof)
    count_prof = numpy.cumsum(count_prof, axis=0)
    vol = 4 * numpy.pi / 3. * redges ** 3
    # this is over density averaged over all halos in this mass bin
    den_prof = count_prof[:-1, 1:-1] / vol[:, None] / \
        Nhalo_per_massbin[None, 1:-1] / Ntot - 1

    if comm.rank == 0:
        if ns.output != '-':
            ff = open(ns.output, 'w')
        else:
            ff = stdout
        with ff:
            ff.write('# nhalo_per_massbin: %s\n' % ' '.join([str(n) for n in Nhalo_per_massbin]))
            ff.write('# edges of the mass bins %s\n' % 
                ' '.join([str(m) for m in ns.logMedges])
            )
            ff.write('# r cumulative_dens_prof\n')
            numpy.savetxt(ff, numpy.concatenate(
            (redges[:, None] * ns.boxsize, den_prof), axis=-1))



main()
