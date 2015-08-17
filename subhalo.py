from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser
import h5py

parser = ArgumentParser("Subhalo finder ",
        description=
     """ 
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

parser.add_argument("linklength", type=float,
        help='Linking length of subhalos, in units of mean particle seperation')

parser.add_argument("vfactor", type=float, default=0.368,
        help='velocity linking length in units of 1d velocity dispersion.')

parser.add_argument("--Nmin", type=int,
        help='minimal length of halo to do FOF6D')

parser.add_argument("output", help='write output to this file')

ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)


import numpy
import nbodykit
from nbodykit import files
from nbodykit.distributedarray import DistributedArray

import mpsort
from mpi4py import MPI
from kdcount import cluster

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

    mystart = Ntot * comm.rank // comm.size
    myend = Ntot * (comm.rank + 1) // comm.size

    data = numpy.empty(myend - mystart, dtype=[
                ('Position', ('f4', 3)), 
                ('Velocity', ('f4', 3)), 
                ('Label', ('i4')), 
                ('Rank', ('i4')), 
                ])
    data['Position'] = SNAP.read("Position", mystart, myend)
    data['Velocity'] = SNAP.read("Velocity", mystart, myend)
    data['Label'] = LABEL.read("Label", mystart, myend)

    # remove particles not in any halos
    data = data[data['Label'] != 0]

    Nhalo = comm.allreduce(data['Label'].max(), op=MPI.MAX) + 1

    # now count number of particles per halo
    data['Rank'] = data['Label'] % comm.size

    Nlocal = numpy.bincount(data['Rank'], minlength=comm.size)
    Nlocal = comm.allreduce(Nlocal, op=MPI.SUM)[comm.rank]
    data2 = numpy.empty(Nlocal, data.dtype)

    mpsort.sort(data, orderby='Rank', out=data2)

    assert (data2['Rank'] == comm.rank).all()

    data2.sort(order=['Label'])

    cat = []
    for label in numpy.unique(data2['Label']):
        hstart = data2['Label'].searchsorted(label, side='left')
        hend = data2['Label'].searchsorted(label, side='right')
        if hstart - hend < ns.Nmin: continue
        assert(data2['Label'][hstart:hend] == label).all()
        print 'Halo', label
        cat.append(subfof(data2['Position'][hstart:hend], data2['Velocity'][hstart:hend], 
            ns.linklength * 1.0 / Ntot ** 0.3333, ns.vfactor, label, Ntot))
    cat = numpy.concatenate(cat, axis=0)
    cat = comm.gather(cat)

    if comm.rank == 0:
        cat = numpy.concatenate(cat, axis=0)
        print cat
        with h5py.File(ns.output, mode='w') as f:
            dataset = f.create_dataset('Subhalo', data=cat)
            dataset.attrs['LinkingLength'] = ns.linklength
            dataset.attrs['VFactor'] = ns.vfactor
            dataset.attrs['Ntot'] = Ntot

def subfof(pos, vel, ll, vfactor, haloid, Ntot):
    first = pos[0].copy()
    pos -= first
    pos[pos > 0.5]  -= 1.0 
    pos[pos < -0.5] += 1.0 
    pos += first

    oldvel = vel.copy()
    vmean = vel.mean(axis=0, dtype='f8')
    vel -= vmean
    sigma_1d = (vel** 2).mean(dtype='f8') ** 0.5
    vel /= (vfactor * sigma_1d)
    vel *= ll
    data = numpy.concatenate(( pos, vel), axis=1)
    #data = pos

    data = cluster.dataset(data)
    fof = cluster.fof(data, linking_length=ll, np=0)
    Nsub = (fof.length > 20).sum()
    output = numpy.empty(Nsub, dtype=[
        ('Position', ('f4', 3)),
        ('Velocity', ('f4', 3)),
        ('R200', 'f4'),
        ('R1200', 'f4'),
        ('Length', 'i4'),
        ('HaloID', 'i4'),
        ])
    output['Position'][...] = fof.center()[:Nsub, :3]
    output['Length'][...] = fof.length[:Nsub]
    output['HaloID'][...] = haloid

    for i in range(3):
        output['Velocity'][..., i] = fof.sum(oldvel[:, i])[:Nsub] / output['Length']

    del fof
    del data
    data = cluster.dataset(pos)
    for i in range(Nsub):
        center = output['Position'][i] 
        r1 = (1.0 * output['Length'][i] / Ntot) ** 0.3333 * 3
        output['R200'][i] = so(center, data, r1, Ntot, 200.)
        output['R1200'][i] = so(center, data, output['R200'][i] * 0.5, Ntot, 1200.)
    return output

def so(center, data, r1, nbar, thresh=200):
    center = numpy.array([center])
    dcenter = cluster.dataset(center)
    def delta(r):
        if r == 0:
            return numpy.nan
        N = data.tree.count(dcenter.tree, [r])[0][0]
        n = N / (4 / 3 * numpy.pi * r ** 3)
        return 1.0 * n / nbar - 1
     
    d1 = delta(r1)

    while d1 > thresh:
        r1 *= 1.4
        d1 = delta(r1)
    # d1 < 200
    r2 = r1
    d2 = d1
    while d2 < thresh:
        r2 *= 0.7
        d2 = delta(r2)
    # d2 > 200

    while True:
        r = (r1 * r2) ** 0.5
        d = delta(r)
        x = (d - thresh)
        if x > 0.1:
            r2 = r
        elif x < -0.1:
            r1 = r
        else:
            return r

main()

