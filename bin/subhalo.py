from __future__ import print_function

from sys import argv
from sys import stdout
from sys import stderr
import logging

from nbodykit.utils.pluginargparse import PluginArgumentParser
from nbodykit import plugins
import h5py

parser = PluginArgumentParser(None,
        loader=plugins.load,
        description=
     """ 
        Finding subhalos from FOF groups. This is a variant of FOF6D.
     """,
        epilog=
     """
        This script is written by Yu Feng, as part of `nbodykit'. 
     """
        )

parser.add_argument("datasource", type=plugins.DataSource.open,
        help='Data source')
parser.add_argument("halolabel", 
        help='basename of the halo label files, only nbodykit format is supported in this script')

parser.add_argument("--linklength", type=float, default=0.078,
        help='Linking length of subhalos, in units of mean particle seperation')

parser.add_argument("--vfactor", type=float, default=0.368,
        help='velocity linking length in units of 1d velocity dispersion.')

parser.add_argument("--Nmin", type=int, default=32,
        help='minimal length of halo to do FOF6D')

parser.add_argument("output", help='write output to this file')

ns = parser.parse_args()
print(ns)

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
    LABEL = None
    if comm.rank == 0:
        LABEL = files.Snapshot(ns.halolabel, files.HaloLabelFile)

    LABEL = comm.bcast(LABEL)
 
    offset = 0
    
    [Label] = files.read(comm, ns.halolabel, files.HaloLabelFile, columns=['Label'], bunchsize=None)
    Label = Label['Label']
    mask = Label != 0
    PIG = numpy.empty(mask.sum(), dtype=[
            ('Position', ('f4', 3)), 
            ('Velocity', ('f4', 3)), 
            ('Label', ('i4')), 
            ('Rank', ('i4')), 
            ])
    PIG['Label'] = Label[mask]
    del Label
    [[Position]] = ns.datasource.read(['Position'], comm, full=True)
    PIG['Position'] = Position[mask]
    del Position
    [[Velocity]] = ns.datasource.read(['Velocity'], comm, full=True)
    PIG['Velocity'] = Velocity[mask]
    del Velocity
 
    Ntot = comm.allreduce(len(mask))
    del mask

    Nhalo = comm.allreduce(
        PIG['Label'].max() if len(PIG['Label']) > 0 else 0, op=MPI.MAX) + 1

    # now count number of particles per halo
    PIG['Rank'] = PIG['Label'] % comm.size

    Nlocal = comm.allreduce(
                numpy.bincount(PIG['Rank'], minlength=comm.size)
             )[comm.rank]

    PIG2 = numpy.empty(Nlocal, PIG.dtype)

    mpsort.sort(PIG, orderby='Rank', out=PIG2)
    del PIG

    assert (PIG2['Rank'] == comm.rank).all()

    PIG2.sort(order=['Label'])

    logging.info('halos = %d', Nhalo)
    cat = []
    for haloid in numpy.unique(PIG2['Label']):
        hstart = PIG2['Label'].searchsorted(haloid, side='left')
        hend = PIG2['Label'].searchsorted(haloid, side='right')
        if hend - hstart < ns.Nmin: continue
        assert(PIG2['Label'][hstart:hend] == haloid).all()
        print('Processing', haloid)
        cat.append(
            subfof(
                PIG2['Position'][hstart:hend], 
                PIG2['Velocity'][hstart:hend], 
                ns.linklength * (ns.datasource.BoxSize.prod() / Ntot) ** 0.3333, 
                ns.vfactor, haloid, Ntot, ns.datasource.BoxSize))

    cat = numpy.concatenate(cat, axis=0)
    cat = comm.gather(cat)

    if comm.rank == 0:
        cat = numpy.concatenate(cat, axis=0)
        print(cat)
        with h5py.File(ns.output, mode='w') as f:
            dataset = f.create_dataset('Subhalos', data=cat)
            dataset.attrs['LinkingLength'] = ns.linklength
            dataset.attrs['VFactor'] = ns.vfactor
            dataset.attrs['Ntot'] = Ntot
            dataset.attrs['BoxSize'] = ns.datasource.BoxSize

def subfof(pos, vel, ll, vfactor, haloid, Ntot, boxsize):
    nbar = Ntot / boxsize.prod()
    first = pos[0].copy()
    pos -= first
    pos /= boxsize
    pos[pos > 0.5]  -= 1.0 
    pos[pos < -0.5] += 1.0 
    pos *= boxsize
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
    Nsub = 0
    while Nsub == 0:
        fof = cluster.fof(data, linking_length=ll, np=0)
        ll *= 2
        Nsub = (fof.length > 20).sum()

    output = numpy.empty(Nsub, dtype=[
        ('Position', ('f4', 3)),
        ('Velocity', ('f4', 3)),
        ('LinkingLength', 'f4'),
        ('R200', 'f4'),
        ('R500', 'f4'),
        ('R1200', 'f4'),
        ('R2400', 'f4'),
        ('R6000', 'f4'),
        ('Length', 'i4'),
        ('HaloID', 'i4'),
        ])

    output['Position'][...] = fof.center()[:Nsub, :3]
    output['Length'][...] = fof.length[:Nsub]
    output['HaloID'][...] = haloid
    output['LinkingLength'][...] = ll

    for i in range(3):
        output['Velocity'][..., i] = fof.sum(oldvel[:, i])[:Nsub] / output['Length']

    del fof
    del data
    data = cluster.dataset(pos)

    for i in range(Nsub):
        center = output['Position'][i] 
        rmax = (((pos - center) ** 2).sum(axis=-1) ** 0.5).max()
        r1 = rmax
        output['R200'][i] = so(center, data, r1, nbar, 200.)
        output['R500'][i] = so(center, data, r1, nbar, 500.)
        output['R1200'][i] = so(center, data, output['R200'][i] * 0.5, nbar, 1200.)
        output['R2400'][i] = so(center, data, output['R1200'][i] * 0.5, nbar, 2400.)
        output['R6000'][i] = so(center, data, output['R2400'][i] * 0.5, nbar, 6000.)
    return output

def so(center, data, r1, nbar, thresh=200):
    center = numpy.array([center])
    dcenter = cluster.dataset(center)

    def delta(r):
        if r < 1e-7:
            raise StopIteration
        N = data.tree.count(dcenter.tree, [r])[0][0]
        n = N / (4 / 3 * numpy.pi * r ** 3)
        return 1.0 * n / nbar - 1
     
    try:
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

    except StopIteration:
        return numpy.nan
main()

