from __future__ import print_function

from sys import argv
import logging

from nbodykit.plugins import ArgumentParser
from nbodykit.extensionpoints import DataSource, Painter
from pmesh.particlemesh import ParticleMesh
import numpy
import h5py

parser = ArgumentParser(None,
        description=
        """
        Create a subsample from a data source. The number density is calculated,
        but if a mass per particle is given, the density is calculated.
        """,
        epilog=
        """
        This script is written by Yu Feng, as part of `nbodykit'. 
        """
        )

h = "Data source to read particle position:\n\n"
parser.add_argument("datasource", type=DataSource.create,
        help=h + DataSource.format_help())
parser.add_argument("Nmesh", type=int,
        help='Size of FFT mesh for painting')
parser.add_argument("output", 
        help='output file; will be stored as a hdf5 file.')
parser.add_argument("--seed", type=int, default=12345,
        help='seed')
parser.add_argument("--ratio", type=float, default=0.01,
        help='fraction of particles to keep')
parser.add_argument("--smoothing", type=float, default=None,
        help='Smoothing Length in distance units. '
              'It has to be greater than the mesh resolution. '
              'Otherwise the code will die. Default is the mesh resolution.')


ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)
from mpi4py import MPI

import nbodykit
from pmesh import particlemesh

def main():
    comm = MPI.COMM_WORLD
    pm = ParticleMesh(ns.datasource.BoxSize, ns.Nmesh, dtype='f4', comm=None)
    if ns.smoothing is None:
        ns.smoothing = ns.datasource.BoxSize[0] / ns.Nmesh
    elif (ns.datasource.BoxSize / ns.Nmesh > ns.smoothing).any():
        raise ValueError("smoothing is too small")
 
    painter = Painter.create("DefaultPainter")
    painter.paint(pm, ns.datasource)
    pm.r2c()
    def Smoothing(pm, complex):
        k = pm.k
        k2 = 0
        for ki in k:
            ki2 = ki ** 2
            complex *= numpy.exp(-0.5 * ki2 * ns.smoothing ** 2)
        
    pm.transfer([Smoothing])
    pm.c2r()
    columns = ['Position', 'ID', 'Velocity']
    rng = numpy.random.RandomState(ns.Nmesh)
    seedtable = rng.randint(1024*1024*1024, size=comm.size)
    rngtable = [numpy.random.RandomState(seed) for seed in seedtable]

    dtype = numpy.dtype([
            ('Position', ('f4', 3)),
            ('Velocity', ('f4', 3)),
            ('ID', 'u8'),
            ('Density', 'f4'),
            ]) 

    subsample = []
    stat = {}
    for Position, ID, Velocity in ns.datasource.read(columns, comm, stat):
        u = rngtable[comm.rank].uniform(size=len(ID))
        keep = u < ns.ratio
        Nkeep = keep.sum()
        if Nkeep == 0: continue 
        data = numpy.empty(Nkeep, dtype=dtype)
        data['Position'][:] = Position[keep]       
        data['Velocity'][:] = Velocity[keep]       
        data['ID'][:] = ID[keep] 

        layout = pm.decompose(data['Position'])
        pos1 = layout.exchange(data['Position'])
        density1 = pm.readout(pos1)
        density = layout.gather(density1)

        data['Density'][:] = density
        subsample.append(data)

    subsample = numpy.concatenate(subsample)
    subsample = comm.gather(subsample)
    if comm.rank == 0:
        subsample = numpy.concatenate(subsample)
        subsample.sort(order='ID')
        with h5py.File(ns.output, 'w') as ff:
            dataset = ff.create_dataset(
                name='Subsample', data=subsample
                )
            dataset.attrs['Ratio'] = ns.ratio
            dataset.attrs['CommSize'] = comm.size
            dataset.attrs['Seed'] = ns.seed
            dataset.attrs['Smoothing'] = ns.smoothing
            dataset.attrs['Nmesh'] = ns.Nmesh
            dataset.attrs['Original'] = ns.datasource.string

if __name__ == '__main__':
    main()
