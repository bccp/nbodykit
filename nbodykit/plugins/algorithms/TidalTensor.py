from nbodykit.extensionpoints import Algorithm
import logging
import numpy
from itertools import product

# for output
import h5py
import bigfile
import mpsort

from mpi4py import MPI

import nbodykit
from pmesh.particlemesh import ParticleMesh
from nbodykit.extensionpoints import DataSource, painters


class TidalTensor(Algorithm):
    plugin_name = "TidalTensor"
    
    def __init__(self, field, points, Nmesh, smoothing=None):
        pass

    @classmethod
    def register(cls):
        s = cls.schema
        s.description = "compute the tidal force tensor"
        
        s.add_argument("field", type=DataSource.from_config,
                help="DataSource; run `nbkit.py --list-datasources` for all options")
        s.add_argument("points", type=DataSource.from_config,
                help="A small set of points to calculate tidal force on; "
                     "run `nbkit.py --list-datasources` for all options")
        s.add_argument("Nmesh", type=int,
                help='Size of FFT mesh for painting')
        s.add_argument("smoothing", type=float,
                help='Smoothing Length in distance units. '
                      'It has to be greater than the mesh resolution. '
                      'Otherwise the code will die. Default is the mesh resolution.')
                      
    def Smoothing(self, pm, complex):
        k = pm.k
        k2 = 0
        for ki in k:
            ki2 = ki ** 2
            complex *= numpy.exp(-0.5 * ki2 * self.smoothing ** 2)

    def NormalizeDC(self, pm, complex):
        """ removes the DC amplitude. This effectively
            divides by the mean
        """
        w = pm.w
        comm = pm.comm
        ind = []
        value = 0.0
        found = True
        for wi in w:
            if (wi != 0).all():
                found = False
                break
            ind.append((wi == 0).nonzero()[0][0])
        if found:
            ind = tuple(ind)
            value = numpy.abs(complex[ind])
        value = comm.allreduce(value, MPI.SUM)
        complex[:] /= value

    def TidalTensor(self, u, v):
        # k_u k_v / k **2
        def TidalTensor(pm, complex):
            k = pm.k

            for row in range(complex.shape[0]):
                k2 = k[0][row] ** 2
                for ki in k[1:]:
                    k2 = k2 + ki[0] ** 2
                k2[k2 == 0] = numpy.inf
                complex[row] /= k2

            complex *= k[u]
            complex *= k[v]

        return TidalTensor

    def run(self):
        pm = ParticleMesh(self.field.BoxSize, self.Nmesh, dtype='f4', comm=self.comm)
        if self.smoothing is None:
            self.smoothing = self.field.BoxSize[0] / self.Nmesh
        elif (self.field.BoxSize / self.Nmesh > self.smoothing).any():
            raise ValueError("smoothing is too small")
     
        painter = painters.DefaultPainter(weight="Mass")
        painter.paint(pm, self.field)
        pm.r2c()

        pm.transfer([self.Smoothing, self.NormalizeDC])

        with self.points.open() as stream:
            [[Position ]] = stream.read(['Position'], full=True)

        layout = pm.decompose(Position)
        pos1 = layout.exchange(Position)
        value = numpy.empty((3, 3, len(Position)))

        for u, v in product(range(3), range(3)):
            if self.comm.rank == 0:
                logging.info("Working on tensor element (%d, %d)" % (u, v))
            pm.push()
            pm.transfer([self.TidalTensor(u, v)])
            pm.c2r()
            v1 = pm.readout(pos1)
            v1 = layout.gather(v1)
            pm.pop()

            value[u, v] = v1
             
        return value.transpose((2, 0, 1))

    def save(self, output, data):
        self.write_hdf5(data, output)

    def write_hdf5(self, data, output):

        size = self.comm.allreduce(len(data))
        offset = sum(self.comm.allgather(len(data))[:self.comm.rank])

        if self.comm.rank == 0:
            with h5py.File(output, 'w') as ff:
                dataset = ff.create_dataset(name='TidalTensor',
                        dtype=data.dtype, shape=(size, 3, 3))
                dataset.attrs['Smoothing'] = self.smoothing
                dataset.attrs['Nmesh'] = self.Nmesh
                dataset.attrs['Original'] = self.field.string
                dataset.attrs['BoxSize'] = self.field.BoxSize

        for i in range(self.comm.size):
            self.comm.barrier()
            if i != self.comm.rank: continue
                 
            with h5py.File(output, 'r+') as ff:
                dataset = ff['TidalTensor']
                dataset[offset:len(data) + offset] = data

