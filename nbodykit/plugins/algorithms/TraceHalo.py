import logging

import h5py

from nbodykit import files
from nbodykit import halos

import numpy
import mpsort
from nbodykit.extensionpoints import Algorithm, DataSource

class TraceHaloAlgorithm(Algorithm):
    plugin_name = "TraceHalo"
    
    def __init__(self, dest, source, sourcelabel):
        pass
    
    @classmethod
    def register(cls):
        s = cls.schema
        s.description = " Calculate the halo property based on a different set of halo labels."

        s.add_argument("dest", type=DataSource.from_config, help="type: DataSource")
        s.add_argument("source", type=DataSource.from_config, help="type: DataSource")
        s.add_argument("sourcelabel", type=DataSource.from_config,
            help='DataSource of the source halo label files, the Label column is used.')

    def run(self):
        comm = self.comm

        with self.source.open() as source:
            [[ID]] = source.read(['ID'], full=True)

        Ntot = self.comm.allreduce(len(ID))

        start = sum(comm.allgather(len(ID))[:comm.rank])
        end   = sum(comm.allgather(len(ID))[:comm.rank+1])
        data = numpy.empty(end - start, dtype=[
                    ('Label', ('i4')), 
                    ('ID', ('i8')), 
                    ])
        data['ID'] = ID
        del ID
        with self.sourcelabel.open() as sourcelabel:
            [[data['Label'][...]]] = sourcelabel.read(['Label'], full=True)

        mpsort.sort(data, orderby='ID')

        label = data['Label'].copy()
        del data

        data = numpy.empty(end - start, dtype=[
                    ('ID', ('i8')), 
                    ('Position', ('f4', 3)), 
                    ('Velocity', ('f4', 3)), 
                    ])
        with self.dest.open() as dest:
            [[data['Position'][...]]] = dest.read(['Position'], full=True)
            [[data['Velocity'][...]]] = dest.read(['Velocity'], full=True)
            [[data['ID'][...]]] = dest.read(['ID'], full=True)
        mpsort.sort(data, orderby='ID')

        data['Position'] /= self.dest.BoxSize
        data['Velocity'] /= self.dest.BoxSize
        
        N = halos.count(label)
        hpos = halos.centerofmass(label, data['Position'], boxsize=1.0)
        hvel = halos.centerofmass(label, data['Velocity'], boxsize=None)
        return hpos, hvel, N, Ntot

    def save(self, output, data): 
        hpos, hvel, N, Ntot = data
        if self.comm.rank == 0:
            logging.info("Total number of halos: %d" % len(N))
            logging.info("N %s" % str(N))
            N[0] = 0
            with h5py.File(output, 'w') as ff:
                data = numpy.empty(shape=(len(N),), 
                    dtype=[
                    ('Position', ('f4', 3)),
                    ('Velocity', ('f4', 3)),
                    ('Length', 'i4')])
                
                data['Position'] = hpos
                data['Velocity'] = hvel
                data['Length'] = N
                
                # do not create dataset then fill because of
                # https://github.com/h5py/h5py/pull/606

                dataset = ff.create_dataset(
                    name='TracedFOFGroups', data=data
                    )
                dataset.attrs['Ntot'] = Ntot
                dataset.attrs['BoxSize'] = self.source.BoxSize
                dataset.attrs['source'] = self.source.string
                dataset.attrs['sourcelabel'] = self.sourcelabel.string
                dataset.attrs['dest'] = self.dest.string

            logging.info("Written %s" % output)

