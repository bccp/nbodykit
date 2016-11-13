from nbodykit.core import Algorithm, DataSource
import numpy

class TraceHaloAlgorithm(Algorithm):
    plugin_name = "TraceHalo"
    
    def __init__(self, dest, source, sourcelabel):
        
        self.dest        = dest
        self.source      = source
        self.sourcelabel = sourcelabel
    
    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = " calculate the halo property based on a different set of halo labels."

        s.add_argument("dest", type=DataSource.from_config, help="type: DataSource")
        s.add_argument("source", type=DataSource.from_config, help="type: DataSource")
        s.add_argument("sourcelabel", type=DataSource.from_config,
            help='DataSource of the source halo label files, the Label column is used.')

    def run(self):
        """
        Run the TraceHalo Algorithm
        """
        import mpsort
        from nbodykit import halos
        
        comm = self.comm

        with self.source.open() as source:
            [[ID]] = source.read(['ID'], full=True)

        Ntot = self.comm.allreduce(len(ID))

        with self.sourcelabel.open() as sourcelabel:
            [[label]] = sourcelabel.read(['Label'], full=True)

        mpsort.sort(label, orderby=ID, comm=self.comm)
        del ID

        dtype = numpy.dtype([
                    ('ID', ('i8')), 
                    ('Position', ('f4', 3)), 
                    ('Velocity', ('f4', 3)), 
                    ])
        data = numpy.empty(len(label), dtype=dtype)
        with self.dest.open() as dest:
            [[data['Position'][...]]] = dest.read(['Position'], full=True)
            [[data['Velocity'][...]]] = dest.read(['Velocity'], full=True)
            [[data['ID'][...]]] = dest.read(['ID'], full=True)
        mpsort.sort(data, orderby='ID', comm=self.comm)

        data['Position'] /= self.dest.BoxSize
        data['Velocity'] /= self.dest.BoxSize
        
        N = halos.count(label)
        hpos = halos.centerofmass(label, data['Position'], boxsize=1.0)
        hvel = halos.centerofmass(label, data['Velocity'], boxsize=None)
        return hpos, hvel, N, Ntot

    def save(self, output, data): 
        """
        Save the result
        """
        import h5py
        
        hpos, hvel, N, Ntot = data
        if self.comm.rank == 0:
            self.logger.info("Total number of halos: %d" % len(N))
            self.logger.info("N %s" % str(N))
            N[0] = 0
            with h5py.File(output, 'w') as ff:
                
                dtype = numpy.dtype([
                    ('Position', ('f4', 3)),
                    ('Velocity', ('f4', 3)),
                    ('Length', 'i4')])
                data = numpy.empty(shape=(len(N),), dtype=dtype)
                data['Position'] = hpos
                data['Velocity'] = hvel
                data['Length'] = N
                
                # do not create dataset then fill because of
                # https://github.com/h5py/h5py/pull/606

                dataset = ff.create_dataset(
                    name='TracedFOFGroups', data=data
                    )
                dataset.attrs['Ntot']        = Ntot
                dataset.attrs['BoxSize']     = self.source.BoxSize
                dataset.attrs['source']      = self.source.string
                dataset.attrs['sourcelabel'] = self.sourcelabel.string
                dataset.attrs['dest']        = self.dest.string

            self.logger.info("Written %s" % output)

