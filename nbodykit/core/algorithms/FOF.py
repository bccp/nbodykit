from nbodykit.core import Algorithm, DataSource
import numpy

# for output
import h5py
import bigfile

class FOFAlgorithm(Algorithm):
    plugin_name = "FOF"
    
    def __init__(self, datasource, linklength, 
                    absolute=False, without_labels=False, 
                    nmin=32, calculate_initial_position=False):
        
        # set the input parameters
        self.datasource                 = datasource
        self.linklength                 = linklength
        self.absolute                   = absolute
        self.without_labels             = without_labels
        self.nmin                       = nmin
        self.calculate_initial_position = calculate_initial_position
    
    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "a Friends-of-Friends (FOF) halo finder"
        
        s.add_argument("datasource", type=DataSource.from_config,
            help='`DataSource` objects to run FOF against; '
                 'run `nbkit.py --list-datasources` for all options')
        s.add_argument("linklength", type=float, 
            help='the linking length in either absolute or relative units')
        s.add_argument("absolute", type=bool,
            help='If set, the linking length is in absolute units, '
                'otherwise it is relative to the mean particle separation; default is `False`')
        s.add_argument("calculate_initial_position", type=bool,
            help='If set, calculate the initial position of halos based on the '
                'InitialPosition field of DataSource')
        s.add_argument("without_labels", type=bool, help='do not store labels')
        s.add_argument("nmin", type=int, help='minimum number of particles in a halo')
        
    def run(self):
        from nbodykit import fof
        # convert to absolute length
        if not self.absolute:
            if not hasattr(self.datasource, 'size'):
                logging.info("Playing DataSource to measure total size. " +
                      "DataSource `%s' shall be fixed to add a size attribute", self.datasource)
                
                with self.datasource.open() as stream:
                    [[junk]] = stream.read(['Position'], full=True)
                    size = self.comm.allreduce(len(junk))
                    del junk
            else:
                size = self.datasource.size
            ll = self.linklength * (self.datasource.BoxSize.prod() / self.datasource.size) ** 0.3333333
        else:
            ll = self.linklength

        labels = fof.fof(self.datasource, ll, self.nmin, self.comm)
        catalog = fof.fof_catalogue(self.datasource, labels, self.comm, self.calculate_initial_position)

        Ntot = self.comm.allreduce(len(labels))
        if self.without_labels:
            return catalog, Ntot
        else:
            return catalog, labels, Ntot

    def save(self, output, data):
        if self.without_labels:
            catalog, Ntot = data
        else:
            catalog, labels, Ntot = data

        if self.comm.rank == 0:
            with h5py.File(output, 'w') as ff:
                # do not create dataset then fill because of
                # https://github.com/h5py/h5py/pull/606

                dataset = ff.create_dataset(
                    name='FOFGroups', data=catalog
                    )
                dataset.attrs['Ntot'] = Ntot
                dataset.attrs['LinkLength'] = self.linklength
                dataset.attrs['BoxSize'] = self.datasource.BoxSize

        if not self.without_labels:
            output = output.replace('.hdf5', '.labels')
            bf = bigfile.BigFileMPI(self.comm, output, create=True)
            with bf.create_from_array("Label", labels, Nfile=(self.comm.size + 7)// 8) as bb:
                bb.attrs['LinkLength'] = self.linklength
                bb.attrs['Ntot'] = Ntot
                bb.attrs['BoxSize'] = self.datasource.BoxSize
        return


