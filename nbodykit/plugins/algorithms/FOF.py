from nbodykit.extensionpoints import Algorithm
import logging
import numpy

# for output
import h5py
import bigfile

class FOFAlgorithm(Algorithm):
    plugin_name = "FOF"
    
    def __init__(self, datasource, linklength, without_labels=False, nmin=32):
        pass
    
    @classmethod
    def register(cls):
        s = cls.schema
        s.description = "Friend of Friend halo finder"
        
        s.add_argument("datasource", 
            help='`DataSource` objects to run FOF against; run --list-datasource for specifics')
        s.add_argument("linklength", type=float, help='the link length in terms of mean particle sep')
        s.add_argument("without_labels", type=bool, help='do not store labels')
        s.add_argument("nmin", type=int, help='minimum number of particles in a halo')
        
    def run(self):
        from nbodykit import fof
        catalog, labels = fof.fof(self.datasource, self.linklength, self.nmin, self.comm, return_labels=True)
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


