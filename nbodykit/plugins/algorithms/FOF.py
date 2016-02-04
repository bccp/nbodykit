from nbodykit.extensionpoints import Algorithm
import logging
import numpy

# for output
import h5py

class FOFAlgorithm(Algorithm):

    plugin_name = "FOF"
    
    @classmethod
    def register(kls):
        from nbodykit.extensionpoints import DataSource

        p = kls.parser
        p.description = "Friend of Friend halo finder"
        p.add_argument("datasource", type=DataSource.fromstring, 
                        help='`DataSource` objects to run FOF against; run --list-datasource for specifics')
        p.add_argument("linklength", type=float, metavar='0.02', help='Link length')
        p.add_argument("--without-labels", action='store_true', help='Do not store labels')
        p.add_argument("--nmin", type=int, default=32, help='minimum number of particles in a halo')
        
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
            with h5py.File(output + '.hdf5', 'w') as ff:
                # do not create dataset then fill because of
                # https://github.com/h5py/h5py/pull/606

                dataset = ff.create_dataset(
                    name='FOFGroups', data=catalog
                    )
                dataset.attrs['Ntot'] = Ntot
                dataset.attrs['LinkLength'] = self.linklength
                dataset.attrs['BoxSize'] = self.datasource.BoxSize

        if not self.without_labels:
            # todo: switch to bigfile. This is ugly.

            nfile = (Ntot + 512 ** 3 - 1) // (512 ** 3 )
            
            npart = [ 
                (i+1) * Ntot // nfile - i * Ntot // nfile \
                    for i in range(nfile) ]

            if self.comm.rank == 0:
                for i in range(len(npart)):
                    with open(output + '.grp.%02d' % i, 'wb') as ff:
                        numpy.int32(npart[i]).tofile(ff)
                        numpy.float32(self.linklength).tofile(ff)
                        pass

            start = sum(self.comm.allgather(len(labels))[:self.comm.rank])
            end = sum(self.comm.allgather(len(labels))[:self.comm.rank+1])
            labels = numpy.int32(labels)
            written = 0
            for i in range(len(npart)):
                filestart = sum(npart[:i])
                fileend = sum(npart[:i+1])
                mystart = start - filestart
                myend = end - filestart
                if myend <= 0 : continue
                if mystart >= npart[i] : continue
                if myend > npart[i]: myend = npart[i]
                if mystart < 0: mystart = 0
                with open(output + '.grp.%02d' % i, 'rb+') as ff:
                    ff.seek(8, 0)
                    ff.seek(mystart * 4, 1)
                    labels[written:written + myend - mystart].tofile(ff)
                written += myend - mystart

        return


