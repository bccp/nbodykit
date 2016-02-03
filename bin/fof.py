from __future__ import print_function

from sys import argv
from sys import stdout
from sys import stderr
import logging

from nbodykit.plugins import ArgumentParser
from nbodykit.extensionpoints import DataSource
import numpy
import h5py

parser = ArgumentParser(None,
        description=
        """
        Find friend of friend groups from a Nbody simulation snapshot
        """,
        epilog=
        """
        This script is written by Yu Feng, as part of `nbodykit'. 
        """
        )

h = "Data source to read particle position:\n\n"
parser.add_argument("datasource", type=DataSource.fromstring,
        help=h + DataSource.format_help())
parser.add_argument("LinkingLength", type=float, 
        help='LinkingLength in mean separation (0.2)')
parser.add_argument("output", help='output file; output.grp.N and output.halo are written')
parser.add_argument("--nmin", type=float, default=32, help='minimum number of particles in a halo')

ns = parser.parse_args()
from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
name = MPI.Get_processor_name().split('.')[0]
logging.basicConfig(level=logging.DEBUG,
                    format='rank %d on %s: '%(rank,name) + \
                            '%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')


from nbodykit.fof import fof

def main():
    comm = MPI.COMM_WORLD

    catalogue, labels = fof(ns.datasource, ns.LinkingLength, ns.nmin, comm, return_labels=True)
    Ntot = comm.allreduce(len(labels))

    if comm.rank == 0:
        with h5py.File(ns.output + '.hdf5', 'w') as ff:
            # do not create dataset then fill because of
            # https://github.com/h5py/h5py/pull/606

            dataset = ff.create_dataset(
                name='FOFGroups', data=catalogue
                )
            dataset.attrs['Ntot'] = Ntot
            dataset.attrs['LinkLength'] = ns.LinkingLength
            dataset.attrs['BoxSize'] = ns.datasource.BoxSize

    nfile = (Ntot + 512 ** 3 - 1) // (512 ** 3 )
    
    npart = [ 
        (i+1) * Ntot // nfile - i * Ntot // nfile \
            for i in range(nfile) ]

    if comm.rank == 0:
        for i in range(len(npart)):
            with open(ns.output + '.grp.%02d' % i, 'wb') as ff:
                numpy.int32(npart[i]).tofile(ff)
                numpy.float32(ns.LinkingLength).tofile(ff)
                pass

    start = sum(comm.allgather(len(labels))[:comm.rank])
    end = sum(comm.allgather(len(labels))[:comm.rank+1])
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
        with open(ns.output + '.grp.%02d' % i, 'rb+') as ff:
            ff.seek(8, 0)
            ff.seek(mystart * 4, 1)
            labels[written:written + myend - mystart].tofile(ff)
        written += myend - mystart

    return

if __name__ == '__main__':
    main()
