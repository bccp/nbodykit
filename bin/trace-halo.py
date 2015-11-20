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
     """This script trace particles in halo at datasource_tf by ID
        find their positions datasource_ti, then compute FOF group
        properties at datasource_ti and save it to a file.
     """,
        epilog=
     """
        This script is written by Yu Feng, as part of `nbodykit'. 
     """
        )

parser.add_argument("datasource_ti", type=plugins.DataSource.open,
        help=plugins.DataSource.format_help())
parser.add_argument("datasource_tf", type=plugins.DataSource.open,
        help=plugins.DataSource.format_help())
parser.add_argument("halolabel", 
        help='basename of the halo label files, only nbodykit format is supported in this script')
parser.add_argument("output", help='write output to this file (hdf5 is appended)')

ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)

import numpy
import nbodykit
from nbodykit import files
from nbodykit import halos

import mpsort
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    IC, SNAP, LABEL = None, None, None
    if comm.rank == 0:
        LABEL = files.Snapshot(ns.halolabel, files.HaloLabelFile)

    LABEL = comm.bcast(LABEL)
 
    Ntot = sum(LABEL.npart)

    [[ID]] = ns.datasource_tf.read(['ID'], comm, bunchsize=None)

    start = sum(comm.allgather(len(ID))[:comm.rank])
    end   = sum(comm.allgather(len(ID))[:comm.rank+1])
    data = numpy.empty(end - start, dtype=[
                ('Label', ('i4')), 
                ('ID', ('i8')), 
                ])
    data['ID'] = ID
    del ID
    data['Label'] = LABEL.read("Label", start, end)

    mpsort.sort(data, orderby='ID')

    label = data['Label'].copy()

    data = numpy.empty(end - start, dtype=[
                ('ID', ('i8')), 
                ('Position', ('f4', 3)), 
                ])
    [[data['Position'][...]]] = ns.datasource_ti.read(['Position'], comm, bunchsize=None)
    [[data['ID'][...]]] = ns.datasource_ti.read(['ID'], comm, bunchsize=None)
    mpsort.sort(data, orderby='ID')

    pos = data['Position'] / ns.datasource_ti.BoxSize
    del data
    
    N = halos.count(label)
    hpos = halos.centerofmass(label, pos, boxsize=1.0)
    
    if comm.rank == 0:
        logging.info("Total number of halos: %d" % len(N))
        logging.info("N %s" % str(N))
        LinkingLength = LABEL.get_file(0).linking_length

        with h5py.File(ns.output + '.hdf5', 'w') as ff:
            N[0] = 0
            data = numpy.empty(shape=(len(N),), 
                dtype=[
                ('Position', ('f4', 3)),
                ('Velocity', ('f4', 3)),
                ('Length', 'i4')])
            
            data['Position'] = hpos
            data['Velocity'] = 0
            data['Length'] = N
            
            # do not create dataset then fill because of
            # https://github.com/h5py/h5py/pull/606

            dataset = ff.create_dataset(
                name='TracedFOFGroups', data=data
                )
            dataset.attrs['Ntot'] = Ntot
            dataset.attrs['BoxSize'] = ns.datasource_ti.BoxSize
            dataset.attrs['ti'] = ns.datasource_ti.string
            dataset.attrs['tf'] = ns.datasource_tf.string

        logging.info("Written %s" % ns.output + '.hdf5')


main()
