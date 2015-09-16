from sys import argv
from sys import stdout
from sys import stderr
import logging

import nbodykit

from nbodykit.utils.pluginargparse import PluginArgumentParser
from nbodykit import plugins
from nbodykit.utils.mpilogging import MPILoggerAdapter

import numpy
import h5py

parser = PluginArgumentParser(None,
        loader=plugins.load,
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
parser.add_argument("datasource", type=plugins.DataSource.open,
        help=h + plugins.DataSource.format_help())
parser.add_argument("LinkingLength", type=float, 
        help='LinkingLength in mean separation (0.2)')
parser.add_argument("output", help='output file; output.grp.N and output.halo are written')
parser.add_argument("--nmin", type=float, default=32, help='minimum number of particles in a halo')

ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)
logger = MPILoggerAdapter(logging.getLogger("fof.py"))

from mpi4py import MPI

from nbodykit.distributedarray import DistributedArray
from nbodykit.ndarray import equiv_class, replacesorted
from nbodykit import halos

from kdcount import cluster
from pypm.domain import GridND

def assign_halo_label(data, comm, thresh):
    """ 
    Convert minid to sequential labels starting from 0.

    This routine is used to assign halo label to particles with
    the same minid.
    Halos with less than thresh particles are reclassified to 0.

    Parameters
    ----------
    minid : array_like, ('i8')
        The minimum particle id of the halo. All particles of a halo 
        have the same minid
    thresh : int
        halo with less than thresh particles are merged into halo 0
    comm : py:class:`MPI.Comm`
        communicator. since this is a collective operation

    Returns
    -------
    labels : array_like ('i8')
        The new labels of particles. Note that this is ordered
        by the size of halo, with the exception 0 represents all
        particles that are in halos that contain less than thresh particles.
    
    """
    data['origind'] = numpy.arange(len(data), dtype='i4')
    data['origind'] += sum(comm.allgather(len(data))[:comm.rank]) \
 
    data = DistributedArray(data, comm)


    # first attempt is to assign fofid for each group
    data.sort('fofid')
    label = data['fofid'].unique_labels()
    
    N = label.bincount()
    
    # now eliminate those with less than thresh particles
    small = N.local <= 32

    Nlocal = label.bincount(local=True)
    # mask == True for particles in small halos
    mask = numpy.repeat(small, Nlocal)
 
    # globally shift halo id by one
    label.local += 1
    label.local[mask] = 0

    data['fofid'].local[:] = label.local[:]
    del label

    data.sort('fofid')

    label = data['fofid'].unique_labels() 

    data['fofid'].local[:] = label.local[:]

    data.sort('origind')
    
    label = data['fofid'].local.view('i8')

    Nhalo0 = max(comm.allgather(label.max())) + 1

    Nlocal = numpy.bincount(label, minlength=Nhalo0)
    comm.Allreduce(MPI.IN_PLACE, Nlocal, op=MPI.SUM)

    # sort the labels by halo size
    arg = Nlocal[1:].argsort()[::-1] + 1
    P = numpy.arange(Nhalo0)
    P[arg] = numpy.arange(len(arg)) + 1
    label = P[label]
        
    return label

def local_fof(pos, ll):
    data = cluster.dataset(pos, boxsize=1.0)
    fof = cluster.fof(data, linking_length=ll, np=0, verbose=True)
    labels = fof.labels
    return labels

def main():
    comm = MPI.COMM_WORLD
    np = split_size_2d(comm.size)

    grid = [
        numpy.linspace(0, 1.0, np[0] + 1, endpoint=True),
        numpy.linspace(0, 1.0, np[1] + 1, endpoint=True),
    ]
    domain = GridND(grid)

    logger.info('grid %s' % str(grid), on=0 )

    # read in all !
    [[Position]] = ns.datasource.read(['Position'], comm, bunchsize=None)
    Position /= ns.datasource.BoxSize
    print Position.shape
    print Position.max(axis=0)
    Ntot = sum(comm.allgather(len(Position)))

    logger.info('Total number of particles %d, ll %g' % (Ntot, ns.LinkingLength), on=0)

    ll = ns.LinkingLength * Ntot ** -0.3333333

    #print pos
    #print ((pos[0] - pos[1]) ** 2).sum()** 0.5, ll
  
    layout = domain.decompose(Position, smoothing=ll * 1)

    Position = layout.exchange(Position)
 
    logger.info('domain %d has %d particles' % (comm.rank, len(Position)))

    labels = local_fof(Position, ll)
    del Position

    logger.info('local fof done', on=0)

    [[ID]] = ns.datasource.read(['ID'], comm, bunchsize=None)
    ID = layout.exchange(ID)
    # initialize global labels
    minid = equiv_class(labels, ID, op=numpy.fmin)[labels]
    del ID

    logger.info("equiv class, done", on=0)

    while True:
        # merge, if a particle belongs to several ranks
        # use the global label of the minimal
        minid_new = layout.gather(minid, mode=numpy.fmin)
        minid_new = layout.exchange(minid_new)

        # on my rank, these particles have been merged
        merged = minid_new != minid
        # if no rank has merged any, we are done
        # gl is the global label (albeit with some holes)
        total = comm.allreduce(merged.sum())
            
        logger.info('merged %d halos', total, on=0)

        if total == 0:
            del minid_new
            break
        old = minid[merged]
        new = minid_new[merged]
        arg = old.argsort()
        new = new[arg]
        old = old[arg]
        replacesorted(minid, old, new, out=minid)

    minid = layout.gather(minid, mode=numpy.fmin)
    del layout

    logger.info("merging, done", on=0)

    Nitem = len(minid)

    data = numpy.empty(Nitem, dtype=[
            ('origind', 'u8'), 
            ('fofid', 'u8'),
            ])
    # assign origind for recovery of ordering, since
    # we need to work in sorted fofid 
    data['fofid'] = minid
    del minid

    label = assign_halo_label(data, comm, thresh=ns.nmin)
    label = label.copy()
    del data
    N = halos.count(label, comm=comm)

    logger.info('Length of entries %s ', str(N), on=0)
    logger.info('Length of entries %s ', N.shape[0], on=0)
    logger.info('Total particles %s ', N.sum(), on=0)

    [[Position]] = ns.datasource.read(['Position'], comm, bunchsize=None)

    Position /= ns.datasource.BoxSize
    hpos = halos.centerofmass(label, Position, boxsize=1.0, comm=comm)
    del Position

    [[Velocity]] = ns.datasource.read(['Velocity'], comm, bunchsize=None)
    Velocity /= ns.datasource.BoxSize

    hvel = halos.centerofmass(label, Velocity, boxsize=None, comm=comm)
    del Velocity

    if comm.rank == 0:
        with h5py.File(ns.output + '.hdf5', 'w') as ff:
            N[0] = 0
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
                name='FOFGroups', data=data
                )
            dataset.attrs['Ntot'] = Ntot
            dataset.attrs['LinkLength'] = ns.LinkingLength
            dataset.attrs['BoxSize'] = ns.datasource.BoxSize

    del N
    del hpos

    Ntot = comm.allreduce(len(label))
    nfile = (Ntot + 512 ** 3 - 1) // (512 ** 3 )
    
    npart = [ 
        (i+1) * Ntot // nfile - i * Ntot // nfile \
            for i in range(nfile) ]

    if comm.rank == 0:
        for i in range(len(npart)):
            with open(ns.output + '.grp.%02d' % i, 'w') as ff:
                numpy.int32(npart[i]).tofile(ff)
                numpy.float32(ns.LinkingLength).tofile(ff)
                pass

    start = sum(comm.allgather(len(label))[:comm.rank])
    end = sum(comm.allgather(len(label))[:comm.rank+1])
    label = numpy.int32(label)
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
        with open(ns.output + '.grp.%02d' % i, 'r+') as ff:
            ff.seek(8, 0)
            ff.seek(mystart * 4, 1)
            label[written:written + myend - mystart].tofile(ff)
        written += myend - mystart

    return

def split_size_2d(s):
    """ Split `s` into two integers, 
        a and d, such that a * d == s and a <= d

        returns:  a, d
    """
    a = int(s** 0.5) + 1
    d = s
    while a > 1:
        if s % a == 0:
            d = s // a
            break
        a = a - 1 
    return a, d

main()
