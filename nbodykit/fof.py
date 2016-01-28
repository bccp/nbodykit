from __future__ import print_function

from nbodykit.distributedarray import DistributedArray
from nbodykit.ndarray import equiv_class, replacesorted
from nbodykit import halos

from kdcount import cluster
from pmesh.domain import GridND
import numpy
from mpi4py import MPI

import logging

logger = logging.getLogger('measurestats')

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
    fof = cluster.fof(data, linking_length=ll, np=0)
    labels = fof.labels
    return labels

def fof(datasource, linking_length, nmin, comm=MPI.COMM_WORLD, return_labels=False, log_level=logging.DEBUG):
    """ Run Friend-of-friend halo finder.

        Friend-of-friend was first used by Davis et al 1985 to define
        halos in hierachical structure formation of cosmological simulations.
        The algorithm is also known as DBSCAN in computer science. 
        The subroutine here implements a parallel version of the FOF. 

        The underlying local FOF algorithm is from `kdcount.cluster`, 
        which is an adaptation of the implementation in Volker Springel's 
        Gadget and Martin White's PM. It could have been done faster.

        Parameters
        ----------
        datasource: DataSource
            datasource; must support Position and Velocity.
            datasource.BoxSize is used too.
        linking_length: float
            linking length in terms of mean particle seperation.
            Typical values are 0.2 or 0.168.
        nmin: int
            Minimal length (number of particles) of a halo. Features
            with less than nmin particles are considered noise, and
            removed from the catalogue
        return_labels: boolean
            If true, return the labels. The labels can be a large chunk
            of memory. Default: False

        comm: MPI.Comm
            The mpi communicator.

        Returns
        -------
        catalogue: array_like
            A 1-d array of type 'Position', 'Velocity', 'Length'. 
            The center mass position and velocity of the FOF halo, and
            Length is the number of particles in a halo. The catalogue is
            sorted such that the most massive halo is first.

        label: array_like
            The halo label of each particle, as the sequence they were
            in the data source.
 
    """
    if log_level is not None: logger.setLevel(log_level)

    np = split_size_2d(comm.size)

    grid = [
        numpy.linspace(0, 1.0, np[0] + 1, endpoint=True),
        numpy.linspace(0, 1.0, np[1] + 1, endpoint=True),
    ]
    domain = GridND(grid)

    if comm.rank == 0: logger.debug('grid: %s' % str(grid))

    # read in all !
    stats = {}
    [[Position]] = datasource.read(['Position'], comm, stats, full=True)
    Position /= datasource.BoxSize

    Ntot = stats['Ntot']

    if comm.rank == 0: logger.debug('Ntot: %d' % Ntot)

    ll = linking_length * Ntot ** -0.3333333

    Nread = len(Position)

    layout = domain.decompose(Position, smoothing=ll * 1)
    Position = layout.exchange(Position)

    comm.barrier()
    if comm.rank == 0: logger.info("Starting local fof.")

    labels = local_fof(Position, ll)
    del Position

    comm.barrier()
    if comm.rank == 0: logger.info("Finished local fof.")
    # local done. now do the global stuff.

    ID = numpy.arange(Nread, dtype='intp')
    ID += sum(comm.allgather(Nread)[:comm.rank])

    ID = layout.exchange(ID)
    # initialize global labels
    minid = equiv_class(labels, ID, op=numpy.fmin)[labels]
    del ID

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
            
        if total == 0:
            del minid_new
            break
        old = minid[merged]
        new = minid_new[merged]
        arg = old.argsort()
        new = new[arg]
        old = old[arg]
        replacesorted(minid, old, new, out=minid)

    if comm.rank == 0: logger.info("Merged global FOF.")

    minid = layout.gather(minid, mode=numpy.fmin)
    del layout

    # calculate halo catalogue

    Nitem = len(minid)

    data = numpy.empty(Nitem, dtype=[
            ('origind', 'u8'), 
            ('fofid', 'u8'),
            ])
    # assign origind for recovery of ordering, since
    # we need to work in sorted fofid 
    data['fofid'] = minid
    del minid

    label = assign_halo_label(data, comm, thresh=nmin)
    label = label.copy()
    del data
    N = halos.count(label, comm=comm)

    [[Position]] = datasource.read(['Position'], comm, stats, full=True)

    Position /= datasource.BoxSize
    hpos = halos.centerofmass(label, Position, boxsize=1.0, comm=comm)
    del Position

    [[Velocity]] = datasource.read(['Velocity'], comm, stats, full=True)
    Velocity /= datasource.BoxSize

    hvel = halos.centerofmass(label, Velocity, boxsize=None, comm=comm)
    del Velocity

    if comm.rank == 0: logger.info("Calculated catalogue %d halos found. " % (len(N) -1 ))
    if comm.rank == 0: logger.info("Length = %s " % N[1:])
    if comm.rank == 0: logger.info("%d particles not in halo" % N[0])

    dtype=[
        ('Position', ('f4', 3)),
        ('Velocity', ('f4', 3)),
        ('Length', 'i4')]

    if comm.rank == 0:
        catalogue = numpy.empty(shape=len(N), dtype=dtype)
    
        catalogue['Position'] = hpos
        catalogue['Velocity'] = hvel
        catalogue['Length'] = N
    else:
        catalogue = numpy.empty(shape=0, dtype=dtype)
        
    if return_labels:
        label = numpy.int32(label) - 1
        return catalogue[1:], label
    else:
        return catalogue[1:]

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
