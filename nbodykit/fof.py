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

def fof_halo_label(minid, comm, thresh):
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
    dtype = numpy.dtype([
            ('origind', 'u8'), 
            ('fofid', 'u8'),
            ])
    data = numpy.empty(len(minid), dtype=dtype)
    # assign origind for recovery of ordering, since
    # we need to work in sorted fofid 
    data['fofid'] = minid
    data['origind'] = numpy.arange(len(data), dtype='u4')
    data['origind'] += sum(comm.allgather(len(data))[:comm.rank]) \
 
    data = DistributedArray(data, comm)

    # first attempt is to assign fofid for each group
    data.sort('fofid')
    label = data['fofid'].unique_labels()
    
    N = label.bincount()
    
    # now eliminate those with less than thresh particles
    small = N.local <= thresh

    Nlocal = label.bincount(local=True)
    # mask == True for particles in small halos
    mask = numpy.repeat(small, Nlocal)
 
    # globally shift halo id by one
    label.local += 1
    label.local[mask] = 0

    data['fofid'].local[:] = label.local[:]
    del label

    data.sort('fofid')

    data['fofid'].local[:] = data['fofid'].unique_labels().local[:]

    data.sort('origind')
    
    label = data['fofid'].local.view('i8').copy()
    
    del data

    Nhalo0 = max(comm.allgather(label.max())) + 1
    Nlocal = numpy.bincount(label, minlength=Nhalo0)
    comm.Allreduce(MPI.IN_PLACE, Nlocal, op=MPI.SUM)

    # sort the labels by halo size
    arg = Nlocal[1:].argsort()[::-1] + 1
    P = numpy.arange(Nhalo0, dtype='i4')
    P[arg] = numpy.arange(len(arg), dtype='i4') + 1
    label = P[label]
        
    return label

def local_fof(layout, pos, boxsize, ll, comm):
    N = len(pos)

    pos = layout.exchange(pos)
    data = cluster.dataset(pos, boxsize=boxsize)
    fof = cluster.fof(data, linking_length=ll, np=0)
    labels = fof.labels
    del fof

    PID = numpy.arange(N, dtype='intp')
    PID += sum(comm.allgather(N)[:comm.rank])

    PID = layout.exchange(PID)
    # initialize global labels
    minid = equiv_class(labels, PID, op=numpy.fmin)[labels]

    return minid

def fof_merge(layout, minid, comm):
    # generate global halo id

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

    minid = layout.gather(minid, mode=numpy.fmin)
    return minid

def fof(datasource, linking_length, nmin, comm=MPI.COMM_WORLD, log_level=logging.DEBUG):
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
            datasource; must support Position.
            datasource.BoxSize is used too.
        linking_length: float
            linking length in data units. (Usually Mpc/h).
        nmin: int
            Minimal length (number of particles) of a halo. Features
            with less than nmin particles are considered noise, and
            removed from the catalogue

        comm: MPI.Comm
            The mpi communicator.

        Returns
        -------
        label: array_like
            The halo label of each position. A label of 0 standands for not in any halo.
 
    """
    if log_level is not None: logger.setLevel(log_level)

    np = split_size_3d(comm.size)

    grid = [
        numpy.linspace(0, datasource.BoxSize[0], np[0] + 1, endpoint=True),
        numpy.linspace(0, datasource.BoxSize[1], np[1] + 1, endpoint=True),
        numpy.linspace(0, datasource.BoxSize[2], np[2] + 1, endpoint=True),
    ]
    domain = GridND(grid)

    with datasource.open() as stream:
        [[Position]] = stream.read(['Position'], full=True)

    if comm.rank == 0: logger.info("ll %g MPC/h " % linking_length)
    if comm.rank == 0: logger.debug('grid: %s' % str(grid))

    layout = domain.decompose(Position, smoothing=linking_length * 1)

    comm.barrier()
    if comm.rank == 0: logger.info("Starting local fof.")

    minid = local_fof(layout, Position, datasource.BoxSize, linking_length, comm)
    
    comm.barrier()
    if comm.rank == 0: logger.info("Finished local fof.")

    if comm.rank == 0: logger.info("Merged global FOF.")

    minid = fof_merge(layout, minid, comm)
    del layout
    # sort calculate halo catalogue
    label = fof_halo_label(minid, comm, thresh=nmin)

    return label

def fof_catalogue(datasource, label, comm, calculate_initial=False):
    """ Catalogue of FOF groups based on label from a data source

        Friend-of-friend was first used by Davis et al 1985 to define
        halos in hierachical structure formation of cosmological simulations.
        The algorithm is also known as DBSCAN in computer science. 
        The subroutine here implements a parallel version of the FOF. 

        The underlying local FOF algorithm is from `kdcount.cluster`, 
        which is an adaptation of the implementation in Volker Springel's 
        Gadget and Martin White's PM. It could have been done faster.

        Parameters
        ----------
        label : array_like
            halo label of particles from data source. 

        datasource: DataSource
            datasource; must support Position and Velocity.
            datasource.BoxSize is used too.

        comm: MPI.Comm
            The mpi communicator. Must agree with the datasource

        Returns
        -------
        catalogue: array_like
            A 1-d array of type 'Position', 'Velocity', 'Length'. 
            The center mass position and velocity of the FOF halo, and
            Length is the number of particles in a halo. The catalogue is
            sorted such that the most massive halo is first. catalogue[0]
            does not correspond to any halo.
 
    """
    dtype=[('Position', ('f4', 3)),
        ('Velocity', ('f4', 3)),
        ('Length', 'i4')]

    N = halos.count(label, comm=comm)
    
    # explicitly open the DataSource
    with datasource.keep_cache():
    
        with datasource.open() as stream:
            [[Position]] = stream.read(['Position'], full=True)
        Position /= datasource.BoxSize
        hpos = halos.centerofmass(label, Position, boxsize=1.0, comm=comm)
        del Position

        with datasource.open() as stream: 
            [[Velocity]] = stream.read(['Velocity'], full=True)
        Velocity /= datasource.BoxSize

        hvel = halos.centerofmass(label, Velocity, boxsize=None, comm=comm)
        del Velocity

        if calculate_initial:

            dtype.append(('InitialPosition', ('f4', 3)))
        
            with datasource.open() as stream:
                [[Position]] = stream.read(['InitialPosition'], full=True)
            Position /= datasource.BoxSize
            hpos_init = halos.centerofmass(label, Position, boxsize=1.0, comm=comm)
            del Position

    if comm.rank == 0: logger.info("Calculated catalogue %d halos found. " % (len(N) -1 ))
    if comm.rank == 0: logger.info("Length = %s " % N[1:])
    if comm.rank == 0: logger.info("%d particles not in halo" % N[0])

    dtype = numpy.dtype(dtype)
    if comm.rank == 0:
        catalogue = numpy.empty(shape=len(N), dtype=dtype)

        catalogue['Position'] = hpos
        catalogue['Velocity'] = hvel
        catalogue['Length'] = N
        catalogue['Length'][0] = 0
        if calculate_initial:
            catalogue['InitialPosition'] = hpos_init
    else:
        catalogue = numpy.empty(shape=0, dtype=dtype)
        
    return catalogue

def split_size_3d(s):
    """ Split `s` into two integers, 
        a and d, such that a * d == s and a <= d

        returns:  a, d
    """
    a = int(s** 0.33333) + 1
    d = s
    while a > 1:
        if s % a == 0:
            s = s // a
            break
        a = a - 1 
    b = int(s**0.5) + 1
    while b > 1:
        if s % b == 0:
            s = s // b
            break
        b = b - 1
    return a, b, s
