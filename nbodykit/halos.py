import numpy
from mpi4py import MPI

from .ndarray import equiv_class

def centerofmass(label, pos, boxsize=1.0, comm=MPI.COMM_WORLD):
    """
    Calulate the center of mass of particles of the same label.

    The center of mass is defined as the mean of positions of particles,
    but care has to be taken regarding to the periodic boundary.

    This is a collective operation, and after the call, all ranks
    will have the position of halos.

    Parameters
    ----------
    label : array_like (integers)
        Halo label of particles, >=0
    pos   : array_like (float, 3)
        position of particles.
    boxsize : float or None
        size of the periodic box, or None if no periodic boundary is assumed.
    comm : :py:class:`MPI.Comm`
        communicator for the collective operation.
    
    Returns
    -------
    hpos : array_like (float, 3)
        the center of mass position of the halos.

    """
    Nhalo0 = max(comm.allgather(label.max())) + 1

    N = numpy.bincount(label, minlength=Nhalo0)
    comm.Allreduce(MPI.IN_PLACE, N, op=MPI.SUM)

    if boxsize is not None:
        posmin = equiv_class(label, pos, op=numpy.fmin, dense_labels=True, identity=numpy.inf,
                        minlength=len(N))
        comm.Allreduce(MPI.IN_PLACE, posmin, op=MPI.MIN)
        dpos = pos - posmin[label]
        bhalf = boxsize * 0.5
        dpos[dpos < -bhalf] += boxsize
        dpos[dpos >= bhalf] -= boxsize
    else:
        dpos = pos
    dpos = equiv_class(label, dpos, op=numpy.add, dense_labels=True, minlength=len(N))
    
    comm.Allreduce(MPI.IN_PLACE, dpos, op=MPI.SUM)
    dpos /= N[:, None]

    if boxsize:
        hpos = posmin + dpos
        hpos %= boxsize
    else:
        hpos = dpos
    return hpos
    
def count(label, comm=MPI.COMM_WORLD):
    """
    Count the number of particles of the same label.

    This is a collective operation, and after the call, all ranks
    will have the particle count.

    Parameters
    ----------
    label : array_like (integers)
        Halo label of particles, >=0
    comm : :py:class:`MPI.Comm`
        communicator for the collective operation.
    
    Returns
    -------
    count : array_like
        the count of number of particles in each halo

    """
    Nhalo0 = max(comm.allgather(label.max())) + 1

    N = numpy.bincount(label, minlength=Nhalo0)
    comm.Allreduce(MPI.IN_PLACE, N, op=MPI.SUM)

    return N
