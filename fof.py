from sys import argv
from sys import stdout
from sys import stderr
import logging

from argparse import ArgumentParser
import numpy
from mpi4py import MPI

import nbodykit

from nbodykit.distributedarray import DistributedArray
from nbodykit.tpm import TPMSnapshotFile, read

from kdcount import cluster
from pypm.domain import GridND


parser = ArgumentParser("",
        description=
     "",
        epilog=
     ""
        )

parser.add_argument("filename", 
        help='basename of the input, only runpb format is supported in this script')
parser.add_argument("BoxSize", type=float, 
        help='BoxSize in Mpc/h')
parser.add_argument("LinkingLength", type=float, 
        help='LinkingLength in mean separation (0.2)')
parser.add_argument("output", help='output file')

ns = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)
def equiv_class(labels, values, op, dense_labels=False, identity=None, minlength=None):
    """
    apply operation to equivalent classes by label, on values

    Parameters
    ----------
    labels  : array_like
        the label of objects, starting from 0.
    values  : array_like
        the values of objects (len(labels), ...)
    op      : :py:class:`numpy.ufunc`
        the operation to apply

    Returns
    -------
        the value of each equivalent class

    Examples
    --------
    >>> x = numpy.arange(10)
    >>> print equiv_class(x, x, numpy.fmin, dense_labels=True)
    [0 1 2 3 4 5 6 7 8 9]

    >>> x = numpy.arange(10)
    >>> v = numpy.arange(20).reshape(10, 2)
    >>> x[1] = 0
    >>> print equiv_class(x, 1.0 * v, numpy.fmin, dense_labels=True, identity=numpy.inf)
    [[  0.   1.]
     [ inf  inf]
     [  4.   5.]
     [  6.   7.]
     [  8.   9.]
     [ 10.  11.]
     [ 12.  13.]
     [ 14.  15.]
     [ 16.  17.]
     [ 18.  19.]]

    """
    # dense labels
    if not dense_labels:
        junk, labels = numpy.unique(labels, return_inverse=True)
        del junk
    N = numpy.bincount(labels)
    offsets = numpy.concatenate([[0], N.cumsum()], axis=0)[:-1]
    arg = labels.argsort()
    if identity is None: identity = op.identity
    if minlength is None:
        minlength = len(N)

    result = numpy.empty(minlength, dtype=(values.dtype, values.shape[1:]))
    result[:len(N)] = op.reduceat(values[arg], offsets)

    if (N == 0).any():
        result[N == 0] = identity

    if minlength is not None and len(N) < minlength:
        result[len(N):] = identity

    return result

def replacesorted(arr, sorted, b, out=None):
    """
    replace a with corresponding b in arr

    Parameters
    ----------
    arr : array_like
        input array
    sorted   : array_like 
        sorted

    b   : array_like

    out : array_like,
        output array
    Result
    ------
    newarr  : array_like
        arr with a replaced by corresponding b

    Examples
    --------
    >>> print replacesorted(numpy.arange(10), numpy.arange(5), numpy.ones(5))
    [1 1 1 1 1 5 6 7 8 9]

    """
    if out is None:
        out = arr.copy()
    if len(sorted) == 0:
        return out
    ind = sorted.searchsorted(arr)
    ind.clip(0, len(sorted) - 1, out=ind)
    arr = numpy.array(arr)
    found = sorted[ind] == arr
    out[found] = b[ind[found]]
    return out

def densify(minid, comm, thresh):
    Nitem = len(minid)

    data = numpy.empty(Nitem, dtype=[
            ('origind', 'u8'), 
            ('minid', 'u8'),
            ])
    data['origind'] = sum(comm.allgather(Nitem)[:comm.rank]) \
            + numpy.arange(Nitem)
    data['minid'] = minid
    data = DistributedArray(data, comm)

    data.sort('minid')

    label = data['minid'].unique()
    
    N = label.bincount()
    
    small = N.local <= 32

    Nlocal = label.bincount(local=True)
    mask = numpy.repeat(small, Nlocal)
 
    # globally shift halo id by one
    label.local += 1
    label.local[mask] = 0

    data['minid'].local[:] = label.local[:]

    data.sort('minid')

    label = data['minid'].unique() 

    data['minid'].local[:] = label.local[:]

    data.sort('origind')
    
    label = data['minid'].local.copy().view('i8')

    return label

def main():
    comm = MPI.COMM_WORLD
    np = split_size_2d(comm.size)

    grid = [
        numpy.linspace(0, 1.0, np[0] + 1, endpoint=True),
        numpy.linspace(0, 1.0, np[1] + 1, endpoint=True),
    ]
    domain = GridND(grid)
    if comm.rank == 0:
        logging.info('grid %s' % str(grid) )
    for i, P in enumerate(read(comm, ns.filename, TPMSnapshotFile)):
        pass
    # make sure in one round all particles are in
    assert i == 0

    pos = P['Position'][::1]
    id = P['ID'][::1]
    Ntot = sum(comm.allgather(len(pos)))

    if comm.rank == 0:
        logging.info('Total number of particles %d, ll %g' % (Ntot, ns.LinkingLength))
    ll = ns.LinkingLength * Ntot ** -0.3333333

    #print pos
    #print ((pos[0] - pos[1]) ** 2).sum()** 0.5, ll
  
    layout = domain.decompose(pos, smoothing=ll * 1)

    tpos = layout.exchange(pos)
    tid = layout.exchange(id)
    logging.info('domain %d has %d particles' % (comm.rank, len(tid)))

    data = cluster.dataset(tpos, boxsize=1.0)
    fof = cluster.fof(data, linking_length=ll, np=0, verbose=True)
    
    # initialize global labels
    minid = equiv_class(fof.labels, tid, op=numpy.fmin)[fof.labels]
    del fof
    del data
    del tpos
    del tid

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
            
        if comm.rank == 0:
            print 'merged ', total, 'halos'

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

    label = densify(minid, comm, thresh=32)

    Nhalo0 = max(comm.allgather(label.max())) + 1

    if comm.rank == 0:
        print 'total halos is', Nhalo0

    # size of halos
    N = numpy.bincount(label.view(dtype='i8'), minlength=Nhalo0)
    comm.Allreduce(MPI.IN_PLACE, N, op=MPI.SUM)
    
    # N[0] is nonhalo

    # sort the labels by halo size
    arg = N[1:].argsort()[::-1] + 1
    P = numpy.arange(Nhalo0)
    P[arg] = numpy.arange(len(arg))
    label = P[label]

    del P

    # redo again
    N = numpy.bincount(label, minlength=Nhalo0)
    comm.Allreduce(MPI.IN_PLACE, N, op=MPI.SUM)

    # do center of mass
    posmin = equiv_class(label, pos, op=numpy.fmin, dense_labels=True, identity=numpy.inf,
                    minlength=len(N))
    comm.Allreduce(MPI.IN_PLACE, posmin, op=MPI.MIN)
    dpos = pos - posmin[label]
    dpos[dpos < -0.5] += 1.0
    dpos[dpos >= 0.5] -= 1.0
    dpos = equiv_class(label, dpos, op=numpy.add, dense_labels=True, minlength=len(N))
    
    comm.Allreduce(MPI.IN_PLACE, dpos, op=MPI.SUM)
    dpos /= N[:, None]
    hpos = posmin + dpos

    if comm.rank == 0:
        print N
        print 'total groups', N.shape
        print 'total particles', N.sum()
        print 'above 32', (N > 32).sum()
    
    with open(ns.output + '.%03d' % comm.rank, 'w') as ff:
        label = numpy.int32(label)
        label.tofile(ff)
    with open(ns.output + '.halo', 'w') as ff:
        numpy.int32(len(N)).tofile(ff)
        numpy.int32(N).tofile(ff)
        numpy.float32(hpos).tofile(ff)

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
