import numpy
import mpsort
from mpi4py import MPI
import logging

def GatherArray(data, comm, root=0):
    """
    Gather the input data array from all ranks to the specified ``root``
    
    This uses `Gatherv`, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype
    
    Parameters
    ----------
    data : array_like
        the data on each rank to gather 
    comm : MPI communicator
        the MPI communicator
    root : int
        the rank number to gather the data to
        
    Returns
    -------
    recvbuffer : array_like, None
        the gathered data on root, and `None` otherwise
    """
    if not isinstance(data, numpy.ndarray): 
        raise ValueError("`data` must by numpy array in GatherArray")
        
    # need C-contiguous order
    if not data.flags['C_CONTIGUOUS']:
        data = numpy.ascontiguousarray(data)
    local_length = data.shape[0]
    
    # check dtypes and shapes
    shapes = comm.gather(data.shape, root=root)
    dtypes = comm.gather(data.dtype, root=root)
    
    if comm.rank == root:
        if any(s[1:] != shapes[0][1:] for s in shapes):
            raise ValueError("mismatch between shape[1:] across ranks in GatherArray")
        if any(dt != dtypes[0] for dt in dtypes):
            raise ValueError("mismatch between dtypes across ranks in GatherArray")
        
    shape = data.shape
    dtype = data.dtype
        
    # setup the custom dtype 
    duplicity = numpy.product(numpy.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()
        
    # compute the new shape for each rank
    newlength = comm.allreduce(local_length)
    newshape = list(shape)
    newshape[0] = newlength

    # the return array
    if comm.rank == root:
        recvbuffer = numpy.empty(newshape, dtype=dtype, order='C')
    else:
        recvbuffer = None

    # the recv counts
    counts = comm.allgather(local_length)
    counts = numpy.array(counts, order='C')
    
    # the recv offsets
    offsets = numpy.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]
    
    # gather to root
    comm.Barrier()
    comm.Gatherv([data, dt], [recvbuffer, (counts, offsets), dt], root=root)
    dt.Free()
    
    return recvbuffer
     
def ScatterArray(data, comm, root=0):
    """
    Scatter the input data array across all ranks, assuming `data` is 
    initially only on `root` (and `None` on other ranks)
    
    This uses `Scatterv`, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype
    
    Parameters
    ----------
    data : array_like or None
        on `root`, this gives the data to split and scatter 
    comm : MPI communicator
        the MPI communicator
    root : int
        the rank number that initially has the data
        
    Returns
    -------
    recvbuffer : array_like
        the chunk of `data` that each rank gets
    """
    if comm.rank == root:
        if not isinstance(data, numpy.ndarray): 
            raise ValueError("`data` must by numpy array on root in ScatterArray")
        
        # need C-contiguous order
        if not data.flags['C_CONTIGUOUS']:
            data = numpy.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None
        
    # each rank needs shape/dtype of input data
    shape, dtype = comm.bcast(shape_and_dtype)
    
    # initialize empty data on non-root ranks
    if comm.rank != root:
        np_dtype = numpy.dtype((dtype, shape[1:]))
        data = numpy.empty(0, dtype=np_dtype)
    
    # setup the custom dtype 
    duplicity = numpy.product(numpy.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()
        
    # compute the new shape for each rank
    newshape = list(shape)
    newlength = shape[0] // comm.size
    if comm.rank < shape[0] % comm.size:
        newlength += 1
    newshape[0] = newlength

    # the return array
    recvbuffer = numpy.empty(newshape, dtype=dtype, order='C')

    # the send counts
    counts = comm.allgather(newlength)
    counts = numpy.array(counts, order='C')
    
    # the send offsets
    offsets = numpy.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]
    
    # do the scatter
    comm.Barrier()
    comm.Scatterv([data, (counts, offsets), dt], [recvbuffer, dt])
    dt.Free()
    return recvbuffer

class EmptyRankType(object):
    def __repr__(self):
        return "EmptyRank"
EmptyRank = EmptyRankType()

class LinearTopology(object):
    """ Helper object for the topology of a distributed array 
    """ 
    def __init__(self, local, comm):
        self.local = local
        self.comm = comm

    def heads(self):
        """
        The first items on each rank. 
        
        Returns
        -------
        heads : list
            a list of first items, EmptyRank is used for empty ranks
        """

        head = EmptyRank
        if len(self.local) > 0:
            head = self.local[0]

        return self.comm.allgather(head)

    def tails(self):
        """
        The last items on each rank. 
        
        Returns
        -------
        tails: list
            a list of last items, EmptyRank is used for empty ranks
        """
        tail = EmptyRank
        if len(self.local) > 0:
            tail = self.local[-1]

        return self.comm.allgather(tail)

    def prev(self):
        """
        The item before the local data.

        This method fetches the last item before the local data.
        If the rank before is empty, the rank before is used. 

        If no item is before this rank, EmptyRank is returned

        Returns
        -------
        prev : scalar
            Item before local data, or EmptyRank if all ranks before this rank is empty.

        """

        tails = [EmptyRank]
        oldtail = EmptyRank
        for tail in self.tails():
            if tail is EmptyRank:
                tails.append(oldtail)
            else:
                tails.append(tail)
                oldtail = tail
        prev = tails[self.comm.rank]
        return prev

    def next(self):
        """
        The item after the local data.

        This method the first item after the local data. 
        If the rank after current rank is empty, 
        item after that rank is used. 

        If no item is after local data, EmptyRank is returned.

        Returns
        -------
        next : scalar
            Item after local data, or EmptyRank if all ranks after this rank is empty.

        """
        heads = []
        oldhead = EmptyRank
        for head in self.heads():
            if head is EmptyRank:
                heads.append(oldhead)
            else:
                heads.append(head)
                oldhead = head
        heads.append(EmptyRank)

        next = heads[self.comm.rank + 1]
        return next
    

class DistributedArray(object):
    """
    Distributed Array Object

    A distributed array is striped along ranks

    Attributes
    ----------
    comm : :py:class:`mpi4py.MPI.Comm`
        the communicator

    local : array_like
        the local data

    """
    def __init__(self, local, comm=MPI.COMM_WORLD):
        self.local = local
        self.comm = comm
        self.topology = LinearTopology(local, comm)

    def sort(self, orderby=None):
        """
        Sort array globally by key orderby.

        Due to a limitation of mpsort, self[orderby] must be u8.

        """
        mpsort.sort(self.local, orderby, comm=self.comm)

    def __getitem__(self, key):
        return DistributedArray(self.local[key], self.comm)

    def unique_labels(self):
        """
        Assign unique labels to sorted local. 

        .. warning ::

            local data must be sorted, and of simple type. (numpy.unique)

        Returns
        -------
        label   :  :py:class:`DistributedArray`
            the new labels, starting from 0

        """
        prev, next = self.topology.prev(), self.topology.next()
         
        junk, label = numpy.unique(self.local, return_inverse=True)
        if len(self.local) == 0:
            Nunique = 0
        else:
            # watch out: this is to make sure after shifting first 
            # labels on the next rank is the same as my last label
            # when there is a spill-over.
            if next == self.local[-1]:
                Nunique = len(junk) - 1
            else:
                Nunique = len(junk)

        label += sum(self.comm.allgather(Nunique)[:self.comm.rank])
        return DistributedArray(label, self.comm)

    def bincount(self, local=False):
        """
        Assign count numbers from sorted local data.

        .. warning ::

            local data must be sorted, and of integer type. (numpy.bincount)

        Parameters
        ----------
        local : boolean
            if local is True, only count the local array.

        Returns
        -------
        N :  :py:class:`DistributedArray`
            distributed counts array. If items of the same value spans other
            chunks of array, they are added to N as well.

        Examples
        --------
        if the local array is [ (0, 0), (0, 1)], 
        Then the counts array is [ (3, ), (3, 1)]
        """
        prev = self.topology.prev()
        if prev is not EmptyRank:
            offset = prev
            if len(self.local) > 0:
                if prev != self.local[0]:
                    offset = self.local[0]
        else:
            offset = 0

        N = numpy.bincount(self.local - offset)

        if local:
            return N

        heads = self.topology.heads()
        tails = self.topology.tails()

        distN = DistributedArray(N, self.comm)
        headsN, tailsN = distN.topology.heads(), distN.topology.tails()

        if len(N) > 0:
            for i in reversed(range(self.comm.rank)):
                if tails[i] == self.local[0]:
                    N[0] += tailsN[i]
            for i in range(self.comm.rank + 1, self.comm.size):
                if heads[i] == self.local[-1]:
                    N[-1] += headsN[i]

        return DistributedArray(N, self.comm)

def test():
    comm = MPI.COMM_WORLD
    dtype = numpy.dtype([('key', 'u8'), ('value', 'u8'), ('rank', 'i8')])
    local = numpy.empty((comm.rank), dtype=dtype)
    d = DistributedArray(local)
    local['key'] = numpy.arange(len(local))
    local['value'] = d.comm.rank * 10 + local['key']
    local['rank'] = d.comm.rank

    print(d.topology.heads())

    a = d.comm.allgather(d.local['key'])
    if d.comm.rank == 0:
        print('old', a)

    d.sort('key')
    a = d.comm.allgather(d.local['key'])
    if d.comm.rank == 0:
        print('new', a)

    u = d['key'].unique_labels()
    a = d.comm.allgather(u.local)
    if d.comm.rank == 0:
        print('unique', a)

    N = u.bincount()
    a = d.comm.allgather(N.local)
    if d.comm.rank == 0:
        print('count', a)

    N = u.bincount(local=True)
    a = d.comm.allgather(N)
    if d.comm.rank == 0:
        print('count local', a)

    d['key'].local[:] = u.local
    d.sort('value')

    a = d.comm.allgather(d.local['value'])
    if d.comm.rank == 0:
        print('back', a)

if __name__ == '__main__': 
    test()
