import numpy
import mpsort
from mpi4py import MPI

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

    @property
    def heads(self):
        head = None
        if len(self.local) > 0:
            head = self.local[0]

        return self.comm.allgather(head)

    @property
    def tails(self):
        tail = None
        if len(self.local) > 0:
            tail = self.local[-1]

        return self.comm.allgather(tail)

    def ring(self):
        """
        The item before and the item after the local data.

        This method fetches the last item before the local data,
        and the first item after the local data. If the rank before /
        after current rank is empty, item before / after that rank is
        used. 

        If no item is before / after local data, None is set to prev, next

        Returns
        -------
        prev : scalar
            Item before local data, or None if all ranks before this rank is empty.
        next : scalar
            Item after local data, or None if all ranks after this rank is empty.

        """

        heads = []
        oldhead = None
        for head in self.heads:
            if head is None:
                heads.append(oldhead)
            else:
                heads.append(head)
                oldhead = head
        heads.append(None)

        tails = [None]
        oldtail = None
        for tail in self.tails:
            if tail is None:
                tails.append(oldtail)
            else:
                tails.append(tail)
                oldtail = tail

        prev = tails[self.comm.rank]
        next = heads[self.comm.rank + 1]
        return prev, next
    
    def sort(self, orderby=None):
        """
        Sort array globally by key orderby.

        Due to a limitation of mpsort, self[orderby] must be u8.

        """
        mpsort.sort(self.local, orderby)

    def __getitem__(self, key):
        return DistributedArray(self.local[key], self.comm)

    def unique(self):
        """
        Assign unique labels to sorted local. 

        .. warning ::

            local data must be sorted, and of simple type. (numpy.unique)

        Returns
        -------
        label   :  :py:class:`DistributedArray`
            the new labels, starting from 0

        """
        prev, next = self.ring()
         
        junk, label = numpy.unique(self.local, return_inverse=True)
        if len(self.local) == 0:
            Nunique = 0
        else:
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
        prev, next = self.ring()
        if prev is not None:
            offset = prev
            if len(self.local) > 0:
                if prev != self.local[0]:
                    offset = self.local[0]
        else:
            offset = 0

        self.local -= offset
        N = numpy.bincount(self.local)
        self.local += offset

        if local:
            return N

        heads = self.heads
        tails = self.tails

        distN = DistributedArray(N, self.comm)
        headsN, tailsN = distN.heads, distN.tails

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
    local = numpy.empty((comm.rank + 1), 
            dtype=[('key', 'u8'), ('value', 'u8'), ('rank', 'i8')])
    d = DistributedArray(local)
    local['key'] = numpy.arange(len(local))
    local['value'] = d.comm.rank * 10 + local['key']
    local['rank'] = d.comm.rank

    a = d.comm.allgather(d.local['key'])
    if d.comm.rank == 0:
        print 'old', a

    d.sort('key')
    a = d.comm.allgather(d.local['key'])
    if d.comm.rank == 0:
        print 'new', a

    u = d['key'].unique()
    a = d.comm.allgather(u.local)
    if d.comm.rank == 0:
        print 'unique', a

    N = u.bincount()
    a = d.comm.allgather(N.local)
    if d.comm.rank == 0:
        print 'count', a

    N = u.bincount(local=True)
    a = d.comm.allgather(N)
    if d.comm.rank == 0:
        print 'count local', a

    d['key'].local[:] = u.local
    d.sort('value')

    a = d.comm.allgather(d.local['value'])
    if d.comm.rank == 0:
        print 'back', a

if __name__ == '__main__': 
    test()
