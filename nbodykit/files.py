import numpy
import logging

__all__ = ['read', 'TPMSnapshotFile', 'SnapshotFile']

class SnapshotFile:
    @classmethod
    def enum(filetype, basename):
        """
        Iterate over all files of the type
        """
        i = 0
        while True:
            try:
                yield filetype(basename, i)
            except IOError as e:
                # if file does not open properly, we are done
                break
            i = i + 1

    def read(self, column, mystart, myend):
        if column == "Position":
            return self.read_pos(mystart, myend)
        if column == "ID":
            return self.read_id(mystart, myend)
        if column == "Velocity":
            return self.read_vel(mystart, myend)

    def read_pos(self, mystart, myend):
        """
        Read positions of particles, normalized to (0, 1)        

        Parameters
        ----------
        mystart   : int
            offset to start reading within this file. (inclusive)
        myend     : int
            offset to end reading within this file. (exclusive)

        Returns
        -------
        pos  : array_like (myend - mystart, 3)
            position of particles, normalized to (0, 1)        

        """
        raise NotImplementedError

    def read_id(self, mystart, myend):
        """
        Read ID of particles

        Parameters
        ----------
        mystart   : int
            offset to start reading within this file. (inclusive)
        myend     : int
            offset to end reading within this file. (exclusive)

        Returns
        -------
        ID : array_like (myend - mystart)
            ID of particles, normalized to (0, 1)        

        """
        raise NotImplementedError

class TPMSnapshotFile(SnapshotFile):
    def __init__(self, basename, fid):
        self.filename = basename + ".%02d" % fid
        with open(self.filename, 'r') as ff:
            header = numpy.fromfile(ff, dtype='i4', count=7)
        self.header = header
        self.npart = header[2]

    def read_pos(self, mystart, myend):
        with open(self.filename, 'r') as ff:
            # skip header
            ff.seek(7 * 4, 0)
            # jump to mystart of positions
            ff.seek(mystart * 12, 1)
            return numpy.fromfile(ff, count=myend - mystart, dtype=('f4', 3))

    def read_id(self, mystart, myend):
        with open(self.filename, 'r') as ff:
            # skip header
            ff.seek(7 * 4, 0)
            # jump to mystart of id
            ff.seek(self.npart * 12, 1)
            ff.seek(self.npart * 12, 1)
            ff.seek(mystart * 8, 1)
            return numpy.fromfile(ff, count=myend - mystart, dtype=('i8'))

class Snapshot(object):
    def __init__(self, filename, filetype):
        self.filename = filename
        self.filetype = filetype
        self.npart = numpy.array(
            [ff.npart for ff in filetype.enum(filename)],
            dtype='i8')

    def read(self, column, start, end):
        """this function provides a continuous view of multiple files"""

        NpartPerFile = self.npart
        NpartCumFile = numpy.concatenate([[0], numpy.cumsum(self.npart)])
        data = []
        # add an empty item
        ff = self.filetype(self.filename, 0)
        data.append(ff.read(column, 0, 0))

        for i in range(len(NpartPerFile)):
            if end <= NpartCumFile[i]: continue
            if start >= NpartCumFile[i+1]: continue
            # find the intersection in this file
            mystart = max(start - NpartCumFile[i], 0)
            myend = min(end - NpartCumFile[i], NpartPerFile[i])

            ff = self.filetype(self.filename, i)
            data.append(ff.read(column, mystart, myend))
    
        return numpy.concatenate(data, axis=0)

def read(comm, filename, filetype, columns=['Position', 'ID'], bunchsize=None):
    """
    Parallel reading. This is a generator function.

    Use a for loop. For example
    .. code:: python
        
        for i, P in enumerate(read(comm, 'snapshot', TPMSnapshotFile)):
            ....
            # process P

    Parameters
    ----------
    comm  : :py:class:`MPI.Comm`
        Communicator
    filename : string
        base name of the snapshot file
    bunchsize : int
        Number of particles to read per rank in each iteration.
        if None, all particles are read in one iteration

    filetype : subclass of :py:class:`SnapshotFile`
        type of file

    Yields
    ------
    P   : dict
        P['Position'] is the position of particles, normalized to (0, 1)
        P['ID']       is the ID of particles

    """
    snapshot = None
    if comm.rank == 0:
        snapshot = Snapshot(filename, filetype)
    snapshot = comm.bcast(snapshot)

    Ntot = snapshot.npart.sum()

    mystart = comm.rank * Ntot // comm.size
    myend = (comm.rank + 1) * Ntot // comm.size

    if bunchsize is None:
        bunchsize = int(Ntot)
        # set to a sufficiently large number.

    Nchunk = 0
    for i in range(mystart, myend, bunchsize):
        Nchunk += 1
    
    # ensure every rank yields the same number of times
    # for decompose is a collective operation.

    Nchunk = max(comm.allgather(Nchunk))
    for i in range(Nchunk):
        a, b, c = slice(mystart + i * bunchsize, 
                        mystart + (i +1)* bunchsize)\
                    .indices(myend) 
        P = {}
        for column in columns:
            P[column] = snapshot.read(column, a, b)
        #print comm.allreduce(P['Position'].max(), op=MPI.MAX)
        #print comm.allreduce(P['Position'].min(), op=MPI.MIN)
        #print P['ID'].max(), P['ID'].min()
        yield P
        i = i + bunchsize

