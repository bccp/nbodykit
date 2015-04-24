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

def read(comm, filename, filetype, BunchSize=None):
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
    BunchSize : int
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
    if comm.rank == 0:
        NpartPerFile = numpy.array(
            [ff.npart for ff in filetype.enum(filename)],
            dtype='i8')
        logging.info('found %d files, npart=%d' % (len(NpartPerFile), sum(NpartPerFile)))
    else:
        NpartPerFile = None

    NpartPerFile = comm.bcast(NpartPerFile)
    NpartCumFile = numpy.concatenate([[0], numpy.cumsum(NpartPerFile)])

    def read_chunk(start, end):
        """this function provides a continuous view of multiple files"""
        pos = []
        id = []
        for i in range(len(NpartPerFile)):
            if end <= NpartCumFile[i]: continue
            if start >= NpartCumFile[i+1]: continue
            # find the intersection in this file
            mystart = max(start - NpartCumFile[i], 0)
            myend = min(end - NpartCumFile[i], NpartPerFile[i])

            ff = filetype(filename, i)
            pos.append(ff.read_pos(mystart, myend))
            id.append(ff.read_id(mystart, myend))
            
        # ensure a good shape even if pos = []
        if len(pos) == 0:
            return (numpy.empty((0, 3), dtype='f4'),
                    numpy.empty((0), dtype='i8'))
        return (numpy.concatenate(pos, axis=0).reshape(-1, 3),
                numpy.concatenate(id, axis=0).reshape(-1))


    Ntot = NpartCumFile[-1]
    mystart = comm.rank * Ntot // comm.size
    myend = (comm.rank + 1) * Ntot // comm.size

    if BunchSize is None:
        BunchSize = int(Ntot)
        # set to a sufficiently large number.

    Nchunk = 0
    for i in range(mystart, myend, BunchSize):
        Nchunk += 1
    
    # ensure every rank yields the same number of times
    # for decompose is a collective operation.

    Nchunk = max(comm.allgather(Nchunk))
    for i in range(Nchunk):
        a, b, c = slice(mystart + i * BunchSize, 
                        mystart + (i +1)* BunchSize)\
                    .indices(myend) 
        P = {}
        pos, id = read_chunk(a, b)
        P['Position'] = pos
        P['ID'] = id
        #print comm.allreduce(P['Position'].max(), op=MPI.MAX)
        #print comm.allreduce(P['Position'].min(), op=MPI.MIN)
        #print P['ID'].max(), P['ID'].min()
        P['Mass'] = 1.0
        yield P
        i = i + BunchSize

