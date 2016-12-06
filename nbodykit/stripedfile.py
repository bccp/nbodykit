import numpy
import logging

class StripeFile:
    @classmethod
    def enum(filetype, basename, args={}):
        """
        Iterate over all files of the type
        """
        i = 0
        while True:
            try:
                yield filetype(basename, i, args)
            except IOError as e:
                # if file does not open properly, we are done
                break
            i = i + 1
    
    column_names = set([
       'Position', 
       'Mass', 
       'ID',
       'Velocity',
       'Label',
    ])
    def read(self, column, mystart, myend):
        """
        Read a property column of particles

        Parameters
        ----------
        mystart   : int
            offset to start reading within this file. (inclusive)
        myend     : int
            offset to end reading within this file. (exclusive)

        Returns
        -------
        data : array_like (myend - mystart)
            data in unspecified units.

        """
        return NotImplementedError

    def write(self, column, mystart, data):
        return NotImplementedError

    def readat(self, offset, nitem, dtype):
        with open(self.filename, 'rb') as ff:
            ff.seek(offset, 0)
            return numpy.fromfile(ff, count=nitem, dtype=dtype)

    def writeat(self, offset, data):
        with open(self.filename, 'rb+') as ff:
            ff.seek(offset, 0)
            return ff.tofile(ff)

class DataStorage(object):
    """ 
    DataStorage provides a continuous view of across several files.
    """
    def __init__(self, path, filetype, filetype_args={}):
        """
        Filetype must be of a subclass of StripeFile 
        """
        self.npart = numpy.array(
            [ff.npart for ff in filetype.enum(path, filetype_args)],
            dtype='i8')

        self.path = path
        self.filetype = filetype
        self.filetype_args = filetype_args
        if len(self.npart) == 0:
            raise IOError("No files were found under `%s`" % path)

    @classmethod
    def create(kls, path, filetype, npart, filetype_args={}):
        """
        Create a striped file. 
        
        npart is a list of npart for each file 
        """
        for fid, npart1 in enumerate(npart):
            filetype.create(path, fid, npart1, filetype_args)

        self = kls(path, filetype, filetype_args)
        return self

    def get_file(self, i):
        return self.filetype(self.path, i, self.filetype_args)

    def read(self, column, start, end):

        NpartPerFile = self.npart
        NpartCumFile = numpy.concatenate([[0], numpy.cumsum(self.npart)])
        data = []
        # add an empty item
        ff = self.filetype(self.path, 0, self.filetype_args)
        data.append(ff.read(column, 0, 0))

        for i in range(len(NpartPerFile)):
            if end <= NpartCumFile[i]: continue
            if start >= NpartCumFile[i+1]: continue
            # find the intersection in this file
            mystart = max(start - NpartCumFile[i], 0)
            myend = min(end - NpartCumFile[i], NpartPerFile[i])

            ff = self.filetype(self.path, i, self.filetype_args)
            data.append(ff.read(column, mystart, myend))
        
        data = numpy.concatenate(data, axis=0)
        if not len(data): data = None
        return data

    def write(self, column, start, data):
        """this function provides a continuous view of multiple files"""

        NpartPerFile = self.npart
        NpartCumFile = numpy.concatenate([[0], numpy.cumsum(self.npart)])
        ff = self.filetype(self.path, 0, self.filetype_args)
        end = start + len(data)
        offset = 0
        for i in range(len(NpartPerFile)):
            if end <= NpartCumFile[i]: continue
            if start >= NpartCumFile[i+1]: continue
            # find the intersection in this file
            mystart = max(start - NpartCumFile[i], 0)
            myend = min(end - NpartCumFile[i], NpartPerFile[i])

            ff = self.filetype(self.path, i, self.filetype_args)
            ff.write(column, mystart, data[offset:offset + myend - mystart])
            offset += myend - mystart
    
        return 

    def iter(self, columns, bunchsize, comm):
        """
        Parallel reading. This is a generator function and a collective 
        operation.

        Use a `for` loop. For example:
        
        .. code-block:: python
            
            for i, P in enumerate(read(comm, 'snapshot', TPMSnapshotFile)):
                ....
                # process P

        Parameters
        ----------
        comm  : :py:class:`MPI.Comm`
            Communicator
        bunchsize : int
            Number of particles to read per rank in each iteration.
            if < 0, all particles are read in one iteration

        Yields
        ------
        data in columns 
        """
        
        Ntot = self.npart.sum()

        mystart = comm.rank * Ntot // comm.size
        myend = (comm.rank + 1) * Ntot // comm.size

        if bunchsize < 0:
            # set to a sufficiently large number.
            bunchsize = int(Ntot)

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
            data = []
            for column in columns:
                data.append(self.read(column, a, b))

            # FIXME: we need to fix this ugly thing
            #print comm.allreduce(P['Position'].max(), op=MPI.MAX)
            #print comm.allreduce(P['Position'].min(), op=MPI.MIN)
            yield data
            i = i + bunchsize
