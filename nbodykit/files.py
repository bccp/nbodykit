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
        with open(self.filename, 'r') as ff:
            ff.seek(offset, 0)
            return numpy.fromfile(ff, count=nitem, dtype=dtype)

    def writeat(self, offset, data):
        with open(self.filename, 'r+') as ff:
            ff.seek(offset, 0)
            return ff.tofile(ff)

class TPMSnapshotFile(SnapshotFile):
    def __init__(self, basename, fid):
        self.filename = basename + ".%02d" % fid

        with open(self.filename, 'rb') as ff:
            header = numpy.fromfile(ff, dtype='i4', count=7)
        self.header = header
        self.npart = int(header[2])
        self.offset_table = {
            'Position' : (('f4', 3), 7 * 4), 
            'Velocity' : (('f4', 3), 7 * 4 + 12 * self.npart),
            'ID' : (('u8'), 7 * 4 + 12 * self.npart * 2),
        }

    @classmethod
    def create(kls, basename, fid, npart, meta={}):
        filename = basename + ".%02d" % fid
        header = numpy.zeros(7, dtype='i4')
        header[2] = npart
        header[0] = 1
        
        with open(filename, 'wb') as ff:
            header.tofile(ff)
        self = kls(basename, fid)
        return self
 
    def read(self, column, mystart=0, myend=-1):
        dtype, offset = self.offset_table[column]
        dtype = numpy.dtype(dtype)
        if myend == -1:
            myend = self.npart

        with open(self.filename, 'rb') as ff:
            # skip header
            ff.seek(offset, 0)
            # jump to mystart of positions
            ff.seek(mystart * dtype.itemsize, 1)
            return numpy.fromfile(ff, count=myend - mystart, dtype=dtype)

    def write(self, column, mystart, data):
        dtype, offset = self.offset_table[column]
        dtype = numpy.dtype(dtype)
        with open(self.filename, 'rb+') as ff:
            # skip header
            ff.seek(offset, 0)
            # jump to mystart of positions
            ff.seek(mystart * dtype.itemsize, 1)
            return data.astype(dtype.base).tofile(ff)

class Snapshot(object):
    def __init__(self, filename, filetype):
        self.npart = numpy.array(
            [ff.npart for ff in filetype.enum(filename)],
            dtype='i8')

        self.filename = filename
        self.filetype = filetype
        if len(self.npart) == 0:
            raise IOError("No files were found under `%s`" % filename)

    @classmethod
    def create(kls, filename, filetype, npart):
        """ create a striped snapshot. 
            npart is a list of npart for each file 
        """
        for fid, npart1 in enumerate(npart):
            filetype.create(filename, fid, npart1)

        self = kls(filename, filetype)
        return self

    def get_file(self, i):
        return self.filetype(self.filename, i)

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

    def write(self, column, start, data):
        """this function provides a continuous view of multiple files"""

        NpartPerFile = self.npart
        NpartCumFile = numpy.concatenate([[0], numpy.cumsum(self.npart)])
        ff = self.filetype(self.filename, 0)
        end = start + len(data)
        offset = 0
        for i in range(len(NpartPerFile)):
            if end <= NpartCumFile[i]: continue
            if start >= NpartCumFile[i+1]: continue
            # find the intersection in this file
            mystart = max(start - NpartCumFile[i], 0)
            myend = min(end - NpartCumFile[i], NpartPerFile[i])

            ff = self.filetype(self.filename, i)
            ff.write(column, mystart, data[offset:offset + myend - mystart])
            offset += myend - mystart
    
        return 

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

        # FIXME: we need to fix this ugly thing
        P['__nread__'] = b - a
        #print comm.allreduce(P['Position'].max(), op=MPI.MAX)
        #print comm.allreduce(P['Position'].min(), op=MPI.MIN)
        #print P['ID'].max(), P['ID'].min()
        yield P
        i = i + bunchsize

class HaloLabelFile(SnapshotFile):
    """
    nbodykit halo label file

    Attributes
    ----------
    npart : int
        Number of particles in this file
    linking_length : float
        linking length. For example, 0.2 or 0.168
        
    """
    def __init__(self, filename, fid):
        self.filename = filename + ".%02d" % fid
        with open(self.filename, 'rb') as ff:
            self.npart = numpy.fromfile(ff, 'i4', 1)
            self.linking_length = numpy.fromfile(ff, 'f4', 1)
        self.offset_table = {
            'Label': ('i4', 8),
        }
        self.read = TPMSnapshotFile.read.__get__(self)
        self.write = TPMSnapshotFile.write.__get__(self)


def ReadPower2DPlainText(filename):
    """
    Reads the plain text storage of a 2D power spectrum measurement,
    as output by the `nbodykit.plugins.Measurement2DStorage` plugin
    
    Returns
    -------
    data : dict
        dictionary holding the `edges` data, as well as the
        data columns for the P(k,mu) measurement
    metadata : dict
        any additional metadata to store as part of the 
        P(k,mu) measurement
    """
    toret = {}
    with open(filename, 'r') as ff:
        
        # read number of k and mu bins are first line
        Nk, Nmu = map(int, ff.readline().split())
        N = Nk*Nmu
        
        # names of data columns on second line
        columns = ff.readline().split()
        
        lines = ff.readlines()
        data = numpy.array([map(float, line.split()) for line in lines[:N]])
        data = data.reshape((Nk, Nmu, -1)) #reshape properly to (Nk, Nmu)
                        
        # store as return dict, making complex arrays from real/imag parts
        i = 0
        while i < len(columns):
            name = columns[i]
            nextname = columns[i+1] if i < len(columns)-1 else ''
            if name.endswith('.real') and nextname.endswith('.imag'):
                name = name.split('.real')[0]
                toret[name] = data[...,i] + 1j*data[...,i+1]
                i += 2
            else:
                toret[name] = data[...,i]
                i += 1
        
        # read the edges for k and mu bins
        edges = []
        l1 = int(lines[N].split()[-1]); N = N+1
        edges.append(numpy.array(map(float, lines[N:N+l1])))
        l2 = int(lines[N+l1].split()[-1]); N = N+l1+1
        edges.append(numpy.array(map(float, lines[N:N+l2])))
        toret['edges'] = edges
        
        # read any metadata
        metadata = {}
        if len(lines) > N+l2:
            N_meta = int(lines[N+l2].split()[-1])
            N = N + l2 + 1
            meta = lines[N:N+N_meta]
            for line in meta:
                fields = line.split()
                cast = fields[-1]
                if cast in __builtins__:
                    metadata[fields[0]] = __builtins__[cast](fields[1])
                elif hasattr(numpy, cast):
                     metadata[fields[0]] = getattr(numpy, cast)(fields[1])
                else:
                    raise TypeError("Metadata must have builtin or numpy type")

    return toret, metadata
    
def ReadPower1DPlainText(filename):
    """
    Reads the plain text storage of a 1D power spectrum measurement,
    as output by the `nbodykit.plugins.Measurement1DStorage` plugin.
    
    Notes
    -----
    *   If `edges` is present in the file, they will be returned
        as part of the metadata, with the key `edges`
    *   If the first line of the file specifies column names, 
        they will be returned as part of the metadata
    
    Returns
    -------
    data : dict
        dictionary holding the `edges` data, as well as the
        data columns for the P(k) measurement
    metadata : dict
        any additional metadata to store as part of the 
        P(k) measurement
    """
    # data list
    data = []
    
    # extract the metadata
    metadata = {}
    make_float = lambda x: float(x[1:])
    with open(filename, 'r') as ff:
        
        currline = 0
        lines = ff.readlines()
        
        metadata['cols'] = None
        if lines[0][0] == '#':
            try:
                metadata['cols'] = lines[0][1:].split()
            except:
                pass
        
        while True:
            
            # break if we are at the EOF
            if currline == len(lines): break
            line = lines[currline]
            
            if not line: 
                currline += 1
                continue
                
            if line[0] != '#':
                data.append(map(float, line.split()))
            else:
                line = line[1:]
                
                # read edges
                if 'edges' in line:
                    fields = line.split()
                    N = int(fields[-1]) # number of edges
                    metadata['edges'] = numpy.array(map(make_float, lines[currline+1:currline+1+N]))
                    currline += 1+N
                    continue
        
                # read metadata
                if 'metadata' in line:                
                    # read and cast the metadata properly
                    fields = line.split()
                    N = int(fields[-1]) # number of individual metadata lines
                    for i in range(N):
                        fields = lines[currline+1+i][1:].split()
                        cast = fields[-1]
                        if cast in __builtins__:
                            metadata[fields[0]] = __builtins__[cast](fields[1])
                        elif hasattr(numpy, cast):
                             metadata[fields[0]] = getattr(numpy, cast)(fields[1])
                        else:
                            raise TypeError("metadata must have builtin or numpy type")
                    currline += 1+N
                    continue
                
            # add to the data
            currline += 1
            
    return numpy.asarray(data), metadata
                
            
                
        
