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
        if column == "Label":
            return self.read_label(mystart, myend)

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

    def read_vel(self, mystart, myend):
        """
        Read velocity of particles

        Parameters
        ----------
        mystart   : int
            offset to start reading within this file. (inclusive)
        myend     : int
            offset to end reading within this file. (exclusive)

        Returns
        -------
        vel : array_like (myend - mystart)
            velocity of particles, corrected for ??red-shift distortion        

        """
        raise NotImplementedError

    def read_label(self, mystart, myend):
        raise NotImplementedError

class TPMSnapshotFile(SnapshotFile):
    def __init__(self, basename, fid):
        self.filename = basename + ".%02d" % fid

        with open(self.filename, 'r') as ff:
            header = numpy.fromfile(ff, dtype='i4', count=7)
        self.header = header
        self.npart = int(header[2])

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

    def read_vel(self, mystart, myend):
        with open(self.filename, 'r') as ff:
            # skip header
            ff.seek(7 * 4, 0)
            # jump to mystart of velocity
            ff.seek(self.npart * 12, 1)
            ff.seek(mystart * 12, 1)
            return numpy.fromfile(ff, count=myend - mystart, dtype=('f4', 3))


class Snapshot(object):
    def __init__(self, filename, filetype):
        self.npart = numpy.array(
            [ff.npart for ff in filetype.enum(filename)],
            dtype='i8')

        self.filename = filename
        self.filetype = filetype
        if len(self.npart) == 0:
            raise IOError("No files were found under `%s`" % filename)

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
        with open(self.filename, 'r') as ff:
            self.npart = numpy.fromfile(ff, 'i4', 1)
            self.linking_length = numpy.fromfile(ff, 'f4', 1)
    def read_label(self, mystart, myend):
        with open(self.filename, 'r') as ff:
            ff.seek(8)
            ff.seek(mystart * 4, 1)
            return numpy.fromfile(ff, 'i4', count=myend-mystart)
            

class HaloFile(object):
    """
    nbodykit halo catalogue file

    Attributes
    ----------
    nhalo : int
        Number of halos in the file
    
    """
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'r') as ff:
            self.nhalo = int(numpy.fromfile(ff, 'i4', 1)[0])
            self.linking_length = float(numpy.fromfile(ff, 'f4', 1)[0])

    def read(self, column):
        """
        Read a data column from the catalogue

        Parameters
        ----------
        column : string
            column to read: CenterOfMass or Mass
        
        Returns
        -------
            the data column; all halos are returned.

        """
        if column == 'Position':
            return self.read_pos()
        elif column == 'Mass':
            return self.read_mass()
        elif column == 'Velocity':
            return self.read_vel()
        else:
            raise KeyError("column `%s' unknown" % str(column))

    def read_mass(self):
        with open(self.filename, 'r') as ff:
            ff.seek(8, 0)
            return numpy.fromfile(ff, count=self.nhalo, dtype='i4')

    def read_pos(self):
        with open(self.filename, 'r') as ff:
            ff.seek(8 + self.nhalo * 4, 0)
            return numpy.fromfile(ff, count=self.nhalo, dtype=('f4', 3))

    def read_vel(self):
        with open(self.filename, 'r') as ff:
            ff.seek(8 + self.nhalo * 4, 0)
            ff.seek(self.nhalo * 12, 1)
            return numpy.fromfile(ff, count=self.nhalo, dtype=('f4', 3))

def ReadPower2DPlainText(filename):
    """
    Reads the plain text storage of a 2D power spectrum measurement,
    as output by the `nbodykit.plugins.Power2DStorage` plugin
    
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
        # names of data columns on second line
        columns = ff.readline().split()
        
        # read the column data
        for name in columns: toret[name] = numpy.empty(Nk*Nmu)
        for i in range(Nk*Nmu):
            fields = map(float, ff.readline().split())
            for icol, val in enumerate(fields):
                toret[columns[icol]][i] = val
                
        # reshape properly to (Nk, Nmu)
        for name in columns:        
            toret[name] = toret[name].reshape((Nk,Nmu))
        
        # read the edges for k and mu bins
        edges = []
        edges_names = ['kedges', 'muedges']
        for i, name in enumerate(edges_names):
            fields = ff.readline().split()
            N = int(fields[-1])
            edges.append(numpy.empty(N))
            for j in range(N):
                edges[i][j] = float(ff.readline())
        toret['edges'] = edges
        
        # read any metadata
        metadata = {}
        fields = ff.readline().split()
        if fields[0].strip() == 'metadata':
            N = int(fields[-1])
            for i in range(N):
                fields = ff.readline().split()
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
    as output by the `nbodykit.plugins.Power1DStorage` plugin.
    
    Notes
    -----
    If `edges` is present in the file, they will be returned
    as part of the metadata, with the key `edges`
    
    Returns
    -------
    data : dict
        dictionary holding the `edges` data, as well as the
        data columns for the P(k) measurement
    metadata : dict
        any additional metadata to store as part of the 
        P(k) measurement
    """
    # utility function for removing # char from the start of lines
    def remove_percent_sign(line):
        if isinstance(line, basestring):
            r = line.find('#')
            if r >= 0:
                return line[r+1:]
            else:
                return line
        elif isinstance(line, list):
            return [remove_comments(l) for l in line]
            
    data = []
    metadata = {}
    with open(filename, 'r') as ff:
        
        # loop over each line
        lines = map(remove_percent_sign, ff.readlines())    
        currline = 0
        while True:
            
            # break if we are at the EOF
            if currline == len(lines):
                break
            line = lines[currline]
            
            # read edges
            if 'edges' in line:
                fields = line.split()
                N = int(fields[-1]) # number of edges
                metadata['edges'] = numpy.array(map(float, lines[currline+1:currline+1+N]))
                currline += 1+N
                continue
            
            # read metadata
            if 'metadata' in line:                
                # read and cast the metadata properly
                fields = line.split()
                N = int(fields[-1]) # number of individual metadata lines
                for i in range(N):
                    fields = lines[currline+1+i].split()
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
            data.append(map(float, line.split()))
            currline += 1
            
    return numpy.asarray(data), metadata
                
            
                
        
