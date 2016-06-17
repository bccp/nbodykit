import numpy
import logging
from .stripedfile import StripeFile, DataStorage

class TPMSnapshotFile(StripeFile):
    def __init__(self, basename, fid, args={}):
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
        
        # return zero-length array, if we don't support column
        if column not in self.offset_table:
            return numpy.empty(0)
            
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
        
        # crash if we don't support column
        if column not in self.offset_table:
            raise ValueError("available columns to write: %s" %str(self.offset_table))
        dtype, offset = self.offset_table[column]
        dtype = numpy.dtype(dtype)
        with open(self.filename, 'rb+') as ff:
            # skip header
            ff.seek(offset, 0)
            # jump to mystart of positions
            ff.seek(mystart * dtype.itemsize, 1)
            return data.astype(dtype.base).tofile(ff)

class GadgetSnapshotFile(StripeFile):
    def __init__(self, basename, fid, args=dict(ptype=1,
        posdtype='f8', veldtype='f4', massdtype='f4', iddtype='u8')):

        self.ptype = args['ptype']
        self.filename = basename + ".%d" % fid
        posdtype = numpy.dtype(args['posdtype'])
        veldtype = numpy.dtype(args['veldtype'])
        massdtype = numpy.dtype(args['massdtype'])
        iddtype = numpy.dtype(args['iddtype'])
        
        with open(self.filename, 'rb') as ff:
            header = numpy.fromfile(ff, dtype='u1', count=256+8)

        headerdtype = [
          ('N', ('u4', 6)), ('mass', ('f8', 6)),
          ('time', 'f8'), ('redshift', 'f8'),
          ('flag_sfr', 'i4'), ('flag_feedback', 'i4'),
          ('Ntot_low', ('u4', 6)), ('flag_cool', 'i4'),
          ('Nfiles', 'i4'), ('boxsize', 'f8'),
          ('OmegaM', 'f8'), ('OmegaL', 'f8'),
          ('h', 'f8'), ('flag_sft', 'i4'),
          ('flag_met', 'i4'), ('Ntot_high', ('u4', 6)),
          ('flag_entropy', 'i4'), ('flag_double', 'i4'),
          ('flag_ic_info', 'i4'), ('flag_lpt_scalingfactor', 'i4'),
          ('flag_pressure_entropy', 'i1'), ('Ndims', 'i1'),
          ('densitykerneltype', 'i1'), ('unused', ('u1', 45)),
        ]
        header = header[4:-4].view(dtype=headerdtype)[0]
        self.header = header
        self.npart = int(header['N'][self.ptype])

        nall = int(header['N'].sum())
        nmass = int((header['N'] * (header['mass'] == 0)).sum())
        nallc = int(header['N'][:self.ptype].sum())
        nmassc = int((header['N'][:self.ptype] * 
                        (header['mass'][:self.ptype] == 0)).sum())
        o = {}
        offset = 256 + 8
        o['Position'] = ((posdtype, 3), offset + 4 + nallc *
                posdtype.itemsize * 3) 
        offset += posdtype.itemsize * 3 * nall + 8
        o['Velocity'] = ((veldtype, 3), offset + 4 + nallc *
                veldtype.itemsize * 3) 
        offset += veldtype.itemsize * 3* nall + 8
        o['ID'] = (iddtype, offset + 4 + nallc * 8)
        offset += iddtype.itemsize * 3 * nall
        o['Mass'] = (massdtype, offset + 4 + nmassc * massdtype.itemsize),
        self.offset_table = o

    def read(self, column, mystart=0, myend=-1):
        # return zero-length array, if we don't support column
        if column not in self.offset_table:
            return numpy.empty(0)
            
        dtype, offset = self.offset_table[column]
        dtype = numpy.dtype(dtype)
        if myend == -1:
            myend = self.npart

        if column == 'Mass' and self.header['mass'][self.ptype] != 0:
            return numpy.ones(myend - mystart, dtype=dtype) * self.header['mass'][self.ptype]

        with open(self.filename, 'rb') as ff:
            # skip header
            ff.seek(offset, 0)
            # jump to mystart of positions
            ff.seek(mystart * dtype.itemsize, 1)
            return numpy.fromfile(ff, count=myend - mystart, dtype=dtype)

class GadgetGroupTabFile(StripeFile):
    def __init__(self, basename, fid, args=dict(
        posdtype='f8', veldtype='f4', massdtype='f4')):

        self.filename = basename + ".%d" % fid
        posdtype = numpy.dtype(args['posdtype'])
        veldtype = numpy.dtype(args['veldtype'])
        massdtype = numpy.dtype(args['massdtype'])

        headerdtype = [('N', ('i4', (1,))),
                ('Ntot', ('i4', (1,))),
                ('Nids',  'i4'),
                ('TotNids', 'u8'),
                ('Nfiles', 'i4')
        ]
        with open(self.filename, 'rb') as ff:
            header = numpy.fromfile(ff, dtype=headerdtype, count=1)[0]

        self.header = header
        self.npart = int(header['N'])

        o = {}
        offset = header.itemsize
        o['Length'] = ('i4', offset)
        offset += 4 * self.npart
        o['Offset'] = ('i4', offset)
        offset += 4 * self.npart
        o['Mass'] =  (massdtype, offset)
        offset += massdtype.itemsize * self.npart
        o['Position'] = ((posdtype, 3), offset)
        offset += posdtype.itemsize * 3 * self.npart
        o['Velocity'] = ((veldtype, 3), offset)
        offset += veldtype.itemsize * 3 * self.npart
        self.offset_table = o

    def read(self, column, mystart=0, myend=-1):
        # return zero-length array, if we don't support column
        if column not in self.offset_table:
            return numpy.empty(0)
            
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

class HaloLabelFile(StripeFile):
    """
    nbodykit halo label file

    Attributes
    ----------
    npart : int
        Number of particles in this file
    linking_length : float
        linking length. For example, 0.2 or 0.168
        
    """
    def __init__(self, filename, fid, args):
        self.filename = filename + ".%02d" % fid
        with open(self.filename, 'rb') as ff:
            self.npart = numpy.fromfile(ff, 'i4', 1)
            self.linking_length = numpy.fromfile(ff, 'f4', 1)
        self.offset_table = {
            'Label': ('i4', 8),
        }
        self.read = TPMSnapshotFile.read.__get__(self)
        self.write = TPMSnapshotFile.write.__get__(self)


def Read2DPlainText(filename):
    """
    Reads the plain text storage of a 2D measurement,
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
        Nk, Nmu = [int(l) for l in ff.readline().split()]
        N = Nk*Nmu
        
        # names of data columns on second line
        columns = ff.readline().split()
        
        lines = ff.readlines()
        data = numpy.array([float(l) for line in lines[:N] for l in line.split()])
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
        edges.append(numpy.array([float(line) for line in lines[N:N+l1]]))
        l2 = int(lines[N+l1].split()[-1]); N = N+l1+1
        edges.append(numpy.array([float(line) for line in lines[N:N+l2]]))
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
    
def Read1DPlainText(filename):
    """
    Reads the plain text storage of a 1D measurement,
    as output by the `nbodykit.plugins.Measurement1DStorage` plugin.
    
    Notes
    -----
    *   If `edges` is present in the file, they will be returned
        as part of the metadata, with the key `edges`
    *   If the first line of the file specifies column names, 
        they will be returned as part of the metadata with the
        `columns` key
    
    Returns
    -------
    data : array_like
        the 1D data stacked vertically, such that each columns
        represents a separate data variable
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
        
        metadata['columns'] = None
        if lines[0][0] == '#':
            try:
                metadata['columns'] = lines[0][1:].split()
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
                data.append([float(l) for l in line.split()])
            else:
                line = line[1:]
                
                # read edges
                if 'edges' in line:
                    fields = line.split()
                    N = int(fields[-1]) # number of edges
                    metadata['edges'] = numpy.array([make_float(l) for l in lines[currline+1:currline+1+N]])
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
                
            
                
        
