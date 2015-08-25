from nbodykit.plugins.fileformats import FileFormat
from argparse import ArgumentParser
import numpy

def yield_segments(npart1, start, end):
    """ 
        Yield (fid, mystart, myend) that overlap start:end 
    """
    file_end = numpy.cumsum(npart1)
    file_start = numpy.concatenate([[0], file_end[1:]], axis=0)

    for i in range(len(npart1)):
        if end <= file_start[i]: continue
        if start >= file_end[i]: continue
        # find the intersection in this file
        mystart = max(start - file_start[i], 0)
        myend = min(end - file_start[i], npart1[i])
        yield i, mystart, myend

class TPM(FileFormat):
    parser = ArgumentParser("")

    parser.add_argument("basename")
    parser.add_argument("nfile", type=int, help='Total number of files')
    parser.add_argument("ng", type=int, help='Number of particles per side')
    parser.add_argument("omegam", type=float, help='OmegaM')
    parser.add_argument("boxsize", type=float, help='BoxSize in Mpc/h')
    parser.add_argument("-labelfile", help='Group label file', default=None)

    def __init__(self, words):
        ns = self.parser.parse_args(words)
        self.basename = ns.basename
        self.size = ns.ng * ns.ng * ns.ng

        self.boxsize = ns.boxsize
        self.omegam = ns.omegam

        self.nfile = ns.nfile
        self.columns = ['Position', 'Velocity', 'ID', 'Label']
        self._npart1 = numpy.array([
            SnapshotFile(self.basename,  i).npart
            for i in range(self.nfile)
        ], dtype='i8')
        self._npart1_label = numpy.array([
            LabelFile(self.basename,  i).npart
            for i in range(self.nfile)
        ], dtype='i8')

    def read(self, column, start, end):
        """this function provides a continuous view of multiple files"""

        data = []
        # add an empty item
        if column == 'Label':
            ff = LabelFile(self.labelfile, 0)
            data.append(ff.read(column, 0, 0))

            for i, mystart, myend in yield_segments(self._npart1, start, end):
                ff = LabelFile(self.basename, i)
                data.append(ff.read(column, mystart, myend))
        elif column == 'Mass':
            return 27.75e10 * self.boxsize ** 3 / self.size * self.omegam
        else:
            ff = SnapshotFile(self.basename, 0)
            data.append(ff.read(column, 0, 0))

            for i, mystart, myend in yield_segments(self._npart1, start, end):
                ff = SnapshotFile(self.basename, i)
                data.append(ff.read(column, mystart, myend))
         
        return numpy.concatenate(data, axis=0)

    def write(self, column, start, data):
        """this function provides a continuous view of multiple files"""

        end = start + len(data)
        offset = 0
        for i, mystart, myend in yield_segments(self,npart1, start, end):
            if column != 'Label':
                ff = SnapshotFile(self.basename, i)
            else:
                ff = LabelFile(self.labelfile, i)
            ff.write(column, mystart, data[offset:offset + myend - mystart])
            offset += myend - mystart
        return 

    @classmethod
    def usage(kls):
        return kls.parser.format_help()

class NBKHalo(FileFormat):
    parser = ArgumentParser("")

    parser.add_argument("halofile")
    parser.add_argument("m0", type=float, help='mass of a particle 27.75 * (BoxSize/Ng) ** 3 * OmegaM')
    parser.add_argument("boxsize", type=float, help='BoxSize in Mpc/h')

    def __init__(self, words):
        ns = self.parser.parse_args(words)
        self.halofile = ns.halofile
        self.m0 = ns.m0
        self.boxsize = ns.boxsize
        ff = HaloFile(ns.halofile)
        self.size = ff.nhalo

    @classmethod
    def usage(kls):
        return kls.parser.format_help()

    def read(self, column, start, end):
        """this function provides a continuous view of multiple files"""
        ff = HaloFile(self.halofile)
        data = ff.read(column, start, end)
        if column == 'Mass':
            data = data * self.m0
        return data

    def write(self, column, start, data):
        """this function provides a continuous view of multiple files"""
        ff = HaloFile(self.halofile)
        if column == 'Mass':
            data = data / self.m0
            # take care of round-off errors
            data = numpy.int32(data + 0.5)
        data = ff.write(column, start, data)

class SingleFile:
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
        return NotImplementedError

    def write(self, column, mystart, data):
        return NotImplementedError

class SnapshotFile(SingleFile):
    def __init__(self, basename, fid):
        self.filename = basename + ".%02d" % fid

        with open(self.filename, 'r') as ff:
            header = numpy.fromfile(ff, dtype='i4', count=7)
        self.header = header
        self.npart = int(header[2])
        self.offset_table = {
            'Position' : (('f4', 3), 7 * 4), 
            'Velocity' : (('f4', 3), 7 * 4 + 12 * self.npart),
            'ID' : (('u8'), 7 * 4 + 12 * self.npart * 2),
        }

    @classmethod
    def create(kls, basename, fid, npart, **kwargs):
        filename = basename + ".%02d" % fid
        header = numpy.zeros(7, dtype='i4')
        header[2] = npart
        header[0] = 1
        
        with open(filename, 'w') as ff:
            header.tofile(ff)
        self = kls(basename, fid)
        return self
 
    def read(self, column, mystart=0, myend=-1):
        dtype, offset = self.offset_table[column]
        dtype = numpy.dtype(dtype)
        if myend == -1:
            myend = self.npart
        
        with open(self.filename, 'r') as ff:
            # skip header
            ff.seek(offset, 0)
            # jump to mystart of positions
            ff.seek(mystart * dtype.itemsize, 1)
            return numpy.fromfile(ff, count=myend - mystart, dtype=dtype)

    def write(self, column, mystart, data):
        dtype, offset = self.offset_table[column]
        dtype = numpy.dtype(dtype)
        with open(self.filename, 'r+') as ff:
            # skip header
            ff.seek(offset, 0)
            # jump to mystart of positions
            ff.seek(mystart * dtype.itemsize, 1)
            return data.astype(dtype.base).tofile(ff)

class HaloFile(SingleFile):
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

        self.offset_table = {
            'Mass': ('i4', 8),
            'Position': (('f4', 3), 8 + 4 * self.nhalo),
            'Velocity': (('f4', 3), 8 + 4 * self.nhalo + 12 * self.nhalo),
        }

        self.read = SnapshotFile.read.__get__(self)
        self.write = SnapshotFile.write.__get__(self)

class LabelFile(SingleFile):
    def __init__(self, filename, fid):
        self.filename = filename + ".%02d" % fid
        with open(self.filename, 'r') as ff:
            self.npart = numpy.fromfile(ff, 'i4', 1)
            self.linking_length = numpy.fromfile(ff, 'f4', 1)
        self.offset_table = {
            'Label': ('i4', 8),
        }
        self.read = SnapshotFile.read.__get__(self)
        self.write = SnapshotFile.write.__get__(self)
    
