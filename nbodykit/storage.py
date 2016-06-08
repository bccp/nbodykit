import sys
import contextlib
import numpy

class MeasurementStorage(object):
    """
    Class to write a 1D or 2D measurement to a plaintext file
    """    
    def __init__(self, path):
        self.path = path

    @classmethod
    def create(cls, dim, path):
        if dim == '1d':
            return Measurement1DStorage(path)
        elif dim == '2d':
            return Measurement2DStorage(path)
        else:
            raise ValueError("storage dimension must be `1d` or `2d`")
        
    @contextlib.contextmanager
    def open(self):
        if self.path and self.path != '-':
            ff = open(self.path, 'wb')
        else:
            ff = sys.stdout
            
        try:
            yield ff
        finally:
            if ff is not sys.stdout:
                ff.close()

    def write(self, edges, cols, data, **meta):
        return NotImplemented
        
class Measurement1DStorage(MeasurementStorage):

    def write(self, edges, cols, data, **meta):
        """
        Write out a 1D measurement to file
        
        Notes
        -----
        Any lines not holding data values i.e., `edges` or `metadata`,
        will start with the `#` character. This allows the output
        file to be read using the default arguments of `numpy.loadtxt`,
        which treats lines beginning with `#` as comments and skips them. 
        
        Parameters
        ----------
        edges : array_like
            the edges of the bins used for the measurement
        cols : list
            list of names for the corresponding `data`
        data : list of arrays
            list of 1D arrays specifying the data, which are written 
            as columns to file; complex arrays will be written as two
            columns (real and imag).
        meta : 
            Any additional metadata to write to file, specified as keyword 
            arguments
        """
        if len(cols) != len(data):
            raise ValueError("size mismatch between column names and data arrays")
            
        with self.open() as ff:
            
            # split any complex fields into separate columns
            columns = []
            names = []
            for name, column in zip(cols, data):
                if numpy.iscomplexobj(column):
                    columns.append(column.real)
                    columns.append(column.imag)
                    names.append(name + '.real')
                    names.append(name + '.imag')
                else:
                    columns.append(column)
                    names.append(name)

            # write out column names first
            ff.write(("# "+" ".join(names) + "\n").encode())
                     
            # write out the 1D data arrays
            columns = numpy.vstack(columns).T
            numpy.savetxt(ff, columns, '%s')

            # write out the bin edges
            header = "# edges %d\n" %len(edges)
            values = "".join("# %0.7g\n" %e for e in edges)
            ff.write((header+values).encode())
            
            # lastly, write out metadata, if any
            if len(meta):
                ff.write(("# metadata %d\n" %len(meta)).encode())
                for k,v in meta.items():
                    if not numpy.isscalar(v):
                        name = self.__class__.__name__
                        raise NotImplementedError("%s cannot write out non-scalar metadata yet" %name)
                    ff.write(("# %s %s %s\n" %(k, str(v), type(v).__name__)).encode())
            ff.flush()
            
            
class Measurement2DStorage(MeasurementStorage):

    def write(self, edges, cols, data, **meta):
        """
        Write a 2D measurement as plain text.
                
        Parameters
        ----------
        edges : list 
            list specifying the bin edges in both dimensions
        cols : list
            list of names for the corresponding `data`
        data : list
            list of 2D arrays holding the data; complex arrays will be
            treated as two arrays (real and imag).
        meta : dict
            any additional metadata to write out at the end of the file
        """
        if len(cols) != len(data):
            raise ValueError("size mismatch between column names and data arrays")
            
        with self.open() as ff:
            
            # write number of mu and k bins first
            N1, N2 = data[0].shape
            ff.write(("%d %d\n" %(N1, N2)).encode())
            
            # split any complex fields into separate columns
            columns = []
            names = []
            for name, column in zip(cols, data):
                if numpy.iscomplexobj(column):
                    columns.append(column.real)
                    columns.append(column.imag)
                    names.append(name + '.real')
                    names.append(name + '.imag')
                else:
                    columns.append(column)
                    names.append(name)

            # write out column names
            ff.write((" ".join(names) + "\n").encode())
            
            # write out flattened columns
            columns = numpy.dstack(columns).reshape(-1, len(columns))
            numpy.savetxt(ff, columns, '%s')
            
            # write out edges                
            for name, e in zip(['kedges', 'muedges'], edges):
                header = "%s %d\n" %(name, len(e))
                values = "".join("%0.7g\n" %x for x in e)
                ff.write((header+values).encode())
            
            # lastly, write out metadata, if any
            if len(meta):
                ff.write(("metadata %d\n" %len(meta)).encode())
                for k,v in meta.items():
                    if not numpy.isscalar(v):
                        name = self.__class__.__name__
                        raise NotImplementedError("%s cannot write out non-scalar metadata yet" %name)
                    ff.write(("%s %s %s\n" %(k, str(v), type(v).__name__)).encode())
            
            ff.flush()