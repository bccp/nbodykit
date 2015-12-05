from nbodykit.plugins import MeasurementStorage
import numpy

class Measurement1DStorage(MeasurementStorage):
    field_type = "1d"

    @classmethod
    def register(kls):
        MeasurementStorage.add_storage_klass(kls)

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
            numpy.savetxt(ff, columns, '%0.7g')

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
            
            
