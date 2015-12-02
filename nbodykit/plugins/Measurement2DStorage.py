from nbodykit.plugins import MeasurementStorage
import numpy

class Measurement2DStorage(MeasurementStorage):
    field_type = "2d"

    @classmethod
    def register(kls):
        MeasurementStorage.add_storage_klass(kls)

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
            columns = numpy.dstack(columns).reshape(-1, len(cols))
            numpy.savetxt(ff, columns, '%0.7g')
            
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
            
            
