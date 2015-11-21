from nbodykit.plugins import PowerSpectrumStorage
import numpy

class Power1DStorage(PowerSpectrumStorage):
    field_type = "1d"

    @classmethod
    def register(kls):
        PowerSpectrumStorage.add_storage_klass(kls)

    def write(self, data, edges=None, **meta):
        """
        Write out a 1D power spectrum measurement to file
        
        Notes
        -----
        Any lines holding keyword arguments, i.e., `edges` or `metadata`,
        will start with the `#` character. This allows the output
        file to be read using the default arguments of `numpy.loadtxt`,
        which treats lines beginning with `#` as comments and skips them. 
        
        Parameters
        ----------
        data : list of arrays
            A list of 1D arrays specifying the data columns. These
            are written in columns to file first
        edges : array_like, optional
            The edges of the k-bins used for the power spectrum
        meta : 
            Any additional metadata to write to file, specified as keyword 
            arguments
        """
        with self.open() as ff:
            
            # write out the 1D data arrays
            numpy.savetxt(ff, zip(*data), '%0.7g')

            # write out the kedges, if provided
            if edges is not None:
                header = "# edges %d\n" %len(edges)
                values = "".join("# %0.7g\n" %e for e in edges)
                ff.write((header+values).encode())
            
            # lastly, write out metadata, if any
            if len(meta):
                ff.write(("# metadata %d\n" %len(meta)).encode())
                for k,v in meta.items():
                    if not numpy.isscalar(v):
                        raise NotImplementedError("Power2DStorage cannot write out non-scalar metadata yet")
                    ff.write(("# %s %s %s\n" %(k, str(v), type(v).__name__)).encode)
            ff.flush()
            
            
