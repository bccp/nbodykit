from nbodykit.plugins import PowerSpectrumStorage
import numpy

class Power2DStorage(PowerSpectrumStorage):
    field_type = "2d"

    @classmethod
    def register(kls):
        PowerSpectrumStorage.add_storage_klass(kls)

    def write(self, data, **meta):
        """
        Write a 2D power spectrum as plain text.
                
        Parameters
        ----------
        data : dict
            the data columns to write out
        meta : dict
            any additional metadata to write out at the end of the file
        """
        keys = data.keys()
        
        # necessary and optional 2D data columns
        necessary = ['k', 'mu', 'power']
        optional = ['modes']
        
        # get the names of all 2D columns to write
        names2D = [k for k in necessary+optional if k in keys]

        # do the writing
        with self.open() as ff:
            if not all(k in keys for k in necessary):
                raise ValueError("To write Power2DStorage, please specify %s keys" %necessary)
            
            # write number of mu and k bins first
            Nmu, Nk = data['k'].shape
            ff.write("%d %d\n" %(Nmu, Nk))
            
            # write out column names
            ff.write(" ".join(names2D) + "\n")
            
            # write out flattened columns
            numpy.savetxt(ff, zip(*[data[k].flat for k in names2D]), '%0.7g')
            
            # write out edges
            if 'edges' not in data:
                raise ValueError("To write Power2DStorage, please specify `edges=[k_edges, mu_edges]`")
                
            edges_names = ['kedges', 'muedges']
            for name, edges in zip(edges_names, data['edges']):
                header = "%s %d\n" %(name, len(edges))
                values = "".join("%0.7g\n" %e for e in edges)
                ff.write(header+values)
            
            # lastly, write out metadata, if any
            if len(meta):
                ff.write("metadata %d\n" %len(meta))
                for k,v in meta.iteritems():
                    if not numpy.isscalar(v):
                        raise NotImplementedError("Power2DStorage cannot write out non-scalar metadata yet")
                    ff.write("%s %s %s\n" %(k, str(v), type(v).__name__))
            
            ff.flush()
            
            