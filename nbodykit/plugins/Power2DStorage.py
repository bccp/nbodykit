from nbodykit.plugins import PowerSpectrumStorage
import numpy

class Power2DStorage(PowerSpectrumStorage):
    field_type = "2d"

    @classmethod
    def register(kls):
        PowerSpectrumStorage.add_storage_klass(kls)

    def write(self, data):
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
            
            # lastly, write out edges if we have them
            if 'edges' in data:
                edges_names = ['kedges', 'muedges']
                for name, edges in zip(edges_names, data['edges']):
                    header = "%s\n%d\n" %(name, len(edges))
                    values = "".join("%0.7g\n" %e for e in edges)
                    ff.write(header+values)
                    
            ff.flush()
            