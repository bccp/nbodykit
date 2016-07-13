import numpy
from nbodykit.extensionpoints import DataSource
from nbodykit.utils import selectionlanguage
import os.path

class DataSourcePlugin(DataSource):
    plugin_name = "MBIIDMO"
    
    def __init__(self, path, simulation, type, BoxSize, 
                    rsd=None, posf=0.001, velf=0.001, select=None):
        pass
    
    @classmethod
    def register(cls):
        s = cls.schema
        
        s.add_argument("path", help="path to file")
        s.add_argument("simulation", help="name of simulation", choices=["dmo", "mb2"])
        s.add_argument("type", help="type of objects", choices=["Centrals", "Satellites", "Both"])
        s.add_argument("BoxSize", type=cls.BoxSizeParser,
            help="the size of the isotropic box, or the sizes of the 3 box dimensions")
        s.add_argument("-rsd", choices="xyz", help="direction to do redshift distortion")
        s.add_argument("-posf", type=float, help="factor to scale the positions")
        s.add_argument("-velf", type=float, help="factor to scale the velocities")
        s.add_argument("-select", type=selectionlanguage.Query,
            help="row selection based on logmass or subtype, e.g. logmass > 13 and logmass < 15 and subtype == 'A'")

    def read_block(self, block, dtype, optional=False):
            
        if self.type == "Both":
            types = "Centrals", "Satellites"
        else:
            types = [self.type]
            
        try:
            return numpy.concatenate([
                numpy.fromfile(self.path + '/' + type + '/' + self.simulation + '_' + block, dtype=dtype)
                for type in types], axis=0)
        except Exception as e:
            if not optional:
                raise RuntimeError("error reading %s: %s" %(block, str(e)))
            else:
                return numpy.nan
            
            
    def read(self, columns, comm, bunchsize):
        dtype = numpy.dtype([
                ('Position', ('f4', 3)),
                ('Velocity', ('f4', 3)),
                ('Mass', 'f4'),
                ('logmass', 'f8'), 
                ('subtype', 'S1'),
                ('magnitude', ('f8', 5))])
                
        if comm.rank == 0:
            
            pos = self.read_block('pos', dtype['position'])
            data = numpy.empty(len(pos), dtype=dtype)
            data['Position'] = pos
            data['Velocity'] = self.read_block('vel', dtype['velocity'])
            data['logmass'] = numpy.log10(self.read_block('mass', dtype['logmass']))
            data['subtype'] = self.read_block('subtype', dtype['subtype'], optional=True)
            data['magnitude'] = self.read_block('magnitude', dtype['magnitude'], optional=True)
            
            # select based on selection conditions
            if self.select is not None:
                mask = self.select.get_mask(data)
                data = data[mask]
            self.logger.info("total number of galaxies selected is %d / %d" % (len(data), len(pos)))

            data['Position'] *= self.posf
            data['Velocity'] *= self.velf
            data['Mass'] = 1.0
        else:
            data = numpy.empty(0, dtype=dtype)

        if self.rsd is not None:
            dir = 'xyz'.index(self.rsd)
            data['Position'][:, dir] += data['Velocity'][:, dir]
            data['Position'][:, dir] %= self.BoxSize[dir] # enforce periodic boundary conditions

        yield [data[key] if key in data.dtype.names else None for key in columns]
