from nbodykit.plugins import PowerSpectrumStorage
import numpy

class Power1DStorage(PowerSpectrumStorage):
    field_type = "1d"

    @classmethod
    def register(kls):
        PowerSpectrumStorage.add_storage_klass(kls)

    def write(self, data, **meta):
        with self.open() as ff:
            numpy.savetxt(ff, zip(*data), '%0.7g')
            ff.flush()
            
            