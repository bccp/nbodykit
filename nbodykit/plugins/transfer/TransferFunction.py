from nbodykit.extensionpoints import Transfer
import numpy

class AnisotropicCIC(Transfer):
    """
    Divide by a kernel in Fourier space to account for
    the convolution of the gridded quantity with the 
    cloud-in-cell window function in configuration space
    """
    plugin_name = "AnisotropicCIC"

    @classmethod
    def register(kls):
        pass

    def __call__(self, pm, complex):
        for wi in pm.w:
            tmp = (1 - 2. / 3 * numpy.sin(0.5 * wi) ** 2) ** 0.5
            complex[:] /= tmp
            
            
class NormalizeDC(Transfer):
    """
    Removes the DC amplitude in Fourier space, which effectively
    divides by the mean in configuration space
    """
    plugin_name = "NormalizeDC"

    @classmethod
    def register(kls):
        pass

    def __call__(self, pm, complex):
        ind = []
        value = 0.
        found = True
        for wi in pm.w:
            if (wi != 0).all():
                found = False
                break
            ind.append((wi == 0).nonzero()[0][0])
        if found:
            ind = tuple(ind)
            value = numpy.abs(complex[ind])
        value = pm.comm.allreduce(value)
        complex[:] /= value
        
class RemoveDC(Transfer):
    """
    Remove the DC amplitude, which sets the mean of the 
    field in configuration space to zero
    """
    plugin_name = "RemoveDC"

    @classmethod
    def register(kls):
        pass

    def __call__(self, pm, complex):
        ind = []
        for wi in pm.w:
            if (wi != 0).all():
                return
            ind.append((wi == 0).nonzero()[0][0])
        ind = tuple(ind)
        complex[ind] = 0.
        

class VelocityDivergence(Transfer):
    """
    Apply the k-space kernel which transforms
    v_par(k) into vel_divergence(k)
    """
    plugin_name = "VelocityDivergence"

    @classmethod
    def register(kls):
        h = kls.parser
        h.add_argument("velocity_comp", type=str, help="which velocity component to grid, either 'x', 'y', 'z'")
        
    def __call__(self, pm, complex):
        
        comp = "xyz".index(self.velocity_comp)
        for row in range(len(pm.k[0])):
            
            k2 = numpy.float64(pm.k[0][row]**2)
            for ki in pm.k[1:]:
                k2 = k2 + ki[0]**2
            
            if comp == 0:
                kpar = pm.k[0][row]
            else:
                kpar = pm.k[comp]
            with numpy.errstate(invalid='ignore'):
                complex[row] *= -1j * numpy.nan_to_num(k2 / kpar)


        
            
