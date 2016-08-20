from nbodykit.core import Transfer
import numpy

class TSCWindow(Transfer):
    """
    Divide by a kernel in Fourier space to account for
    the convolution of the gridded quantity with the 
    CIC window function in configuration space
    """
    plugin_name = "TSCWindow"
    
    def __init__(self):
        pass
    
    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "divide by a Fourier-space kernel to account for the TSC gridding window function; "
        s.description += "see Jing et al 2005 (arxiv:0409240)"
        
    def __call__(self, pm, complex):
        for wi in pm.w:
            tmp = ( numpy.sin(0.5 * wi) / (0.5 * wi) ) ** 3
            tmp[wi == 0.] = 1.0
            complex[:] /= tmp            

class CICWindow(Transfer):
    """
    Divide by a kernel in Fourier space to account for
    the convolution of the gridded quantity with the 
    CIC window function in configuration space
    """
    plugin_name = "CICWindow"
    
    def __init__(self):
        pass
    
    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "divide by a Fourier-space kernel to account for the CIC gridding window function; "
        s.description += "see Jing et al 2005 (arxiv:0409240)"
        
    def __call__(self, pm, complex):
        for wi in pm.w:
            tmp = ( numpy.sin(0.5 * wi) / (0.5 * wi) ) ** 2
            tmp[wi == 0.] = 1.0
            complex[:] /= tmp
        
class AnisotropicCIC(Transfer):
    """
    Divide by a kernel in Fourier space to account for
    the convolution of the gridded quantity with the 
    CIC window function in configuration space
    """
    plugin_name = "AnisotropicCIC"
    
    def __init__(self):
        pass
    
    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "divide by a Fourier-space kernel to account for the CIC gridding window function; "
        s.description += "see Jing et al 2005 (arxiv:0409240)"
        
    def __call__(self, pm, complex):
        for wi in pm.w:
            tmp = (1 - 2. / 3 * numpy.sin(0.5 * wi) ** 2) ** 0.5
            complex[:] /= tmp
            
class AnisotropicTSC(Transfer):
    """
    Divide by a kernel in Fourier space to account for
    the convolution of the gridded quantity with the 
    TSC window function in configuration space
    """
    plugin_name = "AnisotropicTSC"
    
    def __init__(self):
        pass
    
    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "divide by a Fourier-space kernel to account for the TSC gridding window function; "
        s.description += "see Jing et al 2005 (arxiv:0409240)"
        
    def __call__(self, pm, complex):
        for wi in pm.w:
            s = numpy.sin(0.5 * wi)**2
            tmp = (1 - s + 2./15 * s**2) ** 0.5
            complex[:] /= tmp
            
            
class NormalizeDC(Transfer):
    """
    Normalize by the DC amplitude in Fourier space, which effectively
    divides by the mean in configuration space
    """
    plugin_name = "NormalizeDC"
    def __init__(self):
        pass
    
    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "normalize the DC amplitude in Fourier space, which effectively divides "
        s.description += "by the mean in configuration space"

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
    def __init__(self):
        pass

    @classmethod
    def fill_schema(cls):
        s = cls.schema
        s.description = "remove the DC amplitude in Fourier space, which sets the mean of the "
        s.description += "field in configuration space to zero"

    def __call__(self, pm, complex):
        ind = []
        for wi in pm.w:
            if (wi != 0).all():
                return
            ind.append((wi == 0).nonzero()[0][0])
        ind = tuple(ind)
        complex[ind] = 0.