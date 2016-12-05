import numpy

def TSCWindow(cfield):
    """
    Return the Fourier-space kernel that accounts for the convolution of 
    the gridded field with the TSC window function in configuration space
    
    References
    ----------
    see equation 18 (with p=3) of Jing et al 2005 (arxiv:0409240)
    """ 
    kny = cfield.Nmesh / cfield.BoxSize # the Nyquist frequency
    
    for kk, slab in zip(cfield.slabs.x, cfield.slabs):
        for i in range(3):
            wi = kk[i]/kny[i]
            tmp = ( numpy.sin(0.5 * wi) / (0.5 * wi) ) ** 3
            tmp[kk[i] == 0.] = 1.
            slab[...] /= tmp

def CICWindow(cfield):
    """
    Return the Fourier-space kernel that accounts for the convolution of 
    the gridded field with the CIC window function in configuration space
    
    References
    ----------
    see equation 18 (with p=3) of Jing et al 2005 (arxiv:0409240)
    """     
    kny = cfield.Nmesh / cfield.BoxSize # the Nyquist frequency
    
    for kk, slab in zip(cfield.slabs.x, cfield.slabs):
        for i in range(3):
            wi = kk[i]/kny[i]
            tmp = ( numpy.sin(0.5 * wi) / (0.5 * wi) ) ** 2
            tmp[kk[i] == 0.] = 1.
            slab[...] /= tmp
    
def TSCAliasingWindow(cfield):
    """
    Return the Fourier-space kernel that accounts for the convolution of 
    the gridded field with the TSC window function in configuration space,
    as well as the approximate aliasing correction
    
    References
    ----------
    see equation 20 of Jing et al 2005 (arxiv:0409240)
    """   
    kny = cfield.Nmesh / cfield.BoxSize # the Nyquist frequency
    
    for kk, slab in zip(cfield.slabs.x, cfield.slabs):
        for i in range(3):
            wi = kk[i]/kny[i]
            s = numpy.sin(0.5 * wi)**2
            slab[...] /= (1 - s + 2./15 * s**2) ** 0.5
    
def CICAliasingWindow(cfield):
    """
    Return the Fourier-space kernel that accounts for the convolution of 
    the gridded field with the CIC window function in configuration space,
    as well as the approximate aliasing correction
    
    References
    ----------
    see equation 20 of Jing et al 2005 (arxiv:0409240)
    """     
    kny = cfield.Nmesh / cfield.BoxSize # the Nyquist frequency
    for kk, slab in zip(cfield.slabs.x, cfield.slabs):
        for i in range(3):
            wi = kk[i]/kny[i]
            slab[...] /= (1 - 2. / 3 * numpy.sin(0.5 * wi) ** 2) ** 0.5
                    
def NormalizeDC(cfield):
    """
    Normalize by the DC amplitude in Fourier space, which effectively
    divides by the mean in configuration space
    """    
    ind = []
    value = 0.
    found = True
    for ki in cfield.x:
        if (ki != 0).all():
            found = False
            break
        ind.append((ki == 0).nonzero()[0][0])
    if found:
        ind = tuple(ind)
        value = numpy.abs(cfield[ind])
    value = cfield.pm.comm.allreduce(value)
    cfield[:] /= value

        
def RemoveDC(cfield):
    """
    Remove the DC amplitude, which sets the mean of the 
    field in configuration space to zero
    """
    ind = []
    for ki in cfield.x:
        if (ki != 0).all():
            return
        ind.append((ki == 0).nonzero()[0][0])
    ind = tuple(ind)
    cfield[ind] = 0.