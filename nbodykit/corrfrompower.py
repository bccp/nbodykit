import numpy

from scipy.interpolate import InterpolatedUnivariateSpline
class interp1d(InterpolatedUnivariateSpline):
  """ this replaces the scipy interp1d which do not always
      pass through the points
      note that kind has to be an integer as it is actually
      a UnivariateSpline.
  """
  def __init__(self, x, y, kind, bounds_error=False, fill_value=numpy.nan, copy=True):
    if copy:
      self.x = x.copy()
      self.y = y.copy()
    else:
      self.x = x
      self.y = y
    InterpolatedUnivariateSpline.__init__(self, self.x, self.y, k=kind)
    self.xmin = self.x[0]
    self.xmax = self.x[-1]
    self.fill_value = fill_value
    self.bounds_error = bounds_error
  def __call__(self, x):
    x = numpy.asarray(x)
    shape = x.shape
    x = x.ravel()
    bad = (x > self.xmax) | (x < self.xmin)
    if self.bounds_error and numpy.any(bad):
      raise ValueError("some values are out of bounds")
    y = InterpolatedUnivariateSpline.__call__(self, x.ravel())
    y[bad] = self.fill_value
    return y.reshape(shape)

def corrfrompower(K, P, R, res=100000):
    """calculate correlation function from power spectrum,
       P is 1d powerspectrum. if R is not None, estimate at
       those points .
       returns R, xi(R)
       internally this does a numerical integral with the trapz
       rule for the radial direction of the fourier transformation,
       with a gaussian damping kernel (learned from Xiaoying) 
       the nan points of P is skipped. (usually when K==0, P is nan)
       input power spectrum is assumed to have the gadget convention,
       AKA, normalized to (2 * pi) ** -3 times sigma_8.
       The integral can also be done on log K instead of K if logscale 
       is True.
    """
    mask = ~numpy.isnan(P) & (K > 0)
    K = K[mask]
    P = P[mask]
    Pfunc = interp1d(K, P, kind=5)
    K = numpy.linspace(K.min(), K.max(), int(res))
    P = Pfunc(K)

    # use a log scale
    weight = K #* numpy.exp(-K**2)
    diff = numpy.log(K)
    XI = [4 * numpy.pi / r * \
        numpy.trapz(P * numpy.sin(K * r) * K * weight, diff) for r in R]
    XI = (2 * numpy.pi) ** -3 * numpy.array(XI)
  
    return R, XI
