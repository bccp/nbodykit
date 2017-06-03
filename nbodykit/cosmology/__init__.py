from .core import Cosmology
from .background import PerturbationGrowth

from .ehpower import LinearPowerBase, EHPower, NoWiggleEHPower

# override these with Cosmology classes below
from astropy.cosmology import Planck13, Planck15, WMAP5, WMAP7, WMAP9

# Planck defaults with sigma8, n_s
Planck13 = Cosmology.from_astropy(Planck13, sigma8=0.8288, n_s=0.9611)
Planck15 = Cosmology.from_astropy(Planck15, sigma8=0.8159, n_s=0.9667)

# WMAP defaults with sigma8, n_s
WMAP5 = Cosmology.from_astropy(WMAP5, sigma8=0.817, n_s=0.962)
WMAP7 = Cosmology.from_astropy(WMAP7, sigma8=0.810, n_s=0.967)
WMAP9 = Cosmology.from_astropy(WMAP9, sigma8=0.820, n_s=0.9608)


