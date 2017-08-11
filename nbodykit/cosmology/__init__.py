from .core import Cosmology
from .background import PerturbationGrowth
from .ehpower import LinearPowerBase, EHPower, NoWiggleEHPower

# override these with Cosmology classes below
from astropy.cosmology import Planck13, Planck15, WMAP5, WMAP7, WMAP9

# Planck defaults with sigma8, n_s
Planck13 = Cosmology.from_astropy(Planck13, sigma8=0.8288, n_s=0.9611)
"""Planck13 instance of FlatLambdaCDM cosmology

From Planck Collaboration 2014, A&A, 571, A16 (Paper XVI), Table 5 (Planck + WP + highL + BAO)
"""

Planck15 = Cosmology.from_astropy(Planck15, sigma8=0.8159, n_s=0.9667)
"""Planck15 instance of FlatLambdaCDM cosmology

From Planck Collaboration 2016, A&A, 594, A13 (Paper XIII), Table 4 (TT, TE, EE + lowP + lensing + ext)
"""

# WMAP defaults with sigma8, n_s
WMAP5 = Cosmology.from_astropy(WMAP5, sigma8=0.817, n_s=0.962)
"""
WMAP5 instance of FlatLambdaCDM cosmology

From Komatsu et al. 2009, ApJS, 180, 330, doi: 10.1088/0067-0049/180/2/330.
Table 1 (WMAP + BAO + SN ML).
"""

WMAP7 = Cosmology.from_astropy(WMAP7, sigma8=0.810, n_s=0.967)
"""
WMAP7 instance of FlatLambdaCDM cosmology

From Komatsu et al. 2011, ApJS, 192, 18, doi: 10.1088/0067-0049/192/2/18.
Table 1 (WMAP + BAO + H0 ML).
"""

WMAP9 = Cosmology.from_astropy(WMAP9, sigma8=0.820, n_s=0.9608)
"""WMAP9 instance of FlatLambdaCDM cosmology

From Hinshaw et al. 2013, ApJS, 208, 19, doi: 10.1088/0067-0049/208/2/19.
Table 4 (WMAP9 + eCMB + BAO + H0, last column)
"""
