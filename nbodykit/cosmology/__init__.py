from .cosmology import Cosmology
from .background import PerturbationGrowth
from .power import *
from .correlation import CorrelationFunction, xi_to_pk, pk_to_xi

# override these with Cosmology classes below
from astropy.cosmology import Planck13, Planck15, WMAP5, WMAP7, WMAP9

# Planck defaults with sigma8, n_s
_kws = {'ln10^{10}A_s':3.0973, 'n_s':0.9611, 'k_pivot':0.05, 'tau_reio':0.0952}
Planck13 = Cosmology.from_astropy(Planck13, **_kws)
"""Planck13 instance of FlatLambdaCDM cosmology

From Planck Collaboration 2014, A&A, 571, A16 (Paper XVI), Table 5 (Planck + WP + highL + BAO)
"""

_kws = {'ln10^{10}A_s':3.064, 'n_s':0.9667, 'k_pivot':0.05, 'tau_reio':0.066}
Planck15 = Cosmology.from_astropy(Planck15, **_kws)
"""Planck15 instance of FlatLambdaCDM cosmology

From Planck Collaboration 2016, A&A, 594, A13 (Paper XIII), Table 4 (TT, TE, EE + lowP + lensing + ext)
"""

# WMAP defaults with sigma8, n_s
_kws = {'A_s':2.46e-9, 'k_pivot':0.002, 'n_s':0.962, 'tau_reio':0.088}
WMAP5 = Cosmology.from_astropy(WMAP5, **_kws)
"""
WMAP5 instance of FlatLambdaCDM cosmology

From Komatsu et al. 2009, ApJS, 180, 330, doi: 10.1088/0067-0049/180/2/330.
Table 1 (WMAP + BAO + SN ML).
"""

_kws = {'A_s':2.42e-9, 'k_pivot':0.002, 'n_s':0.967, 'tau_reio':0.085}
WMAP7 = Cosmology.from_astropy(WMAP7, **_kws)
"""
WMAP7 instance of FlatLambdaCDM cosmology

From Komatsu et al. 2011, ApJS, 192, 18, doi: 10.1088/0067-0049/192/2/18.
Table 1 (WMAP + BAO + H0 ML).
"""

_kws = {'A_s':2.464e-9, 'k_pivot':0.002, 'n_s':0.9608, 'tau_reio':0.081}
WMAP9 = Cosmology.from_astropy(WMAP9, **_kws)
"""WMAP9 instance of FlatLambdaCDM cosmology

From Hinshaw et al. 2013, ApJS, 208, 19, doi: 10.1088/0067-0049/208/2/19.
Table 4 (WMAP9 + eCMB + BAO + H0, last column)
"""
