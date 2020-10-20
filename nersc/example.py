from nbodykit.lab import *
from nbodykit import setup_logging

setup_logging("debug")

# initialize a linear power spectrum class
cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='CLASS')

# get some lognormal particles
source = LogNormalCatalog(Plin=Plin, nbar=3e-7, BoxSize=1380., Nmesh=8, seed=42)

# apply RSD
source['Position'] += transform.VectorProjection(source['VelocityOffset'], [0,0,1])

# compute P(k,mu) and multipoles
result = FFTPower(source, mode='2d', poles=[0,2,4], los=[0,0,1])

# and save
output = "./nbkit_example_power.json"
result.save(output)
