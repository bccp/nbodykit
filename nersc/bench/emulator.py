import fastpm
from fastpm.nbkit import FastPMCatalogSource
from nbodykit.lab import *
from nbodykit import setup_logging

setup_logging()

# setup initial conditions
cosmo = cosmology.Planck15
power = cosmology.LinearPower(cosmo, 0)
linear = LinearMesh(power, BoxSize=512, Nmesh=512)

# run a computer simulation!
sim = FastPMCatalogSource(linear, Nsteps=10)

# run FOF to identify halo groups
fof = FOF(sim, 0.2, nmin=20)
halos = fof.to_halos(1e12, cosmo, 0.)

# populate halos with galaxies
hod = HODCatalog(halos.to_halotools())

# compute and save galaxy power spectrum
# result consistent on all ranks
r = FFTPower(hod, mode='1d', Nmesh=128)
r.save('galaxy-power.json')
