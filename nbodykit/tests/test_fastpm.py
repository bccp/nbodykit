from nbodykit.lab import *
from nbodykit import setup_logging
from mpi4py_test import MPITest

import dask
dask.set_options(get=dask.get)

@MPITest([4])
def test_lpt(comm):
    cosmo = cosmology.Planck15
    linear = Source.LinearMesh(Plin=cosmology.EHPower(cosmo, 0.0),
                BoxSize=128, Nmesh=64, seed=42)

    fastpm = Source.LPTParticles(complex=linear.to_field(mode='complex'), cosmo=cosmo)

    print(fastpm._source['InitialPosition'])
    print(fastpm._source['dx1'].std(axis=0))
