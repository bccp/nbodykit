from nbodykit.lab import *
from nbodykit import setup_logging
import pytest

setup_logging()

def test_strong_scaling(benchmark, sample):

    # generate fake ra,dec,z
    with benchmark("Data"):
        data = sample.data(seed=42)
        randoms = sample.randoms(10, seed=84)

    # run ConvolvedFFTPower
    with benchmark("Algorithm"):

        # the FKP source
        fkp = FKPCatalog(data, randoms)
        fkp = fkp.to_mesh(Nmesh=sample.Nmesh, dtype='f8', nbar='NZ')

        # compute the multipoles
        r = ConvolvedFFTPower(fkp, poles=[0,2,4], dk=0.005)

    # save meta-data
    benchmark.attrs.update(N=sample.N, sample=sample.name)
