from nbodykit.lab import *
import pytest

def test_strong_scaling(benchmark, sample):

    # lognormal particles
    with benchmark("Data"):
        cat = sample.data(seed=42)

    # run
    with benchmark("Algorithm"):
        r = FFTPower(cat, mode='2d', Nmesh=sample.Nmesh, kmin=0.001, Nmu=10)

    # save meta-data
    benchmark.attrs.update(N=sample.N, sample=sample.name)
