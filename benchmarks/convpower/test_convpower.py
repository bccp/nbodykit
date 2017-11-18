from nbodykit.lab import *
import pytest

def run_benchmark(benchmark, sample):

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


def test_strong_scaling(benchmark, sample):

    # run the benchmark
    run_benchmark(benchmark, sample)

@pytest.mark.parametrize('N', [1e4, 1e5, 1e6])
def test_weak_scaling(benchmark, sample, N):

    # set N properly
    sample.N = N

    # run with this config
    run_benchmark(benchmark, sample)
