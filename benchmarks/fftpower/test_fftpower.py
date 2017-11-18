from nbodykit.lab import *
import pytest

def run_benchmark(benchmark, sample):

    # lognormal particles
    with benchmark("Data"):
        cat = sample.data(seed=42)

    # run
    with benchmark("Algorithm"):
        r = FFTPower(cat, mode='2d', Nmesh=sample.Nmesh, kmin=0.001, Nmu=10)

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
