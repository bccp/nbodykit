from nbodykit.lab import *
from nbodykit import setup_logging
import pytest

setup_logging()

def test_strong_scaling(benchmark, sample):

    # generate fake data
    with benchmark("Data"):
        data = sample.data(seed=42)

    # run
    with benchmark("Algorithm"):

        # r binning
        nbins = 10
        edges = numpy.linspace(10., 150.0, nbins+1)

        # run the algorithm
        r = SimulationBoxPairCount('1d', data, edges, periodic=True)

    # save meta-data
    benchmark.attrs.update(N=sample.N, sample=sample.name)
