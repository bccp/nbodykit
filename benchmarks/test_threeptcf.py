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
        nbins = 8
        edges = numpy.linspace(0, 150.0, nbins+1)

        # run the algorithm
        ells = list(range(0, 11))
        r = SimulationBox3PCF(data, ells, edges, BoxSize=sample.BoxSize)

    # save meta-data
    benchmark.attrs.update(N=sample.N, sample=sample.name)
