from nbodykit.lab import *
from nbodykit import setup_logging
import pytest

setup_logging()

def test_strong_scaling(benchmark, sample):

    # generate fake (x,y,z)
    with benchmark("Data"):
        data = sample.data(seed=42)

    # run FOF
    with benchmark("Algorithm"):
        fof = FOF(data, linking_length=0.2, nmin=20)
        peaks = fof.find_features()

    # save meta-data
    benchmark.attrs.update(N=sample.N, sample=sample.name)
