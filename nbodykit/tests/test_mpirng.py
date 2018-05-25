from runtests.mpi import MPITest
from nbodykit import setup_logging
from nbodykit.mpirng import MPIRandomState
from numpy.testing import assert_array_equal
import numpy
from mpi4py import MPI
import os
import pytest

setup_logging("debug")

@MPITest([4])
def test_mpirng_large_chunk(comm):
    rng = MPIRandomState(comm, seed=1234, size=1, chunksize=10)

    local = rng.uniform()
    all = numpy.concatenate(comm.allgather(local), axis=0)

    rng1 = MPIRandomState(MPI.COMM_SELF, seed=1234, size=rng.csize, chunksize=rng.chunksize)

    correct = rng1.uniform()

    assert_array_equal(all, correct)

@MPITest([4])
def test_mpirng_small_chunk(comm):
    rng = MPIRandomState(comm, seed=1234, size=10, chunksize=3)

    local = rng.uniform()
    all = numpy.concatenate(comm.allgather(local), axis=0)

    rng1 = MPIRandomState(MPI.COMM_SELF, seed=1234, size=rng.csize, chunksize=rng.chunksize)

    correct = rng1.uniform()

    assert_array_equal(all, correct)

@MPITest([4])
def test_mpirng_unique(comm):
    rng = MPIRandomState(comm, seed=1234, size=10, chunksize=3)

    local1 = rng.uniform()
    local2 = rng.uniform()

    # it shouldn't be the same!
    assert (local1 != local2).any()

@MPITest([4])
def test_mpirng_args(comm):
    rng = MPIRandomState(comm, seed=1234, size=10, chunksize=3)

    local = rng.uniform(low=numpy.ones(rng.size) * 0.5)
    all = numpy.concatenate(comm.allgather(local), axis=0)

    rng1 = MPIRandomState(MPI.COMM_SELF, seed=1234, size=rng.csize, chunksize=rng.chunksize)

    correct = rng1.uniform(low=0.5)

    assert_array_equal(all, correct)

@MPITest([4])
def test_mpirng_itemshape(comm):
    rng = MPIRandomState(comm, seed=1234, size=10, chunksize=3)

    local = rng.uniform(low=numpy.ones(rng.size)[:, None] * 0.5, itemshape=(3,))
    all = numpy.concatenate(comm.allgather(local), axis=0)

    rng1 = MPIRandomState(MPI.COMM_SELF, seed=1234, size=rng.csize, chunksize=rng.chunksize)

    correct = rng1.uniform(low=0.5, itemshape=(3,))

    assert_array_equal(all, correct)

@MPITest([4])
def test_mpirng_poisson(comm):
    rng = MPIRandomState(comm, seed=1234, size=10, chunksize=3)

    local = rng.poisson(lam=numpy.ones(rng.size)[:, None] * 0.5, itemshape=(3,))
    all = numpy.concatenate(comm.allgather(local), axis=0)

    rng1 = MPIRandomState(MPI.COMM_SELF, seed=1234, size=rng.csize, chunksize=rng.chunksize)

    correct = rng1.poisson(lam=0.5, itemshape=(3,))

    assert_array_equal(all, correct)

