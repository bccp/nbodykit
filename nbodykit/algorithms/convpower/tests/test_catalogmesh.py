from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_array_equal, assert_allclose
import pytest

setup_logging()

@MPITest([1, 4])
def test_paint(comm):

    NBAR1 = 3e-5; WEIGHT1 = 1.05
    NBAR2 = 3e-3; WEIGHT2 = 0.95
    P0_FKP = 2e4

    # the catalog
    source1 = UniformCatalog(nbar=NBAR1, BoxSize=512., seed=42, comm=comm)
    source2 = UniformCatalog(nbar=NBAR2, BoxSize=512., seed=84, comm=comm)

    # add completeness weights
    source1['Weight'] = WEIGHT1
    source2['Weight'] = WEIGHT2

    # add n(z)
    source1['NZ'] = NBAR1
    source2['NZ'] = NBAR2

    # add FKP weights
    FKP_WEIGHT1 = 1.0 / (1 + NBAR1*P0_FKP)
    FKP_WEIGHT2 = 1.0 / (1 + NBAR2*P0_FKP)
    source1['FKPWeight'] = FKP_WEIGHT1
    source2['FKPWeight'] = FKP_WEIGHT2

    # initialize the catalog
    cat = FKPCatalog(source1, source2)

    # the meshes
    mesh = cat.to_mesh(Nmesh=32, BoxSize=512)
    mesh1 = source1.to_mesh(Nmesh=32, BoxSize=512)
    mesh2 = source2.to_mesh(Nmesh=32, BoxSize=512)

    # update weights for source1 and source2
    mesh1['Weight']   = source1['Weight'] * source1['FKPWeight']
    mesh2['Weight']   = source2['Weight'] * source2['FKPWeight']

    # paint the re-centered Position
    mesh1['Position'] = source1['Position'] - mesh.attrs['BoxCenter']
    mesh2['Position'] = source2['Position'] - mesh.attrs['BoxCenter']

    # alpha is the sum of Weight
    alpha = 1. * source1.csize * WEIGHT1 / (source2.csize * WEIGHT2)

    # paint
    real1 = mesh1.to_real_field(normalize=False)
    real2 = mesh2.to_real_field(normalize=False)

    # the FKP density field: n_data - alpha*n_randoms
    vol_per_cell = (512.0/32)**3
    fkp_density = (real1.value - alpha*real2.value)/vol_per_cell

    # from the FKPCatalog
    combined = mesh.to_real_field()

    # check meta-data
    assert_allclose(combined.attrs['alpha'], alpha)

    # total number
    assert_allclose(combined.attrs['data.N'], source1.csize)
    assert_allclose(combined.attrs['randoms.N'], source2.csize)

    # total weighted number
    assert_allclose(combined.attrs['data.W'], source1.csize*WEIGHT1)
    assert_allclose(combined.attrs['randoms.W'], source2.csize*WEIGHT2)

    # must be the same
    assert_allclose(combined.value, fkp_density, atol=1e-5)
