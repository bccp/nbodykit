from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_array_equal, assert_allclose
import pytest

setup_logging()

@MPITest([1])
def test_boxsize_nmesh(comm):

    # the catalog
    source1 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42, comm=comm)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84, comm=comm)

    cat = MultipleSpeciesCatalog(['data', 'randoms'], source1, source2)

    # this should work (infer BoxSize)
    mesh = cat.to_mesh(Nmesh=32)

    # this should not work (no Nmesh to infer)
    with pytest.raises(ValueError):
        mesh = cat.to_mesh()

    # this not should work
    cat.attrs['data.BoxSize'] *= 10.
    with pytest.raises(ValueError):
        mesh = cat.to_mesh(Nmesh=32)

@MPITest([1, 4])
def test_getitem(comm):

    # the catalog
    source1 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42, comm=comm)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84, comm=comm)
    cat = MultipleSpeciesCatalog(['data', 'randoms'], source1, source2)

    # the mesh
    mesh = cat.to_mesh(Nmesh=32, BoxSize=512)

    for source, name in zip([source1, source2], ['data', 'randoms']):
        submesh = mesh[name] # should be equal to source
        assert submesh.source is cat[name]

@MPITest([1, 4])
def test_compute(comm):

    # the catalog
    source1 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=42, comm=comm)
    source2 = UniformCatalog(nbar=3e-5, BoxSize=512., seed=84, comm=comm)
    cat = MultipleSpeciesCatalog(['data', 'randoms'], source1, source2)

    # the meshes
    mesh = cat.to_mesh(Nmesh=32, BoxSize=512)
    mesh1 = source1.to_mesh(Nmesh=32, BoxSize=512)
    mesh2 = source2.to_mesh(Nmesh=32, BoxSize=512)

    # paint
    real1 = mesh1.to_real_field()
    real2 = mesh2.to_real_field()

    # un-normalize real1 and real2
    real1[:] *= real1.attrs['num_per_cell']
    real2[:] *= real2.attrs['num_per_cell']
    norm = real1.attrs['num_per_cell'] + real2.attrs['num_per_cell']

    # the combined density field
    combined = mesh.to_real_field()

    # must be the same
    assert_allclose(combined.value, (real1.value + real2.value)/norm, atol=1e-5)

@MPITest([1, 4])
def test_paint_interlaced(comm):

    # the test case fails only if there is enough particles to trigger
    # the second loop of the interlaced painter; these parameters will do it.

    # the catalog
    source1 = UniformCatalog(nbar=1e-0, BoxSize=111, seed=111, comm=comm)
    source2 = UniformCatalog(nbar=1e-0, BoxSize=111, seed=111, comm=comm)
    source1['Weight'] = 1.0
    source2['Weight'] = 0.1
    cat = MultipleSpeciesCatalog(['data', 'randoms'], source1, source2)

    # the meshes
    mesh = cat.to_mesh(Nmesh=32, interlaced=True)
    mesh1 = source1.to_mesh(Nmesh=32, interlaced=True)
    mesh2 = source2.to_mesh(Nmesh=32, interlaced=True)

    # paint
    real1 = mesh1.to_real_field()
    real2 = mesh2.to_real_field()
    assert_allclose(real1.cmean(), 1.0)
    assert_allclose(real2.cmean(), 1.0)

    # un-normalize real1 and real2
    real1[:] *= real1.attrs['num_per_cell']
    real2[:] *= real2.attrs['num_per_cell']
    norm = real1.attrs['num_per_cell'] + real2.attrs['num_per_cell']

    # the combined density field
    #combined = mesh.to_real_field()
    combined = mesh.compute()

    assert_allclose(combined.cmean(), 1.0)
    # must be the same
    assert_allclose(combined.value, (real1.value + real2.value)/norm, atol=1e-5)
