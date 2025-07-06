from mpi4py import MPI
import numpy
from nbodykit import cosmology
from nbodykit import transform
from nbodykit.source.catalog import RandomCatalog, UniformCatalog
from nbodykit import setup_logging
from nbodykit.transform import ConstantArray
from numpy.testing import assert_allclose
import pytest

# debug logging
setup_logging("debug")

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_sky_to_cartesian(comm):

    cosmo = cosmology.Planck15

    # make source
    s = RandomCatalog(csize=100, seed=42, comm=comm)

    # ra, dec, z
    s['z']   = s.rng.normal(loc=0.5, scale=0.1)
    s['ra']  = s.rng.uniform(low=110, high=260)
    s['dec'] = s.rng.uniform(low=-3.6, high=60)

    # make the position array
    s['Position1'] = transform.SkyToCartesian(s['ra'], s['dec'], s['z'], cosmo)

    # wrong name
    with pytest.warns(FutureWarning):
        s['Position0'] = transform.SkyToCartesion(s['ra'], s['dec'], s['z'], cosmo)

    s['Position1'] = transform.SkyToCartesian(s['ra'].compute(), s['dec'], s['z'], cosmo)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_cartesian_to_equatorial(comm):

    # make source
    s = UniformCatalog(nbar=10000, BoxSize=1.0, comm=comm)

    # get RA, DEC
    ra, dec = transform.CartesianToEquatorial(s['Position'], observer=[0.5, 0.5, 0.5])

    # check bounds
    assert ((ra >= 0.)&(ra < 360.)).all().compute()
    assert ((dec >= -90)&(dec < 90.)).all().compute()

    ra, dec = transform.CartesianToEquatorial(s['Position'], observer=[0.5, 0.5, 0.5], frame='galactic')

    # check bounds
    assert ((ra >= 0.)&(ra < 360.)).all().compute()
    assert ((dec >= -90)&(dec < 90.)).all().compute()

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_cartesian_to_sky(comm):
    cosmo = cosmology.Planck15

    # make source
    s = UniformCatalog(nbar=10000, BoxSize=1.0, seed=42, comm=comm)

    # get RA, DEC, Z
    ra, dec, z = transform.CartesianToSky(s['Position'], cosmo, observer=[0.5, 0.5, 0.5])

    # reverse and check
    pos2 = transform.SkyToCartesian(ra, dec, z, cosmo, observer=[0.5, 0.5, 0.5])
    assert_allclose(s['Position'], pos2, rtol=1e-5, atol=1e-7)

    _ = transform.CartesianToSky(s['Position'].compute(), cosmo)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_cartesian_to_sky_galactic(comm):
    cosmo = cosmology.Planck15

    # make source
    s = UniformCatalog(nbar=10000, BoxSize=1.0, seed=42, comm=comm)

    # get RA, DEC, Z
    ra, dec, z = transform.CartesianToSky(s['Position'], cosmo, frame='galactic')

    ra1, dec1, z1 = transform.CartesianToSky(s['Position'].compute(), cosmo, frame='galactic')

    assert_allclose(ra, ra1)
    assert_allclose(dec, dec1)
    assert_allclose(z, z1)

    # reverse and check
    pos2 = transform.SkyToCartesian(ra, dec, z, cosmo, frame='galactic')
    numpy.testing.assert_allclose(s['Position'], pos2, rtol=1e-5)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_cartesian_to_sky_velocity(comm):
    cosmo = cosmology.Planck15

    # make source
    s = UniformCatalog(nbar=1e-5, BoxSize=1380., seed=42, comm=comm)

    # real-space redshift
    _, _, z_real = transform.CartesianToSky(s['Position'], cosmo,
                                            observer=[-1e3, -1e3, -1e3])
    # redshift-space redshift
    _, _, z_redshift = transform.CartesianToSky(s['Position'], cosmo,
                                                velocity=s['Velocity'],
                                                observer=[-1e3, -1e3, -1e3])

    numpy.testing.assert_allclose(z_real, z_redshift, rtol=1e-3)

    # bad z max value
    with pytest.raises(ValueError):
        _, _, z = transform.CartesianToSky(s['Position'], cosmo, observer=[-1e4, -1e4, -1e4], zmax=0.5)
        z = z.compute()



@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_stack_columns(comm):

    # make source
    s = RandomCatalog(csize=100, seed=42, comm=comm)

    # add x,y,z
    s['x'] = s.rng.uniform(0, 2600.)
    s['y'] = s.rng.uniform(0, 2600.)
    s['z'] = s.rng.uniform(0, 2600.)

    # stack
    s['Position'] = transform.StackColumns(s['x'], s['y'], s['z'])

    # test equality
    x, y, z = s.compute(s['x'], s['y'], s['z'])
    pos = numpy.vstack([x,y,z]).T
    numpy.testing.assert_array_equal(pos, s['Position'])

    # requires dask array
    s['Position'] = transform.StackColumns(x,y,z)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_halofuncs(comm):
    from nbodykit.cosmology import Planck15
    # make two sources
    # make source
    s = RandomCatalog(csize=300000, seed=42, comm=comm)

    s['mass'] = s.rng.uniform() * 1e13
    s['z'] = s.rng.uniform()

    r = transform.HaloRadius(s['mass'], redshift=s['z'], cosmo=Planck15)
    r.compute()
    r = transform.HaloConcentration(s['mass'], redshift=s['z'], cosmo=Planck15)
    r.compute()
    r = transform.HaloVelocityDispersion(s['mass'], redshift=s['z'], cosmo=Planck15)
    r.compute()
    r = transform.HaloRadius(s['mass'], redshift=0, cosmo=Planck15)
    r.compute()
    r = transform.HaloConcentration(s['mass'], redshift=0, cosmo=Planck15)
    r.compute()
    r = transform.HaloVelocityDispersion(s['mass'], redshift=0, cosmo=Planck15)
    r.compute()

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_combine(comm):

    # make two sources
    s1 = UniformCatalog(3e-6, 2600, comm=comm)
    s2 = UniformCatalog(3e-6, 2600, comm=comm)

    # concatenate all columns
    cat = transform.ConcatenateSources(s1, s2)

    # check the size and columns
    assert cat.size == s1.size + s2.size
    assert set(cat.columns) == set(s1.columns)

    # only one column
    cat = transform.ConcatenateSources(s1, s2, columns='Position')
    pos = numpy.concatenate([numpy.array(s1['Position']), numpy.array(s2['Position'])], axis=0)
    numpy.testing.assert_array_equal(pos, cat['Position'])

    # fail on invalid column
    with pytest.raises(ValueError):
        cat = transform.ConcatenateSources(s1, s2, columns='InvalidColumn')

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_constarray(comm):
    a = ConstantArray(1.0, 1, chunks=1000)
    assert len(a) == 1
    assert a.shape == (1,)
    a = ConstantArray([1.0, 1.0], 1, chunks=1000)
    assert a.shape == (1, 2)

    a = ConstantArray([1.0, 1.0], 3, chunks=1000)
    assert a.shape == (3, 2)

@pytest.mark.parametrize("comm", [MPI.COMM_WORLD,])
@pytest.mark.mpi
def test_vector_projection(comm):
    # make source
    s = UniformCatalog(nbar=1e-5, BoxSize=1380., seed=42, comm=comm)

    x = transform.VectorProjection(s['Position'], [1, 0, 0])
    y = transform.VectorProjection(s['Position'], [0, 1, 0])
    z = transform.VectorProjection(s['Position'], [0, 0, 1])
    d = transform.VectorProjection(s['Position'], [1, 1, 1])

    nx = transform.VectorProjection(s['Position'], [-2, 0, 0])
    ny = transform.VectorProjection(s['Position'], [0, -2, 0])
    nz = transform.VectorProjection(s['Position'], [0, 0, -2])
    nd = transform.VectorProjection(s['Position'], [-2, -2, -2])

    numpy.testing.assert_allclose(x, s['Position'] * [1, 0, 0], rtol=1e-3)
    numpy.testing.assert_allclose(y, s['Position'] * [0, 1, 0], rtol=1e-3)
    numpy.testing.assert_allclose(z, s['Position'] * [0, 0, 1], rtol=1e-3)
    numpy.testing.assert_allclose(d[:, 0], s['Position'].sum(axis=-1) / 3., rtol=1e-3)

    numpy.testing.assert_allclose(nx, s['Position'] * [1, 0, 0], rtol=1e-3)
    numpy.testing.assert_allclose(ny, s['Position'] * [0, 1, 0], rtol=1e-3)
    numpy.testing.assert_allclose(nz, s['Position'] * [0, 0, 1], rtol=1e-3)
    numpy.testing.assert_allclose(nd[:, 0], s['Position'].sum(axis=-1) / 3., rtol=1e-3)
