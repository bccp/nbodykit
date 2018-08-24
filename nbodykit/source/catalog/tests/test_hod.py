from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit.tutorials import DemoHaloCatalog
from nbodykit import setup_logging
import shutil
import pytest


setup_logging()

@MPITest([1, 4])
def test_no_seed(comm):

    halos = DemoHaloCatalog('bolshoi', 'rockstar', 0.5, comm=comm)
    hod = halos.populate(Zheng07Model)

    # seed is set randomly
    assert hod.attrs['seed'] is not None

@MPITest([1, 4])
def test_bad_model(comm):

    halos = DemoHaloCatalog('bolshoi', 'rockstar', 0.5, comm=comm)
    with pytest.raises(TypeError):
        hod = halos.populate('Zheng07Model')


@MPITest([1, 4])
def test_failed_populate(comm):

    # the demo halos
    halos = DemoHaloCatalog('bolshoi', 'rockstar', 0.5, comm=comm)

    # initialize model with bad MASS column
    model = Zheng07Model.to_halotools(halos.cosmo, halos.attrs['redshift'], '200c')

    with pytest.raises(Exception):
        hod = halos.populate(model)


@MPITest([1, 4])
def test_no_galaxies(comm):

    halos = DemoHaloCatalog('bolshoi', 'rockstar', 0.5, comm=comm)

    # no galaxies populated into halos
    # NOTE: logMmin is unrealistically large here
    with pytest.raises(ValueError):
        hod = halos.populate(Zheng07Model, seed=42, logMmin=17)

@MPITest([1, 4])
def test_repopulate(comm):

    # initialize the halos
    halos = DemoHaloCatalog('bolshoi', 'rockstar', 0.5, comm=comm)

    # populate the mock first
    hod = halos.populate(Zheng07Model, seed=42)
    size = hod.csize

    # repopulate (with same seed --> same catalog)
    hod.repopulate(seed=42)
    assert hod.csize == size

    # repopulate with random seed
    # make sure root sets a random seed and it's saved
    hod.repopulate(seed=None)
    assert hod.attrs['seed'] is not None

    # new params, same seed
    hod.repopulate(seed=42, alpha=1.0)
    assert hod.csize != size

    # bad param name
    with pytest.raises(ValueError):
        hod.repopulate(seed=42, bad_param_name=1.0)


@MPITest([1, 4])
def test_hod_cm(comm):

    redshift = 0.55
    cosmo = cosmology.Planck15
    BoxSize = 512

    # lognormal particles
    Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=3e-3, BoxSize=BoxSize, Nmesh=128, seed=42, comm=comm)

    # run FOF
    r = FOF(source, linking_length=0.2, nmin=20)
    halos = r.to_halos(cosmo=cosmo, redshift=redshift, particle_mass=1e12, mdef='vir', posdef='cm')

    # make the HOD catalog from halotools catalog
    hod = halos.populate(Zheng07Model, seed=42)

    # RSD offset in 'z' direction
    hod['Position'] += hod['VelocityOffset'] * [0, 0, 1]

    # compute the power
    r = FFTPower(hod.to_mesh(Nmesh=128), mode='2d', Nmu=5, los=[0,0,1])

@MPITest([1, 4])
def test_hod_peak(comm):

    redshift = 0.55
    cosmo = cosmology.Planck15
    BoxSize = 512

    # lognormal particles
    Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=3e-3, BoxSize=BoxSize, Nmesh=128, seed=42, comm=comm)

    source['Density'] = KDDensity(source).density

    # run FOF
    r = FOF(source, linking_length=0.2, nmin=20)
    halos = r.to_halos(cosmo=cosmo, redshift=redshift, particle_mass=1e12, mdef='vir', posdef='peak')

    # make the HOD catalog from halotools catalog
    hod = halos.populate(Zheng07Model, seed=42)

    # RSD offset in 'z' direction
    hod['Position'] += hod['VelocityOffset'] * [0, 0, 1]

    # compute the power
    r = FFTPower(hod.to_mesh(Nmesh=128), mode='2d', Nmu=5, los=[0,0,1])

@MPITest([1, 4])
def test_save(comm):

    redshift = 0.55
    cosmo = cosmology.Planck15
    BoxSize = 512

    # lognormal particles
    Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=3e-3, BoxSize=BoxSize, Nmesh=128, seed=42, comm=comm)

    # run FOF
    r = FOF(source, linking_length=0.2, nmin=20)
    halos = r.to_halos(cosmo=cosmo, redshift=redshift, particle_mass=1e12, mdef='vir')

    # make the HOD catalog from halotools catalog
    hod = halos.populate(Zheng07Model, seed=42)

    # save to a tmp file
    hod.save('tmp-hod.bigfile', ['Position', 'Velocity', 'gal_type'])

    # read tmp file
    cat = BigFileCatalog('tmp-hod.bigfile', header="Header", comm=comm)

    try:
        # check attrs
        for name in hod.attrs:
            numpy.testing.assert_array_equal(cat.attrs[name], hod.attrs[name])

        # check same size
        assert hod.csize == cat.csize

        # check total number of satellites
        nsat1 = comm.allreduce(hod['gal_type'].sum())
        nsat2 = comm.allreduce(cat['gal_type'].sum())
        assert nsat1 == nsat2

    finally:
        comm.barrier()
        if comm.rank == 0:
            shutil.rmtree('tmp-hod.bigfile')
