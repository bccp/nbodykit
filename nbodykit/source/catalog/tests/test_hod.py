from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
import shutil
import pytest

setup_logging()

@MPITest([4])
def test_no_seed(comm):

    CurrentMPIComm.set(comm)

    redshift = 0.55
    cosmo = cosmology.Planck15
    BoxSize = 512

    # lognormal particles
    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, redshift),
                                nbar=3e-3, BoxSize=BoxSize, Nmesh=128)

    # seed is set randomly
    assert source.attrs['seed'] is not None

    # run FOF
    r = FOF(source, linking_length=0.2, nmin=20)
    halos = r.to_halos(cosmo=cosmo, redshift=redshift, particle_mass=1e12, mdef='vir')

    # make the HOD catalog from halotools catalog
    hod = HODCatalog(halos.to_halotools())

    # seed is set randomly
    assert hod.attrs['seed'] is not None


@MPITest([4])
def test_no_galaxies(comm):

    CurrentMPIComm.set(comm)

    redshift = 0.55
    cosmo = cosmology.Planck15
    BoxSize = 512
    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, redshift),
                                nbar=3e-3, BoxSize=BoxSize, Nmesh=128, seed=42)

    # run FOF
    r = FOF(source, linking_length=0.2, nmin=20)
    halos = r.to_halos(cosmo=cosmo, redshift=redshift, particle_mass=1e12, mdef='vir', posdef='cm')

    # no galaxies populated into halos
    with pytest.raises(ValueError):
        hod = HODCatalog(halos.to_halotools(), seed=42, logMmin=16)

@MPITest([4])
def test_no_halos(comm):

    CurrentMPIComm.set(comm)

    redshift = 0.55
    cosmo = cosmology.Planck15
    BoxSize = 512

    # really low number density
    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, redshift),
                                nbar=3e-5, BoxSize=BoxSize, Nmesh=128, seed=42)

    # run FOF
    r = FOF(source, linking_length=0.2, nmin=20)
    halos = r.to_halos(cosmo=cosmo, redshift=redshift, particle_mass=1e12, mdef='vir', posdef='cm')

    # no halos available
    with pytest.raises(ValueError):
        hod = HODCatalog(halos.to_halotools(), seed=42)


@MPITest([4])
def test_object_columns(comm):

    CurrentMPIComm.set(comm)

    redshift = 0.55
    cosmo = cosmology.Planck15
    BoxSize = 512
    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, redshift),
                                nbar=3e-3, BoxSize=BoxSize, Nmesh=128, seed=42)

    # run FOF
    r = FOF(source, linking_length=0.2, nmin=20)
    halos = r.to_halos(cosmo=cosmo, redshift=redshift, particle_mass=1e12, mdef='vir', posdef='cm')

    # add an object column to the input halo catalog
    cat = halos.to_halotools()
    test =  numpy.random.random(size=len(cat.halo_table)).astype('O')
    cat.halo_table['test'] = test

    # make the HOD
    with pytest.raises(TypeError):
        hod = HODCatalog(cat, seed=42)


@MPITest([4])
def test_bad_catalog(comm):

    CurrentMPIComm.set(comm)

    redshift = 0.55
    cosmo = cosmology.Planck15
    BoxSize = 512
    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, redshift),
                                nbar=3e-3, BoxSize=BoxSize, Nmesh=128, seed=42)

    # run FOF
    r = FOF(source, linking_length=0.2, nmin=20)
    halos = r.to_halos(cosmo=cosmo, redshift=redshift, particle_mass=1e12, mdef='vir', posdef='cm')

    # test converting from astropy to nbodykit cosmology
    cat = halos.to_halotools()
    cat.cosmology = cat.cosmology.engine

    # make the HOD
    hod = HODCatalog(cat, seed=42)

    # missing required column
    del cat.halo_table['halo_id']
    with pytest.raises(ValueError):
        hod = HODCatalog(cat, seed=42)

    # missing required attribute
    del cat.cosmology
    with pytest.raises(AttributeError):
        hod = HODCatalog(cat, seed=42)


@MPITest([4])
def test_repopulate(comm):

    CurrentMPIComm.set(comm)

    redshift = 0.55
    cosmo = cosmology.Planck15
    BoxSize = 512
    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, redshift),
                                nbar=3e-3, BoxSize=BoxSize, Nmesh=128, seed=42)

    # run FOF
    r = FOF(source, linking_length=0.2, nmin=20)
    halos = r.to_halos(cosmo=cosmo, redshift=redshift, particle_mass=1e12, mdef='vir', posdef='cm')

    # make the HOD catalog from halotools catalog
    hod = HODCatalog(halos.to_halotools(), seed=42)
    size = hod.csize

    # repopulate (with same seed --> same catalog)
    hod.repopulate(seed=42)
    assert hod.csize == size

    # repopulate with random seed --> different catalog
    hod.repopulate()
    assert hod.csize != size

    # new params, same seed
    hod.repopulate(seed=42, alpha=1.0)
    assert hod.csize != size

    # bad param name
    with pytest.raises(ValueError):
        hod.repopulate(seed=42, bad_param_name=1.0)


@MPITest([4])
def test_hod_cm(comm):

    CurrentMPIComm.set(comm)

    redshift = 0.55
    cosmo = cosmology.Planck15
    BoxSize = 512

    # lognormal particles
    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, redshift),
                                nbar=3e-3, BoxSize=BoxSize, Nmesh=128, seed=42)

    # run FOF
    r = FOF(source, linking_length=0.2, nmin=20)
    halos = r.to_halos(cosmo=cosmo, redshift=redshift, particle_mass=1e12, mdef='vir', posdef='cm')

    # make the HOD catalog from halotools catalog
    hod = HODCatalog(halos.to_halotools(), seed=42)

    # RSD offset in 'z' direction
    hod['Position'] += hod['VelocityOffset'] * [0, 0, 1]

    # compute the power
    r = FFTPower(hod.to_mesh(Nmesh=128), mode='2d', Nmu=5, los=[0,0,1])

@MPITest([4])
def test_hod_peak(comm):

    CurrentMPIComm.set(comm)

    redshift = 0.55
    cosmo = cosmology.Planck15
    BoxSize = 512

    # lognormal particles
    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, redshift),
                                nbar=3e-3, BoxSize=BoxSize, Nmesh=128, seed=42)

    source['Density'] = KDDensity(source).density

    # run FOF
    r = FOF(source, linking_length=0.2, nmin=20)
    halos = r.to_halos(cosmo=cosmo, redshift=redshift, particle_mass=1e12, mdef='vir', posdef='peak')

    # make the HOD catalog from halotools catalog
    hod = HODCatalog(halos.to_halotools(), seed=42)

    # RSD offset in 'z' direction
    hod['Position'] += hod['VelocityOffset'] * [0, 0, 1]

    # compute the power
    r = FFTPower(hod.to_mesh(Nmesh=128), mode='2d', Nmu=5, los=[0,0,1])

@MPITest([4])
def test_save(comm):

    CurrentMPIComm.set(comm)

    redshift = 0.55
    cosmo = cosmology.Planck15
    BoxSize = 512

    # lognormal particles
    source = LogNormalCatalog(Plin=cosmology.EHPower(cosmo, redshift),
                                nbar=3e-3, BoxSize=BoxSize, Nmesh=128, seed=42)

    # run FOF
    r = FOF(source, linking_length=0.2, nmin=20)
    halos = r.to_halos(cosmo=cosmo, redshift=redshift, particle_mass=1e12, mdef='vir')

    # make the HOD catalog from halotools catalog
    hod = HODCatalog(halos.to_halotools(), seed=42)

    # save to a tmp file
    hod.save('tmp-hod.bigfile', ['Position', 'Velocity', 'gal_type'])

    # read tmp file
    cat = BigFileCatalog('tmp-hod.bigfile', header="Header")

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
