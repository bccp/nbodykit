from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
import shutil

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
