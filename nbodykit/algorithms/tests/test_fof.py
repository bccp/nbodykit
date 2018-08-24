from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging

from numpy.testing import assert_allclose

# debug logging
setup_logging("debug")

@MPITest([1, 4])
def test_fof(comm):
    cosmo = cosmology.Planck15

    # lognormal particles
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=3e-3, BoxSize=128., Nmesh=32, seed=42, comm=comm)

    # compute P(k,mu) and multipoles
    fof = FOF(source, linking_length=0.2, nmin=20)
    source['Density'] = KDDensity(source, margin=1).density

    # save the halos
    peaks = fof.find_features(peakcolumn='Density')
    peaks.save("FOF-%d" % comm.size, ['CMPosition', 'CMVelocity', 'Length', 'PeakPosition', 'PeakVelocity'])

@MPITest([1, 4])
def test_fof_parallel_no_merge(comm):
    from pmesh.pm import ParticleMesh
    pm = ParticleMesh(BoxSize=[8, 8, 8], Nmesh=[8, 8, 8], comm=comm)
    Q = pm.generate_uniform_particle_grid()
    cat = ArrayCatalog({'Position' : Q}, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh, comm=comm)

    fof = FOF(cat, linking_length=0.9, nmin=0)
    
    labels = numpy.concatenate(comm.allgather((fof.labels)), axis=0)
    # one particle per group
    assert max(labels) == cat.csize - 1

@MPITest([1, 4])
def test_fof_parallel_merge(comm):
    from pmesh.pm import ParticleMesh
    pm = ParticleMesh(BoxSize=[8, 8, 8], Nmesh=[8, 8, 8], comm=comm)
    Q = pm.generate_uniform_particle_grid(shift=0)
    Q1 = Q.copy()
    Q1[:] += 0.01
    Q2 = Q.copy()
    Q2[:] -= 0.01
    Q3 = Q.copy()
    Q3[:] += 0.02
    cat = ArrayCatalog({'Position' : 
            numpy.concatenate([Q, Q1, Q2, Q3], axis=0)}, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh, comm=comm)

    fof = FOF(cat, linking_length=0.011 * 3 ** 0.5, nmin=0, absolute=True)

    labels = numpy.concatenate(comm.allgather((fof.labels)), axis=0)
    assert max(labels) == pm.Nmesh.prod() - 1
    assert all(numpy.bincount(labels) == 4)

@MPITest([1, 4])
def test_fof_nonperiodic(comm):
    cosmo = cosmology.Planck15

    # lognormal particles
    Plin = cosmology.LinearPower(cosmo, redshift=0.55, transfer='EisensteinHu')
    source = LogNormalCatalog(Plin=Plin, nbar=3e-3, BoxSize=128., Nmesh=32, seed=42, comm=comm)

    source['Density'] = KDDensity(source, margin=1).density

    del source.attrs['BoxSize'] # no boxsize 

    # shift left
    source['Position'] -= 100.0
    fof = FOF(source, linking_length=0.2, nmin=20, periodic=False, absolute=True)
    peaks1 = fof.find_features(peakcolumn='Density')

    # shift right
    source['Position'] += 200.0
    fof = FOF(source, linking_length=0.2, nmin=20, periodic=False, absolute=True)
    peaks2 = fof.find_features(peakcolumn='Density')

    assert_allclose(peaks1['CMPosition'] + 200.0, peaks2['CMPosition'], rtol=1e-6)
    assert_allclose(peaks1['CMVelocity'], peaks2['CMVelocity'], rtol=1e-6)
    assert_allclose(peaks1['PeakPosition'] + 200.0, peaks2['PeakPosition'], rtol=1e-6)
    assert_allclose(peaks1['PeakVelocity'], peaks2['PeakVelocity'], rtol=1e-6)
