from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal
import pytest

setup_logging("debug")


@MPITest([4])
def test_bad_input(comm):

    source = UniformCatalog(3e-4, BoxSize=256, seed=42, comm=comm)
    source['gal_type'] = 1

    # cannot rank by missing column
    with pytest.raises(ValueError):
        r = CylindricalGroups(source, 'missing', 10, 10, periodic=True, flat_sky_los=[0,0,1])

    # bad flat_sky_los vector
    with pytest.raises(ValueError):
        r = CylindricalGroups(source, 'gal_type', 10, 10, periodic=True, flat_sky_los=[0,0,0,1])
    with pytest.raises(ValueError):
        r = CylindricalGroups(source, 'gal_type', 10, 10, periodic=True, flat_sky_los=[1,1,1]) # not a unit vector

    # missing BoxSize!
    del source.attrs['BoxSize']
    with pytest.raises(ValueError):
        r = CylindricalGroups(source, 'gal_type', 10, 10, periodic=True, flat_sky_los=[0,0,1])

    # this should work though -- we specified BoxSize
    r = CylindricalGroups(source, 'gal_type', 10, 10, BoxSize=256.0, periodic=False)

    # warn if periodic and no line-of-sight
    with pytest.warns(UserWarning):
        r = CylindricalGroups(source, 'gal_type', 10, 10, BoxSize=256.0, periodic=True)


@MPITest([4])
def test_periodic_cgm(comm):

    source = UniformCatalog(3e-4, BoxSize=256, seed=42, comm=comm)

    # add mass
    logmass = source.rng.uniform(12, 15)
    source['halo_mvir'] = 10**(logmass)

    # add fake galaxy types
    gal_type = numpy.empty(len(source))
    gal_type[logmass<14.5] = 0
    gal_type[logmass>14.5] = 1
    source['gal_type'] = gal_type

    # run the algorithm
    rankby = ['halo_mvir', 'gal_type']
    rpar = 10.0
    rperp = 10.0
    r = CylindricalGroups(source, rpar=rpar, rperp=rperp, rankby=rankby, periodic=True, flat_sky_los=[0,0,1])

    # data for direct CGM
    pos = numpy.concatenate(comm.allgather(source['Position']), axis=0)
    mass = numpy.concatenate(comm.allgather(source['halo_mvir']), axis=0)
    gal_type = numpy.concatenate(comm.allgather(source['gal_type']), axis=0)

    # direct results
    kws = {'periodic':True, 'BoxSize':source.attrs['BoxSize']}
    N_cgm, cgm_gal_type, cen_id = direct_cgm(pos, mass, gal_type, rperp, rpar, **kws)

    # gather and compare
    N_cgm2 = numpy.concatenate(comm.allgather(r.groups['num_cgm_sats']), axis=0)
    cen_id2 = numpy.concatenate(comm.allgather(r.groups['cgm_haloid']), axis=0)
    cgm_gal_type2 = numpy.concatenate(comm.allgather(r.groups['cgm_type']), axis=0)

    assert_array_equal(N_cgm, N_cgm2)
    assert_array_equal(cen_id, cen_id2)
    assert_array_equal(cgm_gal_type, cgm_gal_type2)


@MPITest([1, 4])
def test_nonperiodic_cgm(comm):

    source = UniformCatalog(3e-4, BoxSize=256, seed=42, comm=comm)

    # add mass
    logmass = source.rng.uniform(12, 15)
    source['halo_mvir'] = 10**(logmass)

    # add fake galaxy types
    gal_type = numpy.empty(len(source))
    gal_type[logmass<14.5] = 0
    gal_type[logmass>14.5] = 1
    source['gal_type'] = gal_type

    # run the algorithm
    rankby = ['halo_mvir', 'gal_type']
    rpar = 10.0
    rperp = 10.0
    r = CylindricalGroups(source, rpar=rpar, rperp=rperp, rankby=rankby, periodic=False, flat_sky_los=[0,0,1])

    # data for direct CGM
    pos = numpy.concatenate(comm.allgather(source['Position']), axis=0)
    mass = numpy.concatenate(comm.allgather(source['halo_mvir']), axis=0)
    gal_type = numpy.concatenate(comm.allgather(source['gal_type']), axis=0)

    # direct results
    N_cgm, cgm_gal_type, cen_id = direct_cgm(pos, mass, gal_type, rperp, rpar, periodic=False)

    # gather and compare
    N_cgm2 = numpy.concatenate(comm.allgather(r.groups['num_cgm_sats']), axis=0)
    cen_id2 = numpy.concatenate(comm.allgather(r.groups['cgm_haloid']), axis=0)
    cgm_gal_type2 = numpy.concatenate(comm.allgather(r.groups['cgm_type']), axis=0)

    assert_array_equal(N_cgm, N_cgm2)
    assert_array_equal(cen_id, cen_id2)
    assert_array_equal(cgm_gal_type, cgm_gal_type2)


def direct_cgm(pos, mass, gal_type, rperp, rpar, periodic=False, BoxSize=None):
    """
    Given the position of particles, and the mass and galaxy type data
    to sort by, return the non-periodic CGM results via directly comparing
    each object

    Notes
    -----
    * The function is not collective; all data must be passed to this function
    * The line-of-sight is assumed to be [0,0,1]
    """
    N = len(pos)

    # initialize output arrays
    N_cgm = numpy.zeros(N, dtype='i8')
    cgm_gal_type = numpy.zeros(N, dtype='u1')
    cen_id = numpy.zeros(N, dtype='i8') - 1

    # sort
    dtype = numpy.dtype([
            ('origind', 'u4'),
            ('mass', mass.dtype),
            ('gal_type', gal_type.dtype),
            ('pos', (pos.dtype.str, 3))
            ])
    data = numpy.empty(N, dtype=dtype)
    data['mass'] = mass
    data['gal_type'] = gal_type
    data['pos'] = pos
    data['origind'] = numpy.arange(len(data), dtype='u4')

    # sort pos1 by mass and then gal type
    data.sort(order=['mass', 'gal_type'])
    data = data[::-1]

    ii = 0
    while len(data):

        dr = data['pos'][0] - data['pos'][1:]
        if periodic:
            for axis, col in enumerate(dr.T):
                col[col > BoxSize[axis]*0.5] -= BoxSize[axis]
                col[col <= -BoxSize[axis]*0.5] += BoxSize[axis]
        dr2 = (dr**2).sum(axis=-1)
        rflat_sky_los = dr[:,-1]
        rsky2 = numpy.abs(dr2 - rflat_sky_los ** 2)

        valid = (abs(rflat_sky_los) <= rpar)&(rsky2 <= rperp**2)
        this_cenid = data['origind'][0]
        N_cgm[this_cenid] = valid.sum()
        cgm_gal_type[data['origind'][1:][valid]] = 1
        cen_id[data['origind'][1:][valid]] = ii
        cen_id[data['origind'][0]] = ii

        # delete central
        data = numpy.delete(data, 0, axis=0)

        #delete satellites
        data = numpy.delete(data, numpy.where(valid==True)[0], axis=0)

        ii += 1

    return N_cgm, cgm_gal_type, cen_id
