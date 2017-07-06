from runtests.mpi import MPITest
from nbodykit.lab import *
from nbodykit import setup_logging
from numpy.testing import assert_array_equal

setup_logging("debug")

@MPITest([1, 4])
def test_cgm(comm):

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
    hod = HODCatalog(halos.to_halotools())

    # run the algorithm
    rankby = ['halo_mvir', 'gal_type']
    rpar = 10.0
    rperp = 10.0
    r = CylindricalGroups(hod, rpar=rpar, rperp=rperp, rankby=rankby, periodic=False, los=[0,0,1])

    # data for direct CGM
    pos = numpy.concatenate(comm.allgather(hod['Position']), axis=0)
    mass = numpy.concatenate(comm.allgather(hod['halo_mvir']), axis=0)
    gal_type = numpy.concatenate(comm.allgather(hod['gal_type']), axis=0)

    # direct results
    N_cgm, cgm_gal_type, cen_id = direct_nonperiodic_cgm(pos, mass, gal_type, rperp, rpar)

    # gather and compare
    N_cgm2 = numpy.concatenate(comm.allgather(r.groups['num_cgm_sats']), axis=0)
    cen_id2 = numpy.concatenate(comm.allgather(r.groups['cgm_cenid']), axis=0)
    cgm_gal_type2 = numpy.concatenate(comm.allgather(r.groups['cgm_gal_type']), axis=0)

    assert_array_equal(N_cgm, N_cgm2)
    assert_array_equal(cen_id, cen_id2)
    assert_array_equal(cgm_gal_type, cgm_gal_type2)


def direct_nonperiodic_cgm(pos, mass, gal_type, rperp, rpar):
    """
    Given the position of particles, and the mass and galaxy type data
    to sort by, return the non-periodic CGM results via directly comparing
    each object

    Notes
    -----
    * This is non-periodic only
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
    data = data[::-1] # sort in descending order

    ii = 0
    while len(data):

        dr = data['pos'][0] - data['pos'][1:]
        dr2 = (dr**2).sum(axis=-1)
        rlos = dr[:,-1]
        rsky2 = numpy.abs(dr2 - rlos ** 2)

        valid = (abs(rlos) <= rpar)&(rsky2 <= rperp**2)

        this_cenid = data['origind'][0]
        N_cgm[this_cenid] = valid.sum()
        cgm_gal_type[data['origind'][1:][valid]] = 1
        cen_id[data['origind'][1:][valid]] = this_cenid

        # delete central
        data = numpy.delete(data, 0, axis=0)

        #delete satellites
        data = numpy.delete(data, numpy.where(valid==True)[0], axis=0)

        ii += 1

    return N_cgm, cgm_gal_type, cen_id
