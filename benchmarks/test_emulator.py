from nbodykit.lab import *
from nbodykit import setup_logging
import pytest

try: import fastpm
except ImportError: fastpm = None

setup_logging()

def make_halo_catalog(benchmark, source, cosmo, redshift, mdef='vir',
                        mass='Mass', position='Position', velocity='Velocity'):

    from halotools.empirical_models import model_defaults
    from nbodykit.base.catalog import CatalogSourceBase, CatalogSource

    with benchmark('HaloCatalog-checks'):
        # make sure all of the columns are there
        required = ['mass', 'position', 'velocity']
        for name, col in zip(required, [mass, position, velocity]):
            if col is None:
                raise ValueError("the %s column cannot be None in HaloCatalog" %name)
            if col not in source:
                raise ValueError("input source is missing the %s column; '%s' does not exist" %(name, col))

        if not isinstance(source, CatalogSourceBase):
            raise TypeError("input source to HalotoolsCatalog should be a CatalogSource")

    with benchmark('HaloCatalog-init'):
        comm = source.comm
        toret = CatalogSource.__new__(HaloCatalog, comm=comm)
        toret._source = source
        toret.cosmo = cosmo

    # get the attrs from the source
    with benchmark('HaloCatalog-attrs-1'):
        toret.attrs.update(source.attrs)

    # and save the parameters
    with benchmark('HaloCatalog-attrs-2'):
        toret.attrs['redshift'] = redshift
        toret.attrs['cosmo']    = dict(toret.cosmo)
        toret.attrs['mass']     = mass
        toret.attrs['velocity'] = velocity
        toret.attrs['position'] = position
        toret.attrs['mdef']     = mdef

    # names of the mass and radius fields, based on mass def
    with benchmark('HaloCatalog-halo-keys'):
        toret.attrs['halo_mass_key'] = model_defaults.get_halo_mass_key(mdef)
        toret.attrs['halo_radius_key'] = model_defaults.get_halo_boundary_key(mdef)

    # the size
    with benchmark('HaloCatalog-size'):
        toret._size = toret._source.size

    return toret

def to_halos_with_benchmarks(benchmark, fof, particle_mass, cosmo, redshift,
                                mdef='vir', posdef='cm', peakcolumn='Density'):

    with benchmark('to_halos-import'):
        from nbodykit.source import HaloCatalog
    assert posdef in ['cm', 'peak'], "``posdef`` should be 'cm' or 'peak'"

    # meta-data
    with benchmark('to_halos-attrs'):
        attrs = fof._source.attrs.copy()
        attrs.update(fof.attrs)
        attrs['particle_mass'] = particle_mass

    if posdef == 'cm':
        # using the center-of-mass (Position, Velocity, Length) for each halo
        # not needing a column for peaks.
        peakcolumn = None
    else:
        pass

    data = fof_catalog_with_benchmarks(benchmark, fof._source, fof.labels, fof.comm, peakcolumn=peakcolumn, periodic=fof.attrs['periodic'])

    with benchmark('to_halos-ArrayCatalog'):
        data = data[data['Length'] > 0]
        halos = ArrayCatalog(data, **attrs)
        if posdef == 'cm':
            halos['Position'] = halos['CMPosition']
            halos['Velocity'] = halos['CMVelocity']
        elif posdef == 'peak':
            halos['Position'] = halos['PeakPosition']
            halos['Velocity'] = halos['PeakVelocity']
        # add the halo mass column
        halos['Mass'] = particle_mass * halos['Length']

    coldefs = {'mass':'Mass', 'velocity':'Velocity', 'position':'Position'}
    toret = make_halo_catalog(benchmark, halos, cosmo, redshift, mdef=mdef, **coldefs)
    return toret

def fof_catalog_with_benchmarks(benchmark, source, label, comm,
                                position='Position', velocity='Velocity',
                                initposition='InitialPosition',
                                peakcolumn=None, periodic=True):

    from nbodykit.utils import ScatterArray
    from nbodykit.algorithms.fof import count, centerofmass, equiv_class

    # make sure all of the columns are there
    for col in [position, velocity]:
        if col not in source:
            raise ValueError("the column '%s' is missing from parent source; cannot compute halos" %col)

    with benchmark("fof_catalog.count"):
        dtype=[('CMPosition', ('f4', 3)),('CMVelocity', ('f4', 3)),('Length', 'i4')]
        N = count(label, comm=comm)

    if periodic:
        # make sure BoxSize is there
        boxsize = source.attrs.get('BoxSize', None)
        if boxsize is None:
            raise ValueError("cannot compute halo catalog from source without 'BoxSize' in ``attrs`` dict")
    else:
        boxsize = None

    # center of mass position
    with benchmark("fof_catalog.centerofmass-pos"):
        hpos = centerofmass(label, source.compute(source[position]), boxsize=boxsize, comm=comm)

    # center of mass velocity
    with benchmark("fof_catalog.centerofmass-vel"):
        hvel = centerofmass(label, source.compute(source[velocity]), boxsize=None, comm=comm)

    # center of mass initial position
    if initposition in source:
        with benchmark("fof_catalog.centerofmass-initpos"):
            dtype.append(('InitialPosition', ('f4', 3)))
            hpos_init = centerofmass(label, source.compute(source[initposition]), boxsize=boxsize, comm=comm)
            hpos_init

    if peakcolumn is not None:
        assert peakcolumn in source

        dtype.append(('PeakPosition', ('f4', 3)))
        dtype.append(('PeakVelocity', ('f4', 3)))

        density = source[peakcolumn].compute()
        dmax = equiv_class(label, density, op=numpy.fmax, dense_labels=True, minlength=len(N), identity=-numpy.inf)
        comm.Allreduce(MPI.IN_PLACE, dmax, op=MPI.MAX)
        # remove any non-peak particle from the labels
        label1 = label * (density >= dmax[label])

        # compute the center of mass on the new labels
        ppos = centerofmass(label1, source.compute(source[position]), boxsize=boxsize, comm=comm)
        pvel = centerofmass(label1, source.compute(source[velocity]), boxsize=None, comm=comm)

    dtype = numpy.dtype(dtype)
    if comm.rank == 0:
        catalog = numpy.empty(shape=len(N), dtype=dtype)

        catalog['CMPosition'] = hpos
        catalog['CMVelocity'] = hvel
        catalog['Length'] = N
        catalog['Length'][0] = 0
        if 'InitialPosition' in dtype.names:
            catalog['InitialPosition'] = hpos_init

        if peakcolumn is not None:
            catalog['PeakPosition'] = ppos
            catalog['PeakVelocity'] = pvel
    else:
        catalog = None

    with benchmark("fof_catalog.ScatterArray"):
        toret = ScatterArray(catalog, comm, root=0)
    return toret

@pytest.mark.skipif(fastpm is None, reason="fastpm is not installed")
def test_strong_scaling(benchmark):

    from fastpm.nbkit import FastPMCatalogSource

    # setup initial conditions
    cosmo = cosmology.Planck15
    power = cosmology.LinearPower(cosmo, 0)
    linear = LinearMesh(power, BoxSize=512, Nmesh=512)

    with benchmark("FFTPower-Linear"):
        # compute and save linear P(k)
        r = FFTPower(linear, mode="1d")
        r.save("linear-power.json")

    # run a computer simulation!
    with benchmark("Simulation"):
        sim = FastPMCatalogSource(linear, Nsteps=10)

    with benchmark("FFTPower-Matter"):
        # compute and save matter P(k)
        r = FFTPower(sim, mode="1d", Nmesh=512)
        r.save("matter-power.json")

    # run FOF to identify halo groups
    with benchmark("FOF"):
        fof = FOF(sim, 0.2, nmin=20)

    #with benchmark("FOF-fof_catalog"):
    halos = to_halos_with_benchmarks(benchmark, fof, 1e12, cosmo, 0.)

    with benchmark('halotools-import'):
        from halotools.empirical_models import model_defaults

    with benchmark('halotools-get_keys'):
        mass_key = model_defaults.get_halo_mass_key('vir')
        radius_key = model_defaults.get_halo_boundary_key('vir')

    # with benchmark("FFTPower-Halo"):
    #     # compute and save halo P(k)
    #     r = FFTPower(halos, mode="1d", Nmesh=512)
    #     r.save("halos-power.json")

    # # populate halos with galaxies
    # with benchmark("HOD"):
    #     hod = halos.populate(Zheng07Model)
    #
    # # compute and save galaxy power spectrum
    # # result consistent on all ranks
    # with benchmark("FFTPower-Galaxy"):
    #     r = FFTPower(hod, mode='1d', Nmesh=512)
    #     r.save('galaxy-power.json')
