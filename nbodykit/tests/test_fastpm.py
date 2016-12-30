from nbodykit.lab import *
from nbodykit import setup_logging
from mpi4py_test import MPITest
from numpy.testing import assert_allclose

import dask
dask.set_options(get=dask.get)

@MPITest([1, 4])
def test_lpt_particles(comm):
    cosmo = cosmology.Planck15

    linear = Source.LinearMesh(Plin=cosmology.EHPower(cosmo, 0.0),
                BoxSize=1280, Nmesh=64, seed=42, comm=comm)

    lpt = Source.LPTParticles(dlink=linear.to_field(mode='complex'), cosmo=cosmo, redshift=0.0)

    # let's compsre the large scale growth against linear theory.
    Dsquare = []
    redshifts = [9, 5, 2, 1, 0]
    for z in redshifts:
        lpt.set_redshift(z)
        r = FFTPower(lpt, Nmesh=64, mode='1d')
        use = r.power['k'] < 0.02
        pbar = r.power['power'][use].mean()
        Dsquare.append(pbar)

    # linear theory and za at this scale can have 1 few percent of difference.
    assert_allclose(Dsquare, lpt.cosmo.growth_function(redshifts) ** 2 * Dsquare[-1], rtol=1e-2)

    v2 = comm.allreduce((lpt['LPTDisp1']**2).sum(axis=0).compute())
    vdisp = ((v2 / lpt.csize) ** 0.5)

    # should be almost isotropic. Need to add a term in linear to fix the modulus.
    assert_allclose(vdisp, vdisp.mean(), rtol=5e-2)

@MPITest([1, 4])
def test_lpt_particles_grad(comm):
    cosmo = cosmology.Planck15

    linear = Source.LinearMesh(Plin=cosmology.EHPower(cosmo, 0.0),
                BoxSize=1280, Nmesh=4, seed=42, comm=comm)

    dlink = linear.to_field(mode='complex')
    lpt0 = Source.LPTParticles(dlink=dlink, cosmo=cosmo, redshift=0.0)

    def chi2(lpt):
        return comm.allreduce((lpt['LPTDisp1'] ** 2).sum(dtype='f8').compute())

    def grad_chi2(lpt, dlink):
        lpt['GradLPTDisp1'] = 2 * lpt['LPTDisp1']
        return Source.LPTParticles.gradient(dlink, lpt)

    chi2_0 = chi2(lpt0)
    grad_a = grad_chi2(lpt0, dlink)

    for ind1 in numpy.ndindex(*(list(dlink.cshape) + [2])):
        dlink1 = dlink.copy()
        old = dlink1.cgetitem(ind1)
        diff = 1e-6
        new = dlink1.csetitem(ind1, diff + old)
        diff = new - old

        lpt1 = Source.LPTParticles(dlink=dlink1, cosmo=cosmo, redshift=0.0)

        chi2_1 = chi2(lpt1)
        grad = grad_a.cgetitem(ind1)

        assert_allclose(1 + chi2_1 - chi2_0, 1 + grad * diff, 1e-5)

@MPITest([1, 4])
def test_lpt(comm):
    from pmesh.pm import ParticleMesh
    from nbodykit import fastpm

    pm = ParticleMesh(BoxSize=128.0, Nmesh=(4, 4), comm=comm)

    dlink = pm.create(mode='complex')
    dlink.generate_whitenoise(1234)

    def objective(dlink, pm):
        q = fastpm.create_grid(pm)
        dx1 = fastpm.lpt1(dlink, q)
        return comm.allreduce((dx1**2).sum(dtype='f8'))

    def gradient(dlink, pm):
        q = fastpm.create_grid(pm)
        dx1 = fastpm.lpt1(dlink, q)
        grad_dx1 = 2 * dx1
        return fastpm.lpt1_gradient(dlink, q, grad_dx1)

    y0 = objective(dlink, pm)
    yprime = gradient(dlink, pm)

    num = []
    ana = []

    for ind1 in numpy.ndindex(*(list(dlink.cshape) + [2])):
        dlinkl = dlink.copy()
        dlinkr = dlink.copy()
        old = dlink.cgetitem(ind1)
        left = dlinkl.csetitem(ind1, old - 1e-1)
        right = dlinkr.csetitem(ind1, old + 1e-1)
        diff = right - left
        yl = objective(dlinkl, pm)
        yr = objective(dlinkr, pm)
        grad = yprime.cgetitem(ind1)
        #print ind1, yl, yr, grad * diff, yr - yl
        ana.append(grad * diff)
        num.append(yr - yl)

    assert_allclose(num, ana, rtol=1e-5)

@MPITest([1])
def test_drift(comm):
    from nbodykit import fastpm
    x1 = numpy.ones((10, 2))
    p = numpy.ones((10, 2))

    def objective(x1, p):
        x2 = fastpm.drift(x1, p, 2.0)
        return (x2 **2).sum(dtype='f8')

    def gradient(x1, p):
        x2 = fastpm.drift(x1, p, 2.0)
        return fastpm.drift_gradient(x1, p, 2.0, grad_x2=2 * x2)

    y0 = objective(x1, p)
    yprime_x1, yprime_p = gradient(x1, p)

    num = []
    ana = []

    for ind1 in numpy.ndindex(*x1.shape):
        x1l = x1.copy()
        x1r = x1.copy()
        x1l[ind1] -= 1e-3
        x1r[ind1] += 1e-3
        yl = objective(x1l, p)
        yr = objective(x1r, p)
        grad = yprime_x1[ind1]
        num.append(yr - yl)
        ana.append(grad * (x1r[ind1] - x1l[ind1]))

    assert_allclose(num, ana, rtol=1e-5)

    num = []
    ana = []
    for ind1 in numpy.ndindex(*p.shape):
        pl = p.copy()
        pr = p.copy()
        pl[ind1] -= 1e-3
        pr[ind1] += 1e-3
        yl = objective(x1, pl)
        yr = objective(x1, pr)
        grad = yprime_p[ind1]
        num.append(yr - yl)
        ana.append(grad * (pr[ind1] - pl[ind1]))


    assert_allclose(num, ana, rtol=1e-5)

@MPITest([1])
def test_kick(comm):
    from nbodykit import fastpm
    p1 = numpy.ones((10, 2))
    f = numpy.ones((10, 2))

    def objective(p1, f):
        p2 = fastpm.kick(p1, f, 2.0)
        return (p2 **2).sum(dtype='f8')

    def gradient(p1, f):
        p2 = fastpm.kick(p1, f, 2.0)
        return fastpm.kick_gradient(p1, f, 2.0, grad_p2=2 * p2)

    y0 = objective(p1, f)
    yprime_p1, yprime_f = gradient(p1, f)

    num = []
    ana = []

    for ind1 in numpy.ndindex(*p1.shape):
        p1l = p1.copy()
        p1r = p1.copy()
        p1l[ind1] -= 1e-3
        p1r[ind1] += 1e-3
        yl = objective(p1l, f)
        yr = objective(p1r, f)
        grad = yprime_p1[ind1]
        num.append(yr - yl)
        ana.append(grad * (p1r[ind1] - p1l[ind1]))

    assert_allclose(num, ana, rtol=1e-5)

    num = []
    ana = []
    for ind1 in numpy.ndindex(*f.shape):
        fl = f.copy()
        fr = f.copy()
        fl[ind1] -= 1e-3
        fr[ind1] += 1e-3
        yl = objective(p1, fl)
        yr = objective(p1, fr)
        grad = yprime_f[ind1]
        num.append(yr - yl)
        ana.append(grad * (fr[ind1] - fl[ind1]))


    assert_allclose(num, ana, rtol=1e-5)

@MPITest([1, 4])
def test_gravity(comm):
    from pmesh.pm import ParticleMesh
    from nbodykit import fastpm

    pm = ParticleMesh(BoxSize=4.0, Nmesh=(4, 4), comm=comm, method='cic', dtype='f8')

    dlink = pm.create(mode='complex')
    dlink.generate_whitenoise(1234)

    # FIXME: without the shift some particles have near zero dx1.
    # or near 1 dx1.
    # the gradient is not well approximated by the numerical if
    # any of the left or right value shifts beyond the support of
    # the window.
    #
    q = fastpm.create_grid(pm, shift=0.5)
    dx1 = fastpm.lpt1(dlink, q)
    x1 = q + dx1
    def objective(x, pm):
        f = fastpm.gravity(x, pm, 2.0)
        return comm.allreduce((f**2).sum(dtype='f8'))

    def gradient(x, pm):
        f = fastpm.gravity(x, pm, 2.0)
        return fastpm.gravity_gradient(x, pm, 2.0, 2 * f)

    y0 = objective(x1, pm)
    yprime = gradient(x1, pm)

    num = []
    ana = []

    for ind1 in numpy.ndindex(comm.allreduce(x1.shape[0]), x1.shape[1]):
        diff = 1e-1

        start = sum(comm.allgather(x1.shape[0])[:comm.rank])
        end = start + x1.shape[1]
        x1l = x1.copy()
        x1r = x1.copy()
        if ind1[0] >= start and ind1[0] < end:
            x1l[ind1[0] - start, ind1[1]] -= diff
            x1r[ind1[0] - start, ind1[1]] += diff
            grad = yprime[ind1[0] - start, ind1[1]]
        else:
            grad = 0
        grad = comm.allreduce(grad)

        yl = objective(x1l, pm)
        yr = objective(x1r, pm)
        # Watchout : (yr - yl) / (yr + yl) must be large enough for numerical
        # to be accurate
        #print ind1, yl, yr, grad * diff * 2, yr - yl
        num.append(yr - yl)
        ana.append(grad * 2 * diff)

    assert_allclose(num, ana, rtol=1e-5, atol=1e-7)
