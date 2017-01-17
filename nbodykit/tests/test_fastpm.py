from __future__ import print_function
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
                BoxSize=1280, Nmesh=64, seed=142, remove_variance=True, comm=comm)

    lpt = Source.LPTParticles(dlink=linear.to_field(mode='complex'), cosmo=cosmo, redshift=0.0)

#    print(Dsquare, lpt.cosmo.growth_function(redshifts))
    v12 = comm.allreduce((lpt['LPTDisp1']**2).sum(axis=0).compute())
    v1disp = ((v12 / lpt.csize) ** 0.5)

    v22 = comm.allreduce((lpt['LPTDisp2']**2).sum(axis=0).compute())
    v2disp = ((v22 / lpt.csize) ** 0.5)

    print(v1disp, v2disp)

    # should be almost isotropic. Need to add a term in linear to fix the modulus.
    assert_allclose(v1disp, v1disp.mean(), rtol=5e-2)

    # should be almost isotropic. Need to add a term in linear to fix the modulus.
    assert_allclose(v2disp, v2disp.mean(), rtol=5e-2)

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

@MPITest([1, 4])
def test_lpt_particles_grad(comm):
    cosmo = cosmology.Planck15

    linear = Source.LinearMesh(Plin=cosmology.EHPower(cosmo, 0.0),
                BoxSize=32, Nmesh=4, seed=42, comm=comm)

    dlink = linear.to_field(mode='complex')
    lpt0 = Source.LPTParticles(dlink=dlink, cosmo=cosmo, redshift=0.0, order=2)

    def chi2(lpt):
#        print 'disp1', comm.allreduce(((lpt['LPTDisp1'] + lpt['LPTDisp2'])** 2).sum(dtype='f8').compute())
#        print 'diffpos', comm.allreduce(((lpt['Position'] - lpt['InitialPosition'])** 2).sum(dtype='f8').compute())
        return comm.allreduce(((lpt['Position'] - lpt['InitialPosition'])** 2).sum(dtype='f8').compute())

    def grad_chi2(lpt, dlink):
        lpt['GradPosition'] = 2 * (lpt['Position'] - lpt['InitialPosition'])
        #print 'grad pos', lpt['GradPosition'].compute()
        #print 'grad lpt1', lpt['GradLPTDisp1'].compute()
        #print 'grad lpt2', lpt['GradLPTDisp2'].compute()
        return Source.LPTParticles.gradient(dlink, lpt)

    chi2_0 = chi2(lpt0)
    grad_a = grad_chi2(lpt0, dlink)

    chi2_ndiff = []
    chi2_adiff = []
    i = 0
    for ind1 in numpy.ndindex(*(list(dlink.cshape) + [2])):
        i = i + 1
        if i % 4 != 0: continue
        dlinkl = dlink.copy()
        dlinkr = dlink.copy()
        old = dlink.cgetitem(ind1)
        diff = 1e-6
        dlinkl.csetitem(ind1, -diff + old)
        dlinkr.csetitem(ind1, diff + old)

        lptl = Source.LPTParticles(dlink=dlinkl, cosmo=cosmo, redshift=0.0, order=2)
        lptr = Source.LPTParticles(dlink=dlinkr, cosmo=cosmo, redshift=0.0, order=2)

        chi2_1 = chi2(lptl)
        chi2_2 = chi2(lptr)
        grad = grad_a.cgetitem(ind1)
#        print chi2_2, chi2_1, grad * diff * 2, diff
        chi2_ndiff.append(1 + chi2_2 - chi2_1)
        chi2_adiff.append(1 + grad * diff * 2)

    assert_allclose(chi2_ndiff, chi2_adiff, rtol=1e-4)

@MPITest([1, 4])
def test_lpt(comm):
    from pmesh.pm import ParticleMesh
    from nbodykit import fastpm

    pm = ParticleMesh(BoxSize=128.0, Nmesh=(4, 4), comm=comm)

    dlink = pm.generate_whitenoise(1234, mode='complex')

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
    print('------')
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
        print(ind1, yl, yr, grad * diff, yr - yl)
        ana.append(grad * diff)
        num.append(yr - yl)
    print('------')

    assert_allclose(num, ana, rtol=1e-5)

@MPITest([1])
def test_kick(comm):
    # No need to to thest drift because it is just an alias of kick.
    from nbodykit import fastpm
    p1 = numpy.ones((10, 2))
    f = numpy.ones((10, 2))

    def objective(p1, f):
        p2 = fastpm.kick(p1, f, 2.0)
        return (p2 **2).sum(dtype='f8')

    def gradient(p1, f):
        p2 = fastpm.kick(p1, f, 2.0)
        return fastpm.kick_gradient(2.0, grad_p2=2 * p2)

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

    dlink = pm.generate_whitenoise(12345, mode='complex')

    # FIXME: without the shift some particles have near zero dx1.
    # or near 1 dx1.
    # the gradient is not well approximated by the numerical if
    # any of the left or right value shifts beyond the support of
    # the window.
    #
    q = fastpm.create_grid(pm, shift=0.5)
    # ensure our numerical gradient shift doesn't hit non-concave regions of
    # the window.
    dx1 = fastpm.lpt1(dlink, q)
    dx1 = dx1.clip(-.5, .5)
    dx1 *= 0.5
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
        end = start + x1.shape[0]
        x1l = x1.copy()
        x1r = x1.copy()
        if ind1[0] >= start and ind1[0] < end:
            x1l[ind1[0] - start, ind1[1]] -= diff
            x1r[ind1[0] - start, ind1[1]] += diff
            grad = yprime[ind1[0] - start, ind1[1]]
            dx11 = dx1[ind1[0] - start, ind1[1]]
        else:
            grad = 0
            dx11 = 0
        grad = comm.allreduce(grad)
        dx11 = comm.allreduce(dx11)

        yl = objective(x1l, pm)
        yr = objective(x1r, pm)
        # Watchout : (yr - yl) / (yr + yl) must be large enough for numerical
        # to be accurate
        print(ind1, yl, yr, grad * diff * 2, yr - yl, yr - yl - grad *diff*2, dx11)
        num.append(yr - yl)
        ana.append(grad * 2 * diff)
    print('max difference is', numpy.abs(numpy.subtract(num, ana)).max())
    assert_allclose(num, ana, rtol=1e-5, atol=1e-7)

@MPITest([1, 4])
def test_vm(comm):
    from nbodykit import fastpm
    from nbodykit.cosmology import PerturbationGrowth
    from nbodykit.cosmology import Cosmology
    from pmesh.pm import ParticleMesh

    pt = PerturbationGrowth(Cosmology(Om0=0.3, Tcmb0=0))

    pm = ParticleMesh(BoxSize=128.0, Nmesh=(4,4,4), comm=comm, dtype='f8')
    vm = fastpm.Evolution(pm)
    dlink = pm.generate_whitenoise(12345, mode='complex', unitary=True)
    power = dlink.copy()

    def kernel(k, v):
        kk = sum(ki ** 2 for ki in k)
        ka = kk ** 0.5
        p = (ka / 0.1) ** -2 * .4e4 * (1.0 / pm.BoxSize).prod()
        p[ka == 0] = 0
        return p ** 0.5 + p ** 0.5 * 1j

    power.apply(kernel, out=Ellipsis)
    dlink[...] *= power.real

    q = fastpm.create_grid(pm, shift=0.5)

    data = dlink.c2r()
    data[...] = 0
    sigma = data.copy()
    sigma[...] = 1.0

    code = vm.kdk(pt, 0.1, 1.0, 5)
    code.Paint(pm=data.pm)
    code.Chi2(data_x=data, sigma_x=sigma)

    def objective(dlin_k):
        init = {'dlin_k': dlin_k, 'q' : q}
        return code.compute('chi2', init, monitor=None)

    def gradient(dlin_k):
        tape = vm.tape()
        init = {'dlin_k': dlin_k, 'q' : q}
        code.compute(['chi2', 'f', 's', 'p'], init, tape=tape, monitor=None)
        gcode = vm.gradient(tape)
        init = {'_chi2' : 1, '_q': 0, '_p' : 0, '_s' : 0, '_f' : 0}
        return gcode.compute('_dlin_k', init, monitor=None)

    y0 = objective(dlink)
    yprime = gradient(dlink)

    num = []
    ana = []
    print('------')
    for ind1 in numpy.ndindex(*(list(dlink.cshape) + [2])):
        dlinkl = dlink.copy()
        dlinkr = dlink.copy()
        old = dlink.cgetitem(ind1)
        pert = power.cgetitem(ind1) * 1e-5
        left = dlinkl.csetitem(ind1, old - pert)
        right = dlinkr.csetitem(ind1, old + pert)
        diff = right - left
        yl = objective(dlinkl)
        yr = objective(dlinkr)
        grad = yprime.cgetitem(ind1)
        print(ind1, old, pert, yl, yr, grad * diff, yr - yl)
        ana.append(grad * diff)
        num.append(yr - yl)
    print('------')

    assert_allclose(num, ana, rtol=1e-3)
