from __future__ import print_function
import numpy
import logging
from abopt.vmad import VM

def laplace_kernel(k, v):
    kk = sum(ki ** 2 for ki in k)
    mask = (kk == 0).nonzero()
    kk[mask] = 1
    b = v / kk
    b[mask] = 0
    return b

def diff_kernel(dir, conjugate=False):
    def kernel(k, v):
        mask = numpy.ones(v.shape, '?')
        if conjugate:
            factor = -1j
        else:
            factor = 1j

        for ii, ni in zip(v.i, v.Nmesh):
            # any nyquist modes are set to 0 (False)
            mask &=  ii != (ni // 2)

        return (mask * v) * (factor * k[dir])
    return kernel

def create_grid(basepm, shift=0, dtype='f4'):
    """
        create uniform grid of particles, one per grid point on the basepm mesh

    """
    ndim = len(basepm.Nmesh)
    real = basepm.create('real')

    _shift = numpy.zeros(ndim, 'f8')
    _shift[:] = shift
    # one particle per base mesh point
    source = numpy.zeros((real.size, ndim), dtype=dtype)

    for d in range(len(real.shape)):
        real[...] = 0
        for xi, slab in zip(real.slabs.i, real.slabs):
            slab[...] = (xi[d] + 1.0 * _shift[d]) * (real.BoxSize[d] / real.Nmesh[d])
        source[..., d] = real.value.flat
    return source

def lpt1(dlin_k, q, method='cic'):
    """ Run first order LPT on linear density field, returns displacements of particles
        reading out at q. The result has the same dtype as q.
    """
    basepm = dlin_k.pm

    ndim = len(basepm.Nmesh)
    delta_k = basepm.create('complex')

    # only need to view the size
    delta_x = basepm.create('real', base=delta_k.base)

    layout = basepm.decompose(q)
    local_q = layout.exchange(q)

    source = numpy.zeros((delta_x.size, ndim), dtype=q.dtype)
    for d in range(len(basepm.Nmesh)):
        disp = dlin_k.apply(laplace_kernel) \
                    .apply(diff_kernel(d), out=Ellipsis) \
                    .c2r(out=Ellipsis)
        local_disp = disp.readout(local_q, method=method)
        source[..., d] = layout.gather(local_disp)
    return source

def lpt1_gradient(dlin_k, q, grad_disp, method='cic'):
    """ backtrace gradient of first order LPT on linear density field.
        returns gradient over modes of dlin_k. The positions are assumed to
        not to move, thus gradient over qition is not returned.

        The data partition of grad_disp must matchs the fastpm particle grid.
    """
    basepm = dlin_k.pm
    ndim = len(basepm.Nmesh)

    layout = basepm.decompose(q)
    local_q = layout.exchange(q)

    grad = basepm.create('complex')
    grad[...] = 0
    grad_disp_d = basepm.create('real')

    # for each dimension
    for d in range(ndim):
        local_grad_disp_d = layout.exchange(grad_disp[:, d])
        grad_disp_d.readout_gradient(local_q, local_grad_disp_d, method=method, out_self=grad_disp_d, out_pos=False)
        grad_delta_d_k = grad_disp_d.c2r_gradient(out=Ellipsis) \
                         .apply(laplace_kernel, out=Ellipsis) \
                         .apply(diff_kernel(d, conjugate=True), out=Ellipsis)

        grad.value[...] += grad_delta_d_k.value

    # dlin_k are free modes in the compressed real FFT representation,
    # so we need to take care of decompression

    grad.decompress_gradient(out=Ellipsis)

    return grad

def lpt2source(dlin_k):
    """ Generate the second order LPT source term.  """
    source = dlin_k.pm.create('real')
    source[...] = 0
    D1 = [1, 2, 0]
    D2 = [2, 0, 1]

    phi_ii = []
    if dlin_k.ndim != 3:
        return source.r2c(out=Ellipsis)

    # diagnoal terms
    for d in range(dlin_k.ndim):
        phi_ii_d = dlin_k.apply(laplace_kernel) \
                     .apply(diff_kernel(d), out=Ellipsis) \
                     .apply(diff_kernel(d), out=Ellipsis) \
                     .c2r(out=Ellipsis)
        phi_ii.append(phi_ii_d)

    for d in range(3):
        source[...] += phi_ii[D1[d]].value * phi_ii[D2[d]].value

    # free memory
    phi_ii = []

    phi_ij = []
    # off-diag terms
    for d in range(dlin_k.ndim):
        phi_ij_d = dlin_k.apply(laplace_kernel) \
                 .apply(diff_kernel(D1[d]), out=Ellipsis) \
                 .apply(diff_kernel(D2[d]), out=Ellipsis) \
                 .c2r(out=Ellipsis)

        source[...] -= phi_ij_d[...] ** 2

    # this ensures x = x0 + dx1(t) + d2(t) for 2LPT

    source[...] *= 3.0 / 7
    return source.r2c(out=Ellipsis)

def lpt2source_gradient(dlin_k, grad_source):
    """ Generate the second order LPT source term.  """
    D1 = [1, 2, 0]
    D2 = [2, 0, 1]

    grad_dlin_k = dlin_k.copy()
    grad_dlin_k[...] = 0

    if dlin_k.ndim != 3:
        return grad_dlin_k

    grad_source_x = grad_source.r2c_gradient()

    grad_source_x[...] *= 3.0 / 7

    # diagonal terms, forward
    phi_ii = []
    for d in range(3):
        phi_ii_d = dlin_k.apply(laplace_kernel) \
                     .apply(diff_kernel(d), out=Ellipsis) \
                     .apply(diff_kernel(d), out=Ellipsis) \
                     .c2r(out=Ellipsis)
        phi_ii.append(phi_ii_d)

    # diagonal terms, backward
    for d in range(3):
        # every component is used twice, with D1 and D2
        grad_phi_ii_d = grad_source_x.copy()
        grad_phi_ii_d[...] *= (phi_ii[D1[d]].value + phi_ii[D2[d]].value)
        grad_dlin_k_d = grad_phi_ii_d.c2r_gradient(out=Ellipsis) \
                         .apply(diff_kernel(d, conjugate=True), out=Ellipsis) \
                         .apply(diff_kernel(d, conjugate=True), out=Ellipsis) \
                         .apply(laplace_kernel, out=Ellipsis)

        grad_dlin_k[...] += grad_dlin_k_d

    # off diagonal terms
    for d in range(3):
        # forward
        phi_ij_d = dlin_k.apply(laplace_kernel) \
                 .apply(diff_kernel(D1[d]), out=Ellipsis) \
                 .apply(diff_kernel(D2[d]), out=Ellipsis) \
                 .c2r(out=Ellipsis)

        # backward
        grad_phi_ij_d = phi_ij_d
        grad_phi_ij_d[...] *= -2 * grad_source_x[...]
        grad_dlin_k_d = grad_phi_ij_d.c2r_gradient(out=Ellipsis) \
                    .apply(diff_kernel(D2[d], conjugate=True), out=Ellipsis) \
                    .apply(diff_kernel(D1[d], conjugate=True), out=Ellipsis) \
                    .apply(laplace_kernel, out=Ellipsis)
        grad_dlin_k[...] += grad_dlin_k_d

    return grad_dlin_k

def kick(p1, f, dt, p2=None):
    if p2 is None:
        p2 = numpy.empty_like(p1)
    if p2 is Ellipsis:
        p2 = p1
    p2[...] = p1 + f * dt
    return p2

def kick_gradient(dt, grad_p2, out_p1=None, out_f=None):
    if out_f is None:
        out_f = numpy.empty_like(grad_p2)
    if out_p1 is None:
        out_p1 = numpy.empty_like(grad_p2)

    out_f[...] = grad_p2 * dt
    out_p1[...] = grad_p2

    return out_p1, out_f

def drift(x1, p, dt, x2=None):
    return kick(x1, p, dt, x2)

def drift_gradient(dt, grad_x2, out_x1=None, out_p=None):
    return kick_gradient(dt, grad_x2, out_x1, out_p)

def gravity(x, pm, factor, f=None):
    field = pm.create(mode="real")
    layout = pm.decompose(x)
    field.paint(x, layout=layout, hold=False)

    deltak = field.r2c(out=Ellipsis)
    if f is None:
        f = numpy.empty_like(x)

    for d in range(field.ndim):
        force_d = deltak.apply(laplace_kernel) \
                  .apply(diff_kernel(d), out=Ellipsis) \
                  .c2r(out=Ellipsis)
        force_d.readout(x, layout=layout, out=f[..., d])
    f[...] *= factor
    return f

def gravity_gradient(x, pm, factor, grad_f, out_x=None):
    if out_x is None:
        out_x = numpy.zeros_like(x)

    field = pm.create(mode="real")
    layout = pm.decompose(x)

    field.paint(x, layout=layout, hold=False)
    deltak = field.r2c(out=Ellipsis)
    grad_deltak = pm.create(mode="complex")
    grad_deltak[...] = 0

    for d in range(field.ndim):
        # forward
        force_d = deltak.apply(laplace_kernel) \
                  .apply(diff_kernel(d), out=Ellipsis) \
                  .c2r(out=Ellipsis)

        grad_force_d, grad_x_d = force_d.readout_gradient(
            x, btgrad=grad_f[:, d], layout=layout)

        grad_deltak_d = grad_force_d.c2r_gradient(out=Ellipsis) \
                        .apply(laplace_kernel, out=Ellipsis) \
                        .apply(diff_kernel(d, conjugate=True), out=Ellipsis)
        grad_deltak[...] += grad_deltak_d
        out_x[...] += grad_x_d

    grad_field = grad_deltak.r2c_gradient(out=Ellipsis)
    grad_x, grad_mass = grad_field.paint_gradient(x, layout=layout, out_mass=False)
    out_x[...] += grad_x

    # should have been first applied to grad_f, but it is the same applying it here
    # and saves some memory
    out_x[...] *= factor

    return out_x

class Evolution(VM):
    def __init__(self, pm):
        self.pm = pm

        VM.__init__(self)

    def lpt(self, pt, aend, order):
        code = self.code()
        if order == 1:
            code.Displace(D1=pt.D1(aend), 
                          v1=pt.f1(aend) * pt.D1(aend) * aend ** 2 * pt.E(aend),
                          D2=0,
                          v2=0,
                         )
        if order == 2:
            code.Displace(D1=pt.D1(aend), 
                          v1=pt.f1(aend) * pt.D1(aend) * aend ** 2 * pt.E(aend),
                          D2=pt.D2(aend),
                          v2=pt.f2(aend) * pt.D2(aend) * aend ** 2 * pt.E(aend),
                         )
        return code

    def kdk(self, pt, astart, aend, Nsteps):
        code = self.code()
        code.Displace(D1=pt.D1(astart), 
                      v1=pt.f1(astart) * pt.D1(astart) * astart ** 2 * pt.E(astart),
                      D2=pt.D2(astart),
                      v2=pt.f2(astart) * pt.D2(astart) * astart ** 2 * pt.E(astart),
                     )
        code.Force(factor=1.5 * pt.Om0)

        a = numpy.linspace(astart, aend, Nsteps + 1, endpoint=True)
        def K(ai, af, ar):
            return 1 / (ar ** 2 * pt.E(ar)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ar)
        def D(ai, af, ar):
            return 1 / (ar ** 3 * pt.E(ar)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ar)

        for ai, af in zip(a[:-1], a[1:]):
            ac = (ai * af) ** 0.5

            code.Kick(dda=K(ai, ac, ai))
            code.Drift(dyyy=D(ai, ac, ac))
            code.Drift(dyyy=D(ac, af, ac))
            code.Force(factor=1.5 * pt.Om0)
            code.Kick(dda=K(ac, af, af))

        return code

    @VM.microcode(aout=['b'], ain=['a'])
    def copy(self, a):
        if hasattr(a, 'copy'):
            return a.copy()
        else:
            return 1.0 * a

    @VM.microcode(aout=['s', 'p'], ain=['dlin_k'], literals=['q'])
    def Displace(self, dlin_k, D1, v1, D2, v2):
        q = create_grid(dlin_k.pm, shift=0.5, dtype=dlin_k.real.dtype)
        dx1 = lpt1(dlin_k, q)
        source = lpt2source(dlin_k)
        dx2 = lpt1(source, q)
        s = D1 * dx1 + D2 * dx2
        p = v1 * dx1 + v2 * dx2
        return s, p, q

    @Displace.grad
    def GradientDisplace(self, dlin_k, _s, _p, D1, v1, D2, v2):
        grad_dx1 = _p * (v1) + _s * D1
        grad_dx2 = _p * (v2) + _s * D2
        q = create_grid(dlin_k.pm, shift=0.5, dtype=dlin_k.real.dtype)

        if grad_dx1 is not VM.Zero:
            gradient = lpt1_gradient(dlin_k, q, grad_dx1)
        else:
            gradient = VM.Zero
        # because the exact value of lpt2source is irrelevant, we save some computation
        # by not using lpt2source = fastpm.lpt2source(self.dlink)
        if grad_dx2 is not VM.Zero:
            gradient_lpt2source = lpt1_gradient(dlin_k, q, grad_dx2)
            gradient[...] += lpt2source_gradient(dlin_k, gradient_lpt2source)

        return gradient

    @VM.microcode(aout=['meshforce'], ain=['mesh'])
    def MeshForce(self, mesh, d, factor):
        deltak = field.r2c(out=Ellipsis)
        f = deltak.apply(laplace_kernel) \
                  .apply(diff_kernel(d), out=Ellipsis) \
                  .c2r(out=Ellipsis)
        f[...] *= factor
        return f

    @MeshForce.grad
    def gMeshForce(self, _meshforce, d, factor):
        _mesh = _meshforce.c2r_gradient()\
                           .apply(laplace_kernel, out=Ellipsis) \
                           .apply(diff_kernel(d), out=Ellipsis) \
                           .r2c_gradient(out=Ellipsis)
        _mesh[...] *= factor
        return _mesh

    @VM.microcode(ain=['f', 'meshforce', 's'], literals=['q'], aout=['f'])
    def Readout(self, s, q, meshforce, d, f):
        density_factor = self.pm.Nmesh.prod() / self.pm.comm.allreduce(len(q))
        if f is VM.Zero:
            f = numpy.empty_like(q)
        x = s + q
        layout = pm.decompose(x)
        meshforce.readout(x, layout=layout, out=f[:, d])
        return f
        
    @VM.microcode(aout=['f'], ain=['s'], literals=['q'])
    def Force(self, s, q, factor):
        density_factor = self.pm.Nmesh.prod() / self.pm.comm.allreduce(len(q))
        x = s + q
        return gravity(x, self.pm, factor=density_factor * factor, f=None)

        
    @Force.grad
    def GradientForce(self, s, _f, q, factor):
        if _f is VM.Zero:
            return VM.Zero
        else:
            density_factor = self.pm.Nmesh.prod() / self.pm.comm.allreduce(len(q))
            x = s + q
            return gravity_gradient(x, self.pm, density_factor * factor, _f)


    @VM.microcode(aout=['mesh'], ain=['s'], literals=['q'])
    def Paint(self, s, q, pm):
        x = s + q
        real = pm.create(mode='real')
        layout = pm.decompose(x)
        real.paint(x, layout=layout, hold=False)
        return real

    @Paint.grad
    def GradientPaint(self, _mesh, s, q, pm):
        if _mesh is VM.Zero:
            return VM.Zero
        else:
            x = s + q
            layout = _mesh.pm.decompose(x)
            _s, junk = _mesh.paint_gradient(x, layout=layout, out_mass=False)
            return _s

    @VM.microcode(aout=['chi2'], ain=['mesh'])
    def Chi2(self, mesh, data_x, sigma_x):
        diff = mesh + -1 * data_x
        diff[...] **= 2
        diff[...] /= sigma_x[...]
        return diff.csum()

    @Chi2.grad
    def gchi2(self, _chi2, mesh, data_x, sigma_x):
        diff = mesh + -1 * data_x
        diff[...] *= 2
        diff[...] /= sigma_x[...]
        return diff

    @VM.microcode(aout=['p'], ain=['f', 'p'])
    def Kick(self, f, p, dda):
        return kick(p, f, dda, p2=Ellipsis)

    @Kick.grad
    def GradientKick(self, _p, dda):
        if _p is VM.Zero:
            return VM.Zero, VM.Zero
        else:
            _p, _f = kick_gradient(dda, _p)
            return _f, _p

    @VM.microcode(aout=['s'], ain=['p', 's'])
    def Drift(self, p, s, dyyy):
        return drift(s, p, dyyy, x2=Ellipsis)

    @Drift.grad
    def GradientDrift(self, _s, dyyy):
        if _s is VM.Zero:
            return VM.Zero, VM.Zero
        else:
            _s, _p = drift_gradient(dyyy, _s)
            return _p, _s


