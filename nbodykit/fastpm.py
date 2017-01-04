from __future__ import print_function
import numpy
import logging

def laplace_transfer(delta, out=None):
    if out is None:
        out = delta.copy()
    if out is Ellipsis:
        out = delta

    for k, i, a, b in zip(delta.slabs.x,
                    delta.slabs.i, delta.slabs, out.slabs):
        kk = sum(ki ** 2 for ki in k)
        # set mask == False to 0
        mask = kk == 0
        kk[mask] = 1
        b[...] = a / kk
        b[mask] = 0

    return out

def diff_transfer(delta, dir, out=None):
    if out is None:
        out = delta.copy()
    if out is Ellipsis:
        out = delta
    for k, i, a, b in zip(delta.slabs.x,
                    delta.slabs.i, delta.slabs, out.slabs):
        # set mask == False to 0
        mask = numpy.ones(a.shape, '?')

        for ii, n in zip(i, delta.Nmesh):
            # any nyquist modes are set to 0
            mask &=  ii != (n // 2)

        b[...] = mask * a * 1j * k[dir]
    return out


def create_grid(basepm, shift=0):
    """
        create uniform grid of particles, one per grid point on the basepm mesh

    """
    ndim = len(basepm.Nmesh)
    real = basepm.create('real')

    _shift = numpy.zeros(ndim, 'f8')
    _shift[:] = shift
    # one particle per base mesh point
    source = numpy.zeros((real.size, ndim), dtype='f4')

    for d in range(len(real.shape)):
        real[...] = 0
        for xi, slab in zip(real.slabs.i, real.slabs):
            slab[...] = (xi[d] + 1.0 * _shift[d]) * (real.BoxSize[d] / real.Nmesh[d])
        source[..., d] = real.value.flat
    return source

def lpt1(dlink, q, method='cic'):
    """ Run first order LPT on linear density field, returns displacements of particles
        reading out at q.
    """
    basepm = dlink.pm

    ndim = len(basepm.Nmesh)
    delta_k = basepm.create('complex')

    # only need to view the size
    delta_x = basepm.create('real', base=delta_k.base)

    layout = basepm.decompose(q)
    local_q = layout.exchange(q)

    source = numpy.zeros((delta_x.size, ndim), dtype='f4')
    for d in range(len(basepm.Nmesh)):
        delta_k[...] = dlink
        laplace_transfer(delta_k, out=Ellipsis)
        diff_transfer(delta_k, d, out=Ellipsis)
        disp = delta_k.c2r(out=Ellipsis)
        local_disp = disp.readout(local_q, method=method)
        source[..., d] = layout.gather(local_disp)
    return source

def lpt1_gradient(dlink, q, grad_disp, method='cic'):
    """ backtrace gradient of first order LPT on linear density field.
        returns gradient over modes of dlink. The positions are assumed to
        not to move, thus gradient over qition is not returned.

        The data partition of grad_disp must matchs the fastpm particle grid.
    """
    basepm = dlink.pm
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
        grad_disp_d_k = grad_disp_d.c2r_gradient(out=Ellipsis)

        # FIXME: allow changing this.
        # the force
        # -1 because 1j is conjugated
        grad_delta_d_k = laplace_transfer(grad_disp_d_k, out=Ellipsis)
        grad_delta_d_k = diff_transfer(grad_delta_d_k, d, out=Ellipsis)
        grad_delta_d_k.value[...] *= -1

        grad.value[...] += grad_delta_d_k.value

    # dlink are free modes in the compressed real FFT representation,
    # so we need to take care of decompression

    grad.decompress_gradient(out=Ellipsis)

    return grad

def lpt2source(dlink):
    """ Generate the second order LPT source term.  """
    source = dlink.pm.create('real')
    source[...] = 0
    D1 = [1, 2, 0]
    D2 = [2, 0, 1]

    phi_ii = []
    assert dlink.ndim == 3

    # diagnoal terms
    for d in range(dlink.ndim):
        phi_k = laplace_transfer(dlink)
        phi_i_k = diff_transfer(phi_k, d, out=Ellipsis)
        phi_ii_k = diff_transfer(phi_i_k, d, out=Ellipsis)
        phi_ii.append(phi_ii_k.c2r(out=Ellipsis))

    for d in range(3):
        source[...] += phi_ii[D1[d]].value * phi_ii[D2[d]].value

    # free memory
    phi_ii = []

    phi_ij = []
    # off-diag terms
    for d in range(dlink.ndim):
        phi_k = laplace_transfer(dlink)
        phi_i_k = diff_transfer(phi_k, D1[d], out=Ellipsis)
        phi_ij_k = diff_transfer(phi_i_k, D2[d], out=Ellipsis)
        phi_ij_d = phi_ij_k.c2r(out=Ellipsis)
        source[...] -= phi_ij_d[...] **2

    # this ensures x = x0 + dx1(t) + d2(t) for 2LPT

    source[...] *= 3.0 / 7
    return source.r2c(out=Ellipsis)

def lpt2source_gradient(dlink, grad_source):
    """ Generate the second order LPT source term.  """
    D1 = [1, 2, 0]
    D2 = [2, 0, 1]

    grad_dlink = dlink.copy()
    grad_dlink[...] = 0

    grad_source_x = grad_source.r2c_gradient()

    grad_source_x[...] *= 3.0 / 7

    # diagonal terms, forward
    phi_ii = []
    for d in range(3):
        phi_k = laplace_transfer(dlink)
        phi_i_k = diff_transfer(phi_k, d, out=Ellipsis)
        phi_ii_k = diff_transfer(phi_i_k, d, out=Ellipsis)
        phi_ii.append(phi_ii_k.c2r(out=Ellipsis))

    # diagonal terms, backward
    for d in range(3):
        # every component is used twice, with D1 and D2
        grad_phi_ii_d = grad_source_x.copy()
        grad_phi_ii_d[...] *= (phi_ii[D1[d]].value + phi_ii[D2[d]].value)
        grad_phi_ii_k_d = grad_phi_ii_d.c2r_gradient(out=Ellipsis)
        grad_phi_i_k_d = diff_transfer(grad_phi_ii_k_d, d, out=Ellipsis)
        grad_phi_k_d = diff_transfer(grad_phi_i_k_d, d, out=Ellipsis)
        grad_dlink[...] += laplace_transfer(grad_phi_k_d, out=Ellipsis)

    # off diagonal terms
    for d in range(3):
        # forward
        phi_k = laplace_transfer(dlink)
        phi_i_k = diff_transfer(phi_k, D1[d], out=Ellipsis)
        phi_ij_k = diff_transfer(phi_i_k, D2[d], out=Ellipsis)
        phi_ij_d = phi_ij_k.c2r(out=Ellipsis)

        # backward
        grad_phi_ij_d = phi_ij_d
        grad_phi_ij_d[...] *= -2 * grad_source_x[...]

        grad_phi_ij_k_d = grad_phi_ij_d.c2r_gradient(out=Ellipsis)
        grad_phi_i_k_d = diff_transfer(grad_phi_ij_k_d, D2[d], out=Ellipsis)
        grad_phi_k_d = diff_transfer(grad_phi_i_k_d, D1[d], out=Ellipsis)
        grad_dlink[...] += laplace_transfer(grad_phi_k_d, out=Ellipsis)

    return grad_dlink

def kick(p1, f, dt, p2=None):
    if p2 is None:
        p2 = numpy.empty_like(p1)

    p2[...] = p1 + f * dt
    return p2

def kick_gradient(p1, f, dt, grad_p2, out_p1=None, out_f=None):
    if out_f is None:
        out_f = numpy.empty_like(f)
    if out_p1 is None:
        out_p1 = numpy.empty_like(p1)

    out_f[...] = grad_p2 * dt
    out_p1[...] = grad_p2

    return out_p1, out_f

def drift(x1, p, dt, x2=None):
    return kick(x1, p, dt, x2)

def drift_gradient(x1, p, dt, grad_x2, out_x1=None, out_p=None):
    return kick_gradient(x1, p, dt, grad_x2, out_x1, out_p)

def gravity(x, pm, factor, f=None):
    field = pm.create(mode="real")
    layout = pm.decompose(x)
    field.paint(x, layout=layout, hold=False)

    deltak = field.r2c(out=Ellipsis)
    if f is None:
        f = numpy.empty_like(x)

    for d in range(field.ndim):
        force_k = laplace_transfer(deltak)
        force_k = diff_transfer(force_k, d, out=Ellipsis)
        force_k[...] *= factor
        force = force_k.c2r(out=Ellipsis)
        force.readout(x, layout=layout, out=f[..., d])
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

    for d in range(x.shape[1]):
        force_k = laplace_transfer(deltak)
        force_k = diff_transfer(force_k, d, out=Ellipsis)
        force_k[...] *= factor
        force = force_k.c2r(out=Ellipsis)
        grad_force_d, grad_x_d = force.readout_gradient(
            x, btgrad=grad_f[:, d], layout=layout)
        grad_force_d_k = grad_force_d.c2r_gradient(out=Ellipsis)
        grad_deltak_d = laplace_transfer(grad_force_d_k, out=Ellipsis)
        grad_deltak_d = diff_transfer(grad_deltak_d, d, out=Ellipsis)
        grad_deltak_d[...] *= -1 * factor
        grad_deltak[...] += grad_deltak_d
        out_x[...] += grad_x_d

    grad_field = grad_deltak.r2c_gradient(out=Ellipsis)
    grad_x, grad_mass = grad_field.paint_gradient(x, layout=layout, out_mass=False)
    out_x[...] += grad_x

    return out_x

