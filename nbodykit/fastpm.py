from __future__ import print_function
import numpy
import logging

def za_transfer(delta, dir, out=None):
    if out is None:
        out = delta.copy()

    for k, i, a, b in zip(delta.slabs.x,
                    delta.slabs.i, delta.slabs, out.slabs):
        kk = sum(ki ** 2 for ki in k)
        # set mask == False to 0
        mask = numpy.ones(a.shape, '?')

        for ii, n in zip(i, delta.Nmesh):
            # any nyquist modes are set to 0
            mask &=  ii != (n // 2)
        mask[kk == 0] = False
        kk[kk == 0] = 1
        b[...] = mask * a * 1j * k[dir] / kk
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
        za_transfer(delta_k, d, out=delta_k)
        disp = delta_k.c2r(delta_k)
        local_disp = disp.readout(local_q, method=method)
        source[..., d] = layout.gather(local_disp)
    return source

def lpt1_gradient(dlink, q, grad_disp, method='cic'):
    """ backtrace gradient of first order LPT on linear density field.
        returns gradient over modes of dlink. The qitions are assumed to
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
        grad_disp_d_k = grad_disp_d.c2r_gradient(grad_disp_d)

        # FIXME: allow changing this.
        # the force
        # -1 because 1j is conjugated
        grad_delta_d_k = za_transfer(grad_disp_d_k, d, out=grad_disp_d_k)
        grad_delta_d_k.value[...] *= -1

        grad.value[...] += grad_delta_d_k.value

    # dlink are free modes in the compressed real FFT representation,
    # so we need to take care of decompression

    grad.decompress_gradient(out=grad)

    return grad

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

    deltak = field.r2c(out=field)
    if f is None:
        f = numpy.empty_like(x)

    for d in range(field.ndim):
        force_k = za_transfer(deltak, d)
        force_k[...] *= factor
        force = force_k.c2r(out=force_k)
        force.readout(x, layout=layout, out=f[..., d])
    return f

def gravity_gradient(x, pm, factor, grad_f, out_x=None):
    if out_x is None:
        out_x = numpy.zeros_like(x)

    field = pm.create(mode="real")
    layout = pm.decompose(x)

    field.paint(x, layout=layout, hold=False)
    deltak = field.r2c(out=field)
    grad_deltak = pm.create(mode="complex")
    grad_deltak[...] = 0

    for d in range(x.shape[1]):
        force_k = za_transfer(deltak, d)
        force_k[...] *= factor
        force = force_k.c2r(out=force_k)
        grad_force_d, grad_x_d = force.readout_gradient(
            x, btgrad=grad_f[:, d], layout=layout)
        grad_force_d_k = grad_force_d.c2r_gradient(out=grad_force_d)
        grad_deltak_d = za_transfer(grad_force_d_k, d)
        grad_deltak_d[...] *= -1 * factor
        grad_deltak[...] += grad_deltak_d
        out_x[...] += grad_x_d

    grad_field = grad_deltak.r2c_gradient(out=grad_deltak)
    grad_x, grad_mass = grad_field.paint_gradient(x, layout=layout, out_mass=False)
    out_x[...] += grad_x

    return out_x

