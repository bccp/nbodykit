from __future__ import print_function
import numpy
import logging

def za_transfer(delta, out, dir):
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

def lpt1(dlink, pos, method='cic'):
    """ Run first order LPT on linear density field, returns displacements of particles
        reading out at pos.
    """
    basepm = dlink.pm

    ndim = len(basepm.Nmesh)
    delta_k = basepm.create('complex')

    # only need to view the size
    delta_x = basepm.create('real', base=delta_k.base)

    layout = basepm.decompose(pos)
    local_pos = layout.exchange(pos)

    source = numpy.zeros((delta_x.size, ndim), dtype='f4')
    for d in range(len(basepm.Nmesh)):
        delta_k[...] = dlink
        za_transfer(delta_k, delta_k, d)
        disp = delta_k.c2r(delta_k)
        local_disp = disp.readout(local_pos, method=method)
        source[..., d] = layout.gather(local_disp)
    return source

def lpt1_gradient(dlink, pos, grad_disp, method='cic'):
    """ backtrace gradient of first order LPT on linear density field.
        returns gradient over modes of dlink. The positions are assumed to
        not to move, thus gradient over position is not returned.

        The data partition of grad_disp must matchs the fastpm particle grid.
    """
    basepm = dlink.pm
    ndim = len(basepm.Nmesh)

    layout = basepm.decompose(pos)
    local_pos = layout.exchange(pos)

    grad = basepm.create('complex')
    grad[...] = 0
    grad_disp_d = basepm.create('real')

    # for each dimension
    for d in range(ndim):
        local_grad_disp_d = layout.exchange(grad_disp[:, d])
        grad_disp_d.readout_gradient(local_pos, local_grad_disp_d, method=method, out_self=grad_disp_d, out_pos=False)
        grad_disp_d_k = grad_disp_d.c2r_gradient(grad_disp_d)

        # FIXME: allow changing this.
        # the force
        za_transfer(grad_disp_d_k, grad_disp_d_k, d)
        grad_disp_d_k[...] *= -1 # because 1j is conjugated

        grad.value[...] += grad_disp_d_k.value

    # dlink are free modes in the compressed real FFT representation,
    # so we need to take care of decompression

    grad.decompress_gradient(out=grad)

    return grad
