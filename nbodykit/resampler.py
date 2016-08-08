import numpy
from pmesh.particlemesh import ParticleMesh
import mpsort

def read(pm, ds, Nmesh, isfourier):
    """ Read from ds to pm.

        Parameters
        ----------
            pm : ParticleMesh

            ds : array_like, [start:end] for reading. C ordering.

            Nmesh : int
                Nmesh of ds

            isfourier : bool
                is ds fourier or real?

        Returns
        -------
            pm is updated to contain the real field.
            If Nmesh does not match pm, a Fourier resampling is applied.
    """

    if Nmesh != pm.Nmesh:
        pmsrc = ParticleMesh(pm.BoxSize, Nmesh, dtype='f4', comm=pm.comm)

        directread(pmsrc, ds, Nmesh, isfouier)
        pmsrc.r2c()

        if Nmesh > pm.Nmesh:
            downsample(pmsrc, pm)
        else:
            upsample(pmsrc, pm)
    else:
        directread(pm, ds, Nmesh, isfourier)

def write(pm, Nmesh, isfourier):
    """
        prepare an output into array.

        Parameters
        ----------
        pm : ParticleMesh.
            pm.real contains the density field to be written.

        isfourier:
            transform to Fourier space before writing?

        Nmesh : int
            resolution of the output

        Returns
        -------
        array : array_like
            where to store the data. We don't actually write to files. The array
            object filled shall be written to files by the caller.

    """
    if Nmesh != pm.Nmesh:
        pmdest = ParticleMesh(pm.BoxSize, Nmesh, dtype='f4', comm=pm.comm)

        if Nmesh > pm.Nmesh:
            downsample(pm, pmdest)
        else:
            upsample(pm, pmdest)

        return directwrite(pmdest, Nmesh, isfourier)
    else:
        return directwrite(pm, Nmesh, isfourier)

# Implementation details
#

def directread(pm, ds, Nmesh, isfourier):
    assert Nmesh == pm.Nmesh

    if isfourier:
        ind = build_index(
                [ numpy.arange(s, s + n)
                  for s, n in zip(pm.partition.local_o_start,
                                pm.complex.shape)
                ], [Nmesh, Nmesh, Nmesh // 2 + 1])
        start, end = mpsort.globalrange(ind.flat, pm.comm)

        data = ds[start:end]
        mpsort.permute(data, ind.flat, pm.comm, out=pm.complex.flat)
        # regardless, always ensure PM holds a real field.
        pm.c2r()
    else:
        ind = build_index(
                [ numpy.arange(s, s + n)
                  for s, n in zip(pm.partition.local_i_start,
                                pm.real.shape)
                ], [Nmesh, Nmesh, Nmesh])

        start, end = mpsort.globalrange(ind.flat, pm.comm)

        data = ds[start:end]
        mpsort.permute(data, ind.flat, pm.comm, out=pm.real.flat)

def directwrite(pm, Nmesh, isfourier):
    assert Nmesh == pm.Nmesh

    if isfourier:
        pm.r2c()
        ind = build_index(
                [ numpy.arange(s, s + n)
                  for s, n in zip(pm.partition.local_o_start,
                                pm.complex.shape)
                ], [Nmesh, Nmesh, Nmesh // 2 + 1])
        start, end = mpsort.globalrange(ind.flat, pm.comm)
        array = numpy.empty(end - start, dtype='complex64')
        mpsort.sort(pm.complex.flat, ind.flat, pm.comm, out=array.flat)
    else:
        ind = build_index(
                [ numpy.arange(s, s + n)
                  for s, n in zip(pm.partition.local_i_start,
                                pm.real.shape)
                ], [Nmesh, Nmesh, Nmesh])
        start, end = mpsort.globalrange(ind.flat, pm.comm)
        array = numpy.empty(end - start, dtype='float32')
        mpsort.sort(pm.real.flat, ind.flat, comm=pm.comm, out=array.flat)
    return array

def upsample(pmsrc, pmdest):
    """ upsample.

        This can be done more efficiently, but it is a good starting point.

        FIXME: We shall directly calculate the mapping between the unsorted pmsrc index
        and pmdest, to save a bunch of global sortings.

    """
    assert pmdest.Nmesh >= pmsrc.Nmesh

    ind = build_index(
            [ numpy.arange(s, s + n)
              for s, n in zip(pmsrc.partition.local_o_start,
                            pmsrc.complex.shape)
            ], [pmsrc.Nmesh, pmsrc.Nmesh, pmsrc.Nmesh // 2 + 1])

    mpsort.sort(pmsrc.complex.flat, orderby=ind.flat, comm=pmsrc.comm)

    # indtable stores the index in pmsrc for the mode in pmdest
    # since pmdest > pmsrc, some items are -1
    indtable = reindex(pmsrc.Nmesh, pmdest.Nmesh)

    ind = build_index(
            [ indtable[numpy.arange(s, s + n)]
              for s, n in zip(pmdest.partition.local_o_start,
                            pmdest.complex.shape)
            ], [pmsrc.Nmesh, pmsrc.Nmesh, pmsrc.Nmesh // 2 + 1])

    pmdest.complex[:] = 0

    # fill the points that has values in pmsrc
    mask = ind >= 0
    # their indices
    argind = ind[mask]
    # take the data
    data = mpsort.take(pmsrc.complex.flat, argind, pmsrc.comm)
    # fill in the value
    pmdest.complex[mask] = data

def downsample(pmsrc, pmdest):
    assert pmdest.Nmesh <= pmsrc.Nmesh
    # indtable stores the index in pmsrc for the mode in pmdest
    # since pmdest < pmsrc, all items are alright.
    indtable = reindex(pmsrc.Nmesh, pmdest.Nmesh)

    ind = build_index(
            [ indtable[numpy.arange(s, s + n)]
              for s, n in zip(pmdest.partition.local_o_start,
                            pmdest.complex.shape)
            ], [pmsrc.Nmesh, pmsrc.Nmesh, pmsrc.Nmesh // 2 + 1])

    ind = ind.ravel()
    mpsort.take(pmsrc.complex.flat, ind, pmsrc.comm, out=pmdest.complex.flat)

def build_index(indices, fullshape):
    """
        Build a linear index array based on indices on an array of fullshape.
        This is similar to numpy.ravel_multi_index.

        index value of -1 will on any axes will be translated to -1 in the final.

        Parameters:
            indices : a tuple of index per dimension.

            fullshape : a tuple of the shape of the full array

        Returns:
            ind : a 3-d array of the indices of the coordinates in indices in
                an array of size fullshape. -1 if any indices is -1.

    """
    localshape = [ len(i) for i in indices]
    ndim = len(localshape)
    ind = numpy.zeros(localshape, dtype='i8')
    for d in range(len(indices)):
        i = indices[d]
        i = i.reshape([-1 if dd == d else 1 for dd in range(ndim)])
        ind[...] *= fullshape[d]
        ind[...] += i

    mask = numpy.zeros(localshape, dtype='?')

    # now mask out bad points by -1
    for d in range(len(indices)):
        i = indices[d]
        i = i.reshape([-1 if dd == d else 1 for dd in range(ndim)])
        mask |= i == -1

    ind[mask] = -1
    return ind

def reindex(Nsrc, Ndest):
    """ returns the index in the frequency array for corresponding
        k in Nsrc and composes Ndest

        For those Ndest that doesn't exist in Nsrc, return -1

        Example:
        >>> reindex(8, 4)
        >>> array([0, 1, 2, 7])
        >>> reindex(4, 8)
        >>> array([ 0,  1,  2, -1, -1, -1,  -1,  3])

    """
    reindex = numpy.arange(Ndest)
    reindex[Ndest // 2 + 1:] = numpy.arange(Nsrc - Ndest // 2 + 1, Nsrc, 1)
    reindex[Nsrc // 2 + 1: Ndest -Nsrc //2 + 1] = -1
    return reindex
