import numpy
from .base import MPICorrfuncCallable, MissingCorrfuncError

class CorrfuncMocksCallable(MPICorrfuncCallable):
    """
    A MPI-enabled wrapper of a callable from :mod:`Corrfunc.mocks`.

    Parameters
    ----------
    func : callable
        the Corrfunc function that will be called
    edges : list
        the list of arrays specifying the bin edges in each coordinate direction
    """
    binning_dims = None

    def __init__(self, func, edges, comm, show_progress=True):

        MPICorrfuncCallable.__init__(self, func, comm, show_progress=show_progress)
        self.edges = edges

    def __call__(self, pos1, w1, pos2, w2, **config):

        kws = {}
        kws['autocorr'] = 0
        kws['nthreads'] = 1
        kws['binfile'] = self.edges[0]
        kws['RA2'] = pos2[:,0]
        kws['DEC2'] = pos2[:,1]
        kws['weights2'] = w2.astype(pos2.dtype)
        kws['weight_type'] = 'pair_product'
        kws['output_%savg' %self.binning_dims[0]] = True

        # add in the comoving distances if we have them
        threedims = pos1.shape[1] == 3
        if threedims:
            kws['CZ2'] = pos2[:,2]
            kws['cosmology'] = 1
            kws['is_comoving_dist'] = True

        # add in the additional config keywords
        kws.update(config)

        def callback(kws, chunk):
            kws['RA1'] = pos1[chunk][:,0].astype(pos2.dtype)
            kws['DEC1'] = pos1[chunk][:,1].astype(pos2.dtype)
            if threedims: kws['CZ1'] = pos1[chunk][:,2].astype(pos2.dtype)
            kws['weights1'] = w1[chunk].astype(pos2.dtype)

        # compute the result
        sizes = self.comm.allgather(len(pos1))
        return MPICorrfuncCallable.__call__(self, sizes, kws, callback=callback)


class DDsmu_mocks(CorrfuncMocksCallable):
    """
    A MPI-enabled wrapper of :func:`Corrfunc.mocks.DDsmu_mocks.DDsmu_mocks`.
    """
    binning_dims = ['s', 'mu']

    def __init__(self, edges, Nmu, comm, show_progress=True):
        try:
            from Corrfunc.mocks import DDsmu_mocks
        except ImportError:
            raise MissingCorrfuncError()

        self.Nmu = Nmu
        mu_edges = numpy.linspace(0., 1., Nmu+1)
        CorrfuncMocksCallable.__init__(self, DDsmu_mocks, [edges, mu_edges],
                                        comm, 
                                        show_progress=show_progress)

    def __call__(self, pos1, w1, pos2, w2, **config):
        config['nmu_bins'] = int(self.Nmu)
        config['mu_max'] = 1.0
        return CorrfuncMocksCallable.__call__(self, pos1, w1, pos2, w2, **config)

class DDtheta_mocks(CorrfuncMocksCallable):
    """
    A MPI-enabled wrapper of :func:`Corrfunc.mocks.DDtheta_mocks.DDtheta_mocks`.
    """
    binning_dims = ['theta']

    def __init__(self, edges, comm, show_progress=True):
        try:
            from Corrfunc.mocks import DDtheta_mocks
        except ImportError:
            raise MissingCorrfuncError()

        CorrfuncMocksCallable.__init__(self, DDtheta_mocks, [edges],
                                        comm,
                                        show_progress=show_progress)

class DDrppi_mocks(CorrfuncMocksCallable):
    """
    A MPI-enabled wrapper of :func:`Corrfunc.mocks.DDrppi_mocks.DDrppi_mocks`.
    """
    binning_dims = ['rp', 'pi']

    def __init__(self, edges, pimax, comm, show_progress=True):
        try:
            from Corrfunc.mocks import DDrppi_mocks
        except ImportError:
            raise MissingCorrfuncError()

        self.pimax = pimax
        pi_bins = numpy.linspace(0, pimax, int(pimax)+1)
        CorrfuncMocksCallable.__init__(self, DDrppi_mocks, [edges, pi_bins],
                                        comm,
                                        show_progress=show_progress)

    def __call__(self, pos1, w1, pos2, w2, **config):
        config['pimax'] = float(self.pimax)
        return CorrfuncMocksCallable.__call__(self, pos1, w1, pos2, w2, **config)
