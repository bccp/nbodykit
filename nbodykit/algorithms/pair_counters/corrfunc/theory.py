import numpy
from .base import MPICorrfuncCallable, MissingCorrfuncError

class CorrfuncTheoryCallable(MPICorrfuncCallable):
    """
    A MPI-enabled wrapper of a callable from :mod:`Corrfunc.theory`.

    Parameters
    ----------
    func : callable
        the Corrfunc function that will be called
    edges : list
        the list of arrays specifying the bin edges in each coordinate direction
    periodic : bool
        whether to use periodic boundary conditions
    BoxSize : array_like
        the size of the simulation box in each direction
    """
    binning_dims = None

    def __init__(self, func, edges, periodic, BoxSize, comm, show_progress=True):
        MPICorrfuncCallable.__init__(self, func, comm, show_progress=show_progress)
        self.edges = edges
        self.periodic = periodic

        # check periodic/BoxSize compatibiltity
        if periodic:
            if not numpy.all(BoxSize == BoxSize[0]):
                raise NotImplementedError("periodic wrapping with non-cubic boxes not implemented yet")
            else:
                self.BoxSize = BoxSize[0] # isotropic box
        else:
            self.BoxSize = None


    def __call__(self, pos1, w1, pos2, w2, **config):

        kws = {}
        kws['autocorr'] = 0
        kws['nthreads'] = 1
        kws['binfile'] = self.edges[0]
        kws['X2'] = pos2[:,0]
        kws['Y2'] = pos2[:,1]
        kws['Z2'] = pos2[:,2] # the LOS direction
        kws['weights2'] = w2.astype(pos2.dtype)
        kws['weight_type'] = 'pair_product'
        kws['output_%savg' %self.binning_dims[0]] = True
        kws['periodic'] = self.periodic
        if self.BoxSize is not None: kws['boxsize'] = self.BoxSize

        # add in the additional config keywords
        kws.update(config)

        def callback(kws, chunk):
            kws['X1'] = pos1[chunk][:,0].astype(pos2.dtype)
            kws['Y1'] = pos1[chunk][:,1].astype(pos2.dtype)
            kws['Z1'] = pos1[chunk][:,2].astype(pos2.dtype) # LOS defined with respect to this axis
            kws['weights1'] = w1[chunk].astype(pos2.dtype)

        # compute the result
        sizes = self.comm.allgather(len(pos1))
        return MPICorrfuncCallable.__call__(self, sizes, kws, callback=callback)

class DD(CorrfuncTheoryCallable):
    """
    A MPI-enabled wrapper of :func:`Corrfunc.theory.DD.DD`.
    """
    binning_dims = ['r']

    def __init__(self, edges, periodic, BoxSize, comm, show_progress=True):
        try:
            from Corrfunc.theory import DD
        except ImportError:
            raise MissingCorrfuncError()

        CorrfuncTheoryCallable.__init__(self, DD, [edges], periodic, BoxSize,
                                        comm,
                                        show_progress=show_progress)

class DDsmu(CorrfuncTheoryCallable):
    """
    A MPI-enabled wrapper of :func:`Corrfunc.theory.DDsmu.DDsmu`.
    """
    binning_dims = ['s', 'mu']

    def __init__(self, edges, Nmu, periodic, BoxSize, comm, show_progress=True):
        try:
            from Corrfunc.theory import DDsmu
        except ImportError:
            raise MissingCorrfuncError()

        self.Nmu = Nmu
        mu_edges = numpy.linspace(0., 1., Nmu+1)
        CorrfuncTheoryCallable.__init__(self, DDsmu, [edges, mu_edges],
                                        periodic, BoxSize, comm, show_progress=show_progress)

    def __call__(self, pos1, w1, pos2, w2, **config):
        config['nmu_bins'] = self.Nmu
        config['mu_max'] = 1.0
        return CorrfuncTheoryCallable.__call__(self, pos1, w1, pos2, w2, **config)

class DDrppi(CorrfuncTheoryCallable):
    """
    A MPI-enabled wrapper of :func:`Corrfunc.theory.DDrppi.DDrppi`.
    """
    binning_dims = ['rp', 'pi']

    def __init__(self, edges, pimax, periodic, BoxSize, comm, show_progress=True):
        try:
            from Corrfunc.theory import DDrppi
        except ImportError:
            raise MissingCorrfuncError()

        self.pimax = pimax
        pi_bins = numpy.linspace(0, pimax, pimax+1)
        CorrfuncTheoryCallable.__init__(self, DDrppi, [edges, pi_bins],
                                        periodic, BoxSize, comm, show_progress=show_progress)

    def __call__(self, pos1, w1, pos2, w2, **config):
        config['pimax'] = self.pimax
        return CorrfuncTheoryCallable.__call__(self, pos1, w1, pos2, w2, **config)
