import numpy
import logging

import kdcount
from nbodykit import CurrentMPIComm
from nbodykit.dataset import DataSet

class Multipoles3PCF(object):
    """
    Compute the multipoles of the three-point correlation function

    References
    ----------
    Slepian and Eisenstein, MNRAS 454, 4142–4158 (2015)
    """
    logger = logging.getLogger('Multipoles3PCF')

    def __init__(self, source, poles, edges, BoxSize=None, periodic=True, weight='Weight'):
        """
        Parameters
        ----------
        source : CatalogSource
            the input source of particles providing the 'Position' column
        poles : list of int
            the list of multipole numbers to compute
        edges : array_like
            the radius bin edges; length of nbins+1
        BoxSize : float, 3-vector; optional
            the size of the box; if periodic boundary conditions used, and 'BoxSize'
            not provided in the source 'attrs', it must be provided here
        periodic : bool; optional
            whether to use periodic boundary conditions
        weight : str; optional
            the name of the column in the source specifying the particle weights
        """
        if 'Position' not in source:
            raise ValueError("the 'Position' column must be defined in the source")

        self.source = source
        self.comm = source.comm
        self.attrs = {}

        # need BoxSize
        self.attrs['BoxSize'] = numpy.empty(3)
        BoxSize = source.attrs.get('BoxSize', BoxSize)
        if periodic and BoxSize is None:
            raise ValueError("please specify a BoxSize if using periodic boundary conditions")
        self.attrs['BoxSize'][:] = BoxSize

        # test rmax for PBC
        if periodic and numpy.amax(edges) > 0.5*self.attrs['BoxSize'].min():
            raise ValueError("periodic pair counts cannot be computed for Rmax > 0.5 * BoxSize")

        # save meta-data
        self.attrs['edges']    = edges
        self.attrs['poles']    = poles
        self.attrs['periodic'] = periodic
        self.attrs['weight']   = weight

        self.run()

    def run(self):
        """
        Compute the three-point CF multipoles. This attaches the following
        the attributes to the class:

        Attributes
        ----------
        poles : :class:`~nbodykit.dataset.DataSet` or ``None``
            a DataSet object to hold the multipole results
        """
        from pmesh.domain import GridND

        redges = self.attrs['edges']
        comm   = self.comm
        nbins  = len(redges)-1
        Nell   = len(self.attrs['poles'])

        if self.attrs['periodic']:
            boxsize = self.attrs['BoxSize']
        else:
            boxsize = None

        # determine processor division for domain decomposition
        for Nx in range(int(comm.size**0.3333) + 1, 0, -1):
            if comm.size % Nx == 0: break
        else:
            Nx = 1
        for Ny in range(int(comm.size**0.5) + 1, 0, -1):
            if (comm.size // Nx) % Ny == 0: break
        else:
            Ny = 1
        Nz = comm.size // Nx // Ny
        Nproc = [Nx, Ny, Nz]
        if self.comm.rank == 0:
            self.logger.info("using cpu grid decomposition: %s" %str(Nproc))

        # output zeta
        zeta = numpy.zeros((Nell,nbins,nbins), dtype='f8')

        # compute the Ylm expressions we need
        if self.comm.rank == 0:
            self.logger.info("computing Ylm expressions...")
        Ylm_cache = YlmCache(self.attrs['poles'], comm)
        if self.comm.rank ==  0:
            self.logger.info("...done")

        # get the (periodic-enforced) position
        pos = self.source['Position']
        if self.attrs['periodic']:
            pos %= self.attrs['BoxSize']
        pos, w = self.source.compute(pos, self.source[self.attrs['weight']])

        # global min/max across all ranks
        posmin = numpy.asarray(comm.allgather(pos.min(axis=0))).min(axis=0)
        posmax = numpy.asarray(comm.allgather(pos.max(axis=0))).max(axis=0)

        # domain decomposition
        grid = [numpy.linspace(posmin[i], posmax[i], Nproc[i]+1, endpoint=True) for i in range(3)]
        domain = GridND(grid, comm=comm)

        layout = domain.decompose(pos, smoothing=0)
        pos    = layout.exchange(pos)
        w      = layout.exchange(w)

        # get the position/weight of the secondaries
        rmax = numpy.max(self.attrs['edges'])
        if rmax > self.attrs['BoxSize'].max() * 0.25:
            pos_sec = numpy.concatenate(comm.allgather(pos), axis=0)
            w_sec   = numpy.concatenate(comm.allgather(w), axis=0)
        else:
            layout  = domain.decompose(pos, smoothing=rmax)
            pos_sec = layout.exchange(pos)
            w_sec   = layout.exchange(w)

        # make the KD-tree holding the secondaries
        tree_sec = kdcount.KDTree(pos_sec, boxsize=boxsize).root

        def callback(r, i, j, iprim=None):

            # remove self pairs
            valid = r > 0.
            r = r[valid]; i = i[valid]

            if iprim % 100 == 0 and self.comm.rank == 0:
                self.logger.info("done %d centrals" %iprim)

            # normalized, re-centered position array (periodic)
            dpos = (pos_sec[i] - pos[iprim])

            # enforce periodicity in dpos
            if self.attrs['periodic']:
                for axis, col in enumerate(dpos.T):
                    col[col > boxsize[axis]*0.5] -= boxsize[axis]
                    col[col <= -boxsize[axis]*0.5] += boxsize[axis]
            recen_pos = dpos / r[:,numpy.newaxis]

            # find the mapping of r to rbins
            dig = numpy.searchsorted(self.attrs['edges'], r, side='left')

            # evaluate all Ylms
            Ylms = Ylm_cache(recen_pos[:,0]+1j*recen_pos[:,1], recen_pos[:,2])

            # sqrt of primary weight
            w0 = w[iprim]

            # loop over each (l,m) pair
            for (l,m) in Ylms:

                # the Ylm evaluated at galaxy positions
                weights = Ylms[(l,m)] * w_sec[i]

                # sum over for each radial bin
                alm = numpy.zeros(nbins, dtype='c8')
                alm += numpy.bincount(dig, weights=weights.real, minlength=nbins+2)[1:-1]
                if m != 0:
                    alm += 1j*numpy.bincount(dig, weights=weights.imag, minlength=nbins+2)[1:-1]

                # compute alm * conjugate(alm)
                alm = w0*numpy.outer(alm, alm.conj())
                if m != 0: alm += alm.T # add in the -m contribution for m != 0
                zeta[l,...] += alm.real

        # compute multipoles for each primary
        for iprim in range(len(pos)):
            tree_prim = kdcount.KDTree(numpy.atleast_2d(pos[iprim]), boxsize=boxsize).root
            tree_sec.enum(tree_prim, rmax, process=callback, iprim=iprim)

        # sum across all ranks
        zeta = comm.allreduce(zeta)

        # normalize according to Eq. 15 of Slepian et al. 2015
        # differs by factor of (4 pi)^2 / (2l+1) from the C++ code
        #zeta /= (4*numpy.pi)
        ells = numpy.array(self.attrs['poles'])
        zeta *= (4*numpy.pi) / (2*ells+1)[:,None,None]

        # make a DataSet
        dtype = numpy.dtype([('zeta_%d' %i, zeta.dtype) for i in range(Nell)])
        data = numpy.empty(zeta.shape[-2:], dtype=dtype)
        for i in range(Nell):
            data['zeta_%d' %i] = zeta[i]

        # save the result
        self.poles = DataSet(['r1', 'r2'], [redges, redges], data)

    def __getstate__(self):
        return {'poles':self.poles.data, 'attrs':self.attrs}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.poles = DataSet(['r1', 'r2'], [self.attrs['edges']]*2, self.poles)

    def save(self, output):
        """
        Save result as a JSON file
        """
        import json
        from nbodykit.utils import JSONEncoder

        # only the master rank writes
        if self.comm.rank == 0:
            self.logger.info('measurement done; saving result to %s' % output)

            with open(output, 'w') as ff:
                json.dump(self.__getstate__(), ff, cls=JSONEncoder)

    @classmethod
    @CurrentMPIComm.enable
    def load(cls, output, comm=None):
        """
        Load a result has been saved to disk with :func:`save`.
        """
        import json
        from nbodykit.utils import JSONDecoder
        if comm.rank == 0:
            with open(output, 'r') as ff:
                state = json.load(ff, cls=JSONDecoder)
        else:
            state = None
        state = comm.bcast(state)
        self = object.__new__(cls)
        self.__setstate__(state)
        self.comm = comm
        return self

class YlmCache(object):
    """
    A class to compute spherical harmonics :math:`Y_{lm}` up
    to a specified maximum :math:`\ell`

    During calculation, the necessary power of Cartesian unit
    vectors are cached in memory to avoid repeated calculations
    for separate harmonics
    """
    def __init__(self, ells, comm):

        import sympy as sp
        from sympy.utilities.lambdify import implemented_function

        self.ells = list(ells)
        self.max_ell = max(ells)
        lms = [(int(l),int(m)) for l in ells for m in range(0, l+1)]

        # compute the Ylm string expressions in parallel
        exprs = []
        for i in range(comm.rank, len(lms), comm.size):
            lm = lms[i]
            exprs.append((lm, self._get_Ylm(*lm)))
        exprs = [x for sublist in comm.allgather(exprs) for x in sublist]

        # determine the powers entering into each expression
        args = {}
        for lm, expr in exprs:
            matches = []
            for var in ['xpyhat', 'zhat']:
                for e in range(2, max(ells)+1):
                    name = var + '**' + str(e)
                    if name in str(expr):
                        matches.append((name, 'cached_'+var, str(e)))
                args[lm] = matches


        # define a function to return cached power
        def from_cache(name, pow):
            return self._cache[str(name)+str(pow)]
        f = implemented_function(sp.Function('from_cache'), from_cache)

        # arguments to the sympy functions
        zhat   = sp.Symbol('zhat', real=True, positive=True)
        xpyhat = sp.Symbol('xpyhat', complex=True)

        self._cache = {}

        # make the Ylm functions
        self._Ylms = {}
        for lm, expr in exprs:
            for var in args[lm]:
                expr = expr.replace(var[0], 'from_cache(%s, %s)' %var[1:])
            self._Ylms[lm] = sp.lambdify((xpyhat, zhat), expr)

    def __call__(self, xpyhat, zhat):
        """
        Return a dictionary holding Ylm for each (l,m) combination
        required

        Parameters
        ----------
        xpyhat : array_like
            a complex array holding xhat + i * yhat, where xhat and yhat
            are the two cartesian unit vectors
        zhat : array_like
            the third cartesian unit vector
        """
        # fill the cache first
        self._cache['cached_xpyhat2'] = xpyhat**2
        self._cache['cached_zhat2'] = zhat**2
        for name,x in zip(['cached_xpyhat', 'cached_zhat'], [xpyhat, zhat]):
            for i in range(3, self.max_ell+1):
                self._cache[name+str(i)] = self._cache[name+str(i-1)]*x

        # return a dictionary for each (l,m) tuple
        toret = {}
        for lm in self._Ylms:
            toret[lm] = self._Ylms[lm](xpyhat, zhat)
        return toret

    def _get_Ylm(self, l, m):
        """
        Compute an expression for spherical harmonic of order (l,m)
        in terms of Cartesian unit vectors, :math:`\hat{z}`
        and :math:`\hat{x} + i \hat{y}`

        Parameters
        ----------
        l : int
            the degree of the harmonic
        m : int
            the order of the harmonic; |m| < l

        Returns
        -------
        expr :
            a sympy expression that corresponds to the
            requested Ylm

        References
        ----------
        https://en.wikipedia.org/wiki/Spherical_harmonics
        """
        import sympy as sp

        # the relevant cartesian and spherical symbols
        x, y, z, r = sp.symbols('x y z r', real=True, positive=True)
        xhat, yhat, zhat = sp.symbols('xhat yhat zhat', real=True, positive=True)
        xpyhat = sp.Symbol('xpyhat', complex=True)
        phi, theta = sp.symbols('phi theta')
        defs = [(sp.sin(phi), y/sp.sqrt(x**2+y**2)),
                (sp.cos(phi), x/sp.sqrt(x**2+y**2)),
                (sp.cos(theta), z/sp.sqrt(x**2 + y**2 + z**2))
                ]

        # the cos(theta) dependence encoded by the associated Legendre poly
        expr = sp.assoc_legendre(l, m, sp.cos(theta))

        # the exp(i*m*phi) dependence
        expr *= sp.expand_trig(sp.cos(m*phi)) + sp.I*sp.expand_trig(sp.sin(m*phi))

        # simplifying optimizations
        expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
        expr = expr.expand().subs([(x/r, xhat), (y/r, yhat), (z/r, zhat)])
        expr = expr.factor().factor(extension=[sp.I]).subs(xhat+sp.I*yhat, xpyhat)
        expr = expr.subs(xhat**2 + yhat**2, 1-zhat**2).factor()

        # and finally add the normalization
        amp = sp.sqrt((2*l+1) / (4*numpy.pi) * sp.factorial(l-m) / sp.factorial(l+m))
        expr *= amp

        return expr
