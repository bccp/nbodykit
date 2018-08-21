from nbodykit import CurrentMPIComm
from nbodykit.binned_statistic import BinnedStatistic
from collections import OrderedDict
import numpy
import logging
import kdcount

class Base3PCF(object):
    """
    Base class for implementing common 3PCF calculations.

    Users should use :class:`SimulationBox3PCF` or :class:`SurveyData3PCF`.
    """
    def __init__(self, source, poles, edges, required_cols, BoxSize=None, periodic=None):

        from .pair_counters.base import verify_input_sources

        # verify the input sources
        inspect = periodic is not None
        BoxSize = verify_input_sources(source, None, BoxSize, required_cols, inspect_boxsize=inspect)

        self.source = source
        self.comm = self.source.comm

        # save the meta-data
        self.attrs = {}
        self.attrs['poles'] = poles
        self.attrs['edges'] = edges

        # store periodic/BoxSize for SimulationBox
        if periodic is not None:
            self.attrs['BoxSize'] = BoxSize
            self.attrs['periodic'] = periodic

    def _run(self, pos, w, pos_sec, w_sec, boxsize=None, bunchsize=10000):
        """
        Internal function to run the 3PCF algorithm on the input data and
        weights.

        The input data/weights have already been domain-decomposed, and
        the loads should be balanced on all ranks.
        """
        # maximum radius
        rmax = numpy.max(self.attrs['edges'])

        # the array to hold output values
        nbins  = len(self.attrs['edges'])-1
        Nell   = len(self.attrs['poles'])
        zeta = numpy.zeros((Nell,nbins,nbins), dtype='f8')
        alms = {}
        walms = {}

        # compute the Ylm expressions we need
        if self.comm.rank == 0:
            self.logger.info("computing Ylm expressions...")
        Ylm_cache = YlmCache(self.attrs['poles'], self.comm)
        if self.comm.rank ==  0:
            self.logger.info("...done")

        # make the KD-tree holding the secondaries
        tree_sec = kdcount.KDTree(pos_sec, boxsize=boxsize).root

        def callback(r, i, j, iprim=None):

            # remove self pairs
            valid = r > 0.
            r = r[valid]; i = i[valid]

            # normalized, re-centered position array (periodic)
            dpos = (pos_sec[i] - pos[iprim])

            # enforce periodicity in dpos
            if boxsize is not None:
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
                alm = alms.setdefault((l, m), numpy.zeros(nbins, dtype='c16'))
                walm = walms.setdefault((l, m), numpy.zeros(nbins, dtype='c16'))

                r1 = numpy.bincount(dig, weights=weights.real, minlength=nbins+2)[1:-1]
                alm[...] += r1
                walm[...] += w0 * r1
                if m != 0:
                    i1 = numpy.bincount(dig, weights=weights.imag, minlength=nbins+2)[1:-1]
                    alm[...] += 1j*i1
                    walm[...] += w0*1j*i1

        # determine rank with largest load
        loads = self.comm.allgather(len(pos))
        largest_load = numpy.argmax(loads)
        chunk_size = max(loads) // 10

        # compute multipoles for each primary (s vector in the paper)
        for iprim in range(len(pos)):
            # alms must be clean for each primary particle; (s) in eq 15 and 8 of arXiv:1506.02040v2
            alms.clear()
            walms.clear()
            tree_prim = kdcount.KDTree(numpy.atleast_2d(pos[iprim]), boxsize=boxsize).root
            tree_sec.enum(tree_prim, rmax, process=callback, iprim=iprim, bunch=bunchsize)

            if self.comm.rank == largest_load and iprim % chunk_size == 0:
                self.logger.info("%d%% done" % (10*iprim//chunk_size))

            # combine alms into zeta(s);
            # this cannot be done in the callback because
            # it is a nonlinear function (outer product) of alm.
            for (l, m) in alms:
                alm = alms[(l, m)]
                walm = walms[(l, m)]

                # compute alm * conjugate(alm)
                alm_w_alm = numpy.outer(walm, alm.conj())
                if m != 0: alm_w_alm += alm_w_alm.T # add in the -m contribution for m != 0
                zeta[Ylm_cache.ell_to_iell[l], ...] += alm_w_alm.real

        # sum across all ranks
        zeta = self.comm.allreduce(zeta)

        # normalize according to Eq. 15 of Slepian et al. 2015
        # differs by factor of (4 pi)^2 / (2l+1) from the C++ code
        zeta /= (4*numpy.pi)

        # make a BinnedStatistic
        dtype = numpy.dtype([('corr_%d' % ell, zeta.dtype) for ell in self.attrs['poles']])
        data = numpy.empty(zeta.shape[-2:], dtype=dtype)
        for i, ell in enumerate(self.attrs['poles']):
            data['corr_%d' % ell] = zeta[i]

        # save the result
        edges = self.attrs['edges']
        poles = BinnedStatistic(['r1', 'r2'], [edges, edges], data)
        return poles

    def __getstate__(self):
        return {'poles':self.poles.data, 'attrs':self.attrs}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.poles = BinnedStatistic(['r1', 'r2'], [self.attrs['edges']]*2, self.poles)

    def save(self, output):
        """
        Save the :attr:`poles` result to a JSON file with name ``output``.
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
    def load(cls, filename, comm=None):
        """
        Load a result from ``filename`` that has been saved to
        disk with :func:`save`.
        """
        import json
        from nbodykit.utils import JSONDecoder
        if comm.rank == 0:
            with open(filename, 'r') as ff:
                state = json.load(ff, cls=JSONDecoder)
        else:
            state = None
        state = comm.bcast(state)
        self = object.__new__(cls)
        self.__setstate__(state)
        self.comm = comm
        return self

class SimulationBox3PCF(Base3PCF):
    """
    Compute the multipoles of the isotropic, three-point correlation function
    in configuration space for data in a simulation box.

    This uses the algorithm of Slepian and Eisenstein, 2015 which scales
    as :math:`\mathcal{O}(N^2)`, where :math:`N` is the number of objects.

    Results are computed when the object is inititalized. See the documenation
    of :func:`run` for the attributes storing the results.

    .. note::

        The algorithm expects the positions of objects in a simulation box to
        be the Cartesian ``x``, ``y``, and ``z`` vectors. For survey data,
        in the form of right ascension, declination, and
        redshift, see :class:`~nbodykit.algorithms.SurveyData3PCF`.

    Parameters
    ----------
    source : CatalogSource
        the input source of particles providing the 'Position' column
    poles : list of int
        the list of multipole numbers to compute
    edges : array_like
        the edges of the bins of separation to use; length of nbins+1
    BoxSize : float, 3-vector, optional
        the size of the box; if periodic boundary conditions used, and 'BoxSize'
        not provided in the source :attr:`attrs`, it must be provided here
    periodic : bool, optional
        whether to use periodic boundary conditions when computing separations
        between objects
    weight : str, optional
        the name of the column in the source specifying the particle weights

    References
    ----------
    Slepian and Eisenstein, MNRAS 454, 4142-4158 (2015)
    """
    logger = logging.getLogger("SimulationBox3PCF")

    def __init__(self, source, poles, edges, BoxSize=None, periodic=True, weight='Weight'):

        # initialize the base class
        required_cols = ['Position', weight]
        Base3PCF.__init__(self, source, poles, edges, required_cols,
                            BoxSize=BoxSize, periodic=periodic)

        # save the weight column
        self.attrs['weight'] = weight

        # check largest possible separation
        if periodic:
            min_box_side = 0.5*self.attrs['BoxSize'].min()
            if numpy.amax(edges) > min_box_side:
                raise ValueError(("periodic pair counts cannot be computed for Rmax > BoxSize/2"))

        # run the algorithm
        self.poles = self.run()


    def run(self, pedantic=False):
        """
        Compute the three-point CF multipoles. This attaches the following
        the attributes to the class:

        - :attr:`poles`

        Attributes
        ----------
        poles : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            a BinnedStatistic object to hold the multipole results; the
            binned statistics stores the multipoles as variables ``corr_0``,
            ``corr_1``, etc for :math:`\ell=0,1,` etc. The coordinates
            of the binned statistic are ``r1`` and ``r2``, which give the
            separations between the three objects in CF.
        """
        from .pair_counters.domain import decompose_box_data

        # the box size to use
        if self.attrs['periodic']:
            boxsize = self.attrs['BoxSize']
        else:
            boxsize = None

        # domain decompose the data
        smoothing = numpy.max(self.attrs['edges'])
        (pos, w), (pos_sec, w_sec) = decompose_box_data(self.source, None, self.attrs,
                                                        self.logger, smoothing)

        # run the algorithm
        if pedantic:
            return self._run(pos, w, pos_sec, w_sec, boxsize=boxsize, bunchsize=1)
        else:
            return self._run(pos, w, pos_sec, w_sec, boxsize=boxsize)

class SurveyData3PCF(Base3PCF):
    """
    Compute the multipoles of the isotropic, three-point correlation function
    in configuration space for observational survey data.

    This uses the algorithm of Slepian and Eisenstein, 2015 which scales
    as :math:`\mathcal{O}(N^2)`, where :math:`N` is the number of objects.

    Results are computed when the object is inititalized. See the documenation
    of :func:`run` for the attributes storing the results.

    .. note::

        The algorithm expects the positions of objects from a survey catalog
        be the sky coordinates, right ascension and declination, and redshift.
        For simulation box data in Cartesian coordinates, see
        :class:`~nbodykit.algorithms.SimulationBox3PCF`.

    .. warning::
        The right ascension and declination columns should be specified
        in degrees.

    Parameters
    ----------
    source : CatalogSource
        the input source of particles providing the 'Position' column
    poles : list of int
        the list of multipole numbers to compute
    edges : array_like
        the edges of the bins of separation to use; length of nbins+1
    cosmo : :class:`~nbodykit.cosmology.cosmology.Cosmology`
        the cosmology instance used to convert redshifts into comoving distances
    ra : str, optional
        the name of the column in the source specifying the
        right ascension coordinates in units of degrees; default is 'RA'
    dec : str, optional
        the name of the column in the source specifying the declination
        coordinates; default is 'DEC'
    redshift : str, optional
        the name of the column in the source specifying the redshift
        coordinates; default is 'Redshift'
    weight : str, optional
        the name of the column in the source specifying the object weights
    domain_factor : int, optional
        the integer value by which to oversubscribe the domain decomposition
        mesh before balancing loads; this number can affect the distribution
        of loads on the ranks -- an optimal value will lead to balanced loads

    References
    ----------
    Slepian and Eisenstein, MNRAS 454, 4142-4158 (2015)
    """
    logger = logging.getLogger("SurveyData3PCF")

    def __init__(self, source, poles, edges, cosmo, domain_factor=4,
                    ra='RA', dec='DEC', redshift='Redshift', weight='Weight'):

        # initialize the base class
        required_cols = [ra, dec, redshift, weight]
        Base3PCF.__init__(self, source, poles, edges, required_cols)

        # save meta-data
        self.attrs['cosmo'] = cosmo
        self.attrs['weight'] = weight
        self.attrs['ra'] = ra
        self.attrs['dec'] = dec
        self.attrs['redshift'] = redshift
        self.attrs['domain_factor'] = domain_factor

        # run the algorithm
        self.poles = self.run()

    def run(self):
        """
        Compute the three-point CF multipoles. This attaches the following
        the attributes to the class:

        - :attr:`poles`

        Attributes
        ----------
        poles : :class:`~nbodykit.binned_statistic.BinnedStatistic`
            a BinnedStatistic object to hold the multipole results; the
            binned statistics stores the multipoles as variables ``corr_0``,
            ``corr_1``, etc for :math:`\ell=0,1,` etc. The coordinates
            of the binned statistic are ``r1`` and ``r2``, which give the
            separations between the three objects in CF.
        """
        from .pair_counters.domain import decompose_survey_data

        # domain decompose the data
        # NOTE: pos and pos_sec are Cartesian!
        smoothing = numpy.max(self.attrs['edges'])
        (pos, w), (pos_sec, w_sec) = decompose_survey_data(self.source, None,
                                                            self.attrs, self.logger,
                                                            smoothing,
                                                            return_cartesian=True,
                                                            domain_factor=self.attrs['domain_factor'])

        # run the algorithm
        return self._run(pos, w, pos_sec, w_sec)


class YlmCache(object):
    """
    A class to compute spherical harmonics :math:`Y_{lm}` up
    to a specified maximum :math:`\ell`.

    During calculation, the necessary power of Cartesian unit
    vectors are cached in memory to avoid repeated calculations
    for separate harmonics.
    """
    def __init__(self, ells, comm):

        import sympy as sp
        from sympy.utilities.lambdify import implemented_function
        from sympy.parsing.sympy_parser import parse_expr

        self.ells = numpy.asarray(ells).astype(int)
        self.max_ell = max(self.ells)

        # look up table from ell to iell, index for cummulating results.
        self.ell_to_iell = numpy.empty(self.max_ell + 1, dtype=int)
        for iell, ell in enumerate(self.ells):
            self.ell_to_iell[ell] = iell

        lms = [(l,m) for l in ells for m in range(0, l+1)]

        # compute the Ylm string expressions in parallel
        exprs = []
        for i in range(comm.rank, len(lms), comm.size):
            lm = lms[i]
            exprs.append((lm, str(self._get_Ylm(*lm))))
        exprs = [x for sublist in comm.allgather(exprs) for x in sublist]

        # determine the powers entering into each expression
        args = {}
        for lm, expr in exprs:
            matches = []
            for var in ['xpyhat', 'zhat']:
                for e in range(2, max(ells)+1):
                    name = var + '**' + str(e)
                    if name in expr:
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
        self._Ylms = OrderedDict()
        for lm, expr in exprs:
            expr = parse_expr(expr, local_dict={'zhat':zhat, 'xpyhat':xpyhat})
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
