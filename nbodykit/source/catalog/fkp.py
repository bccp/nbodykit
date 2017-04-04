from nbodykit.utils import attrs_to_dict
from nbodykit.transform import ConstantArray
from nbodykit.base.catalog import CatalogSource, column
from nbodykit.base.catalogmesh import CatalogMeshSource

import numpy
import logging
import functools
import contextlib
from pmesh.pm import RealField, ComplexField

class FKPMeshSource(CatalogMeshSource):
    """
    A subclass of :class:`~nbodykit.base.catalogmesh.CatalogMeshSource`
    designed to paint the FKP density field to a mesh, given
    two catalogs, one for `data` and one for `randoms`
    """
    logger = logging.getLogger('FKPMeshSource')
    
    def __init__(self, source, BoxSize, Nmesh, dtype, selection, weight, comp_weight, fkp_weight, nbar):
        
        self.source      = source
        self.comp_weight = comp_weight
        self.fkp_weight  = fkp_weight
        self.nbar        = nbar
        
        CatalogMeshSource.__init__(self, source, BoxSize, Nmesh, dtype, weight, selection)
        
    @contextlib.contextmanager
    def _set_mesh(self, prefix):
        """
        Internal context manager to set the appropriate
        column names needed for painting, either for "data" or "randoms"
        
        Parameters
        ----------
        prefix : {'data', 'randoms'}
            the prefix prepended to the column names
        """
        # simple sanity check
        if prefix not in ['data', 'randoms']:
            raise ValueError("'name' should be 'data' or 'randoms'")
            
        # save the original columns
        pos, weight, sel = self.position, self.weight, self.selection
        
        # set for data or names
        self.position  = prefix+'.'+pos
        self.weight    = prefix+'.'+weight
        self.selection = prefix+'.'+sel
        yield
        
        # restore
        self.position  = pos
        self.weight    = weight
        self.selection = sel
        
    def to_real_field(self):
        """
        Paint the FKP density field, returning a ``RealField``
        
        Given the ``data`` and ``randoms`` catalogs, this paints:
        
        .. math:: 
            
            F(x) = w_fkp(x) * [w_comp(x)*n_data(x) - alpha * w_comp(x)*n_randoms(x)]
        """
        # paint -1.0*alpha*N_randoms 
        with self._set_mesh('randoms'):
            real = CatalogMeshSource.to_real_field(self)
            
        # from N/Nbar = 1+delta to un-normalized number
        real[:] *= real.attrs['N'] / numpy.prod(self.pm.Nmesh)
        
        # update the statistics
        for name, source in zip(['data', 'randoms'], [self.source.data, self.source.randoms]):
            
            # power spectrum normalization
            real.attrs[name+'.A'] = self._normalization(name)
            
            # power spectrum shot noise
            real.attrs[name+'.S'] = self._shotnoise(name)
            
            # total number
            real.attrs[name+'.N'] = self._unweighted_total(name)
            
            # weighted number (sum of completeness)
            real.attrs[name+'.W'] = self._weighted_total(name)
            
        # compute alpha from sum of completeness weights
        real.attrs['alpha'] = real.attrs['data.W'] / real.attrs['randoms.W']
        
        # finish the statistics
        real.attrs['randoms.A'] *= real.attrs['alpha']
        real.attrs['randoms.S'] *= real.attrs['alpha']**2 / real.attrs['randoms.A']
        real.attrs['data.S'] /=  real.attrs['randoms.A']
        real.attrs['shotnoise'] = real.attrs['data.S'] + real.attrs['randoms.S']
        
        # randoms get -alpha factor; alpha is W_data / W_randoms
        real[:] *= -real.attrs['alpha']

        # paint the data
        with self._set_mesh('data'):
            real2 = CatalogMeshSource.to_real_field(self)
        
        # from N/Nbar = 1+delta to un-normalized number 
        real2[:] *= real2.attrs['N'] / numpy.prod(self.pm.Nmesh)
        
        # data - alpha * randoms
        real[:] += real2[:]
            
        # divide by volume per cell to go from number to number density
        vol_per_cell = (self.pm.BoxSize/self.pm.Nmesh).prod()
        real[:] /= vol_per_cell
            
        return real
                    
    def _normalization(self, name):
        """
        Compute the power spectrum normalization, using either the
        `data` or `randoms` source
    
        This computes
        
        .. math:: 
        
            A = \sum \bar{n} w_\mathrm[comp] w_\mathrm{fkp}^2
        
        References
        ----------
        see Eqs. 13,14 of Beutler et al. 2014, "The clustering of galaxies in the 
        SDSS-III Baryon Oscillation Spectroscopic Survey: testing gravity with redshift 
        space distortions using the power spectrum multipoles"
        """  
        sel = self.compute(self[name+'.'+self.selection])
               
        nbar        = self[name+'.'+self.nbar][sel]
        comp_weight = self[name+'.'+self.comp_weight][sel]
        fkp_weight  = self[name+'.'+self.fkp_weight][sel]
        A           = nbar*comp_weight*fkp_weight**2
        
        A = self.compute(A.sum())
        return self.comm.allreduce(A)
        
    def _shotnoise(self, name):
        """
        Compute the power spectrum shot noise, using either the
        `data` or `randoms` source
    
        This computes
        
        .. math:: 
        
            S = \sum (w_\mathrm[comp] w_\mathrm{fkp})^2
        
        References
        ----------
        see Eq. 15 of Beutler et al. 2014, "The clustering of galaxies in the 
        SDSS-III Baryon Oscillation Spectroscopic Survey: testing gravity with redshift 
        space distortions using the power spectrum multipoles"
        """
        sel = self.compute(self[name+'.'+self.selection])
                 
        comp_weight = self[name+'.'+self.comp_weight][sel]
        fkp_weight  = self[name+'.'+self.fkp_weight][sel]
        S           = (comp_weight*fkp_weight)**2
        
        S = self.compute(S.sum())
        return self.comm.allreduce(S)
                                
    def _weighted_total(self, name):
        """
        Compute the weighted total number of objects, using either the
        `data` or `randoms` source
    
        This is just the sum of the completeness weights: 
        
        .. math:: 
        
            W = \sum w_\mathrm{comp}
        """
        sel = self.compute(self[name+'.'+self.selection])
        comp_weight = self[name+'.'+self.comp_weight][sel]
        
        wsum = self.compute(comp_weight.sum())
        return self.comm.allreduce(wsum)
        
    def _unweighted_total(self, name):
        """
        Compute the total number of objects, using either the
        `data` or `randoms` source
        """
        sel = self.compute(self[name+'.'+self.selection])
        Nsum = self.compute(sel.sum())
        return self.comm.allreduce(Nsum)
                    
def FKPColumn(self, which, col):
    source = getattr(self, which)
    return source[col]

class FKPCatalog(CatalogSource):
    """
    Combine a ``data`` CatalogSource and a ``randoms`` CatalogSource
    into a single unified Source
    
    This main functionality of this class is:
    
        *   provide a uniform interface to accessing columns from the 
            `data` CatalogSource and `randoms` CatalogSource, using
            column names prefixed with "data." or "randoms."
        *   compute the shared :attr:`BoxSize` of the Source, by
            finding the maximum Cartesian extent of the `randoms`
    """
    logger = logging.getLogger('FKPCatalog')

    def __init__(self, data, randoms, BoxSize=None, BoxPad=0.02, use_cache=True):
        """
        Parameters
        ----------
        data : CatalogSource
            the Source of particles representing the `data` catalog
        randoms : CatalogSource
            the Source of particles representing the `randoms` catalog
        BoxSize : float, 3-vector; optional
            the size of the Cartesian box to use for the unified `data` and 
            `randoms`; if not provided, the maximum Cartesian extent of the 
            `randoms` defines the box
        BoxPad : float, 3-vector; optional
            optionally apply this additional buffer to the extent of the 
            Cartesian box
        use_cache : bool; optional
            if ``True``, use the built-in dask cache system to cache
            data, providing significant speed-ups; requires :mod:`cachey`
        """
        # some sanity checks first
        assert data.comm is randoms.comm, "mismatch between communicator of `data` and `randoms"
        self.comm    = data.comm
        self.data    = data
        self.randoms = randoms
                
        # update the dictionary with data/randoms attrs
        self.attrs.update(attrs_to_dict(data, 'data.'))
        self.attrs.update(attrs_to_dict(randoms, 'randoms.'))
        
        # init the base class
        CatalogSource.__init__(self, comm=self.comm, use_cache=use_cache)
        
        # turn on cache?
        if self.use_cache:
            self.data.use_cache = True
            self.randoms.use_cache = True
                    
        # add new weight columns: FKPWeight and TotalWeight
        import dask.array as da
        for source in [self.data, self.randoms]:
            source['FKPWeight'] = ConstantArray(1.0, source.size, chunks=100000)
            source['TotalWeight'] = ConstantArray(1.0, source.size, chunks=100000)
            
        # prefixed columns in this source return on-demand from "data" or "randoms"
        for name in ['data', 'randoms']:
            source = getattr(self, name)
            for col in source.columns:
                f = functools.partial(FKPColumn, col=col, which=name)
                f.__name__ = name+'.'+col
                setattr(self.__class__, name+'_'+col, column(f))
                
        # determine the BoxSize 
        if numpy.isscalar(BoxSize):
            BoxSize = numpy.ones(3)*BoxSize
        self.attrs['BoxSize'] = BoxSize
        if numpy.isscalar(BoxPad):
            BoxPad = numpy.ones(3)*BoxPad
        self.attrs['BoxPad'] = BoxPad
        self._define_cartesian_box()
            
    @property
    def size(self):
        """
        This is not implemented because we actually have size of `data` and `randoms`
        """
        return NotImplemented
        
    def _define_cartesian_box(self):
        """
        Internal function to put the :attr:`randoms` CatalogSource in a Cartesian box
    
        This function add two necessary attribues:
    
        1. :attr:`BoxSize` : array_like, (3,)
            if not provided, the BoxSize in each direction is computed from
            the maximum extent of the Cartesian coordinates of the :attr:`randoms`
            Source, with an optional, additional padding
        2. :attr:`BoxCenter`: array_like, (3,)
            the mean coordinate value in each direction; this is used to re-center
            the Cartesian coordinates of the :attr:`data` and :attr:`randoms`
            to the range of ``[-BoxSize/2, BoxSize/2]``
        """
        # need to compute cartesian min/max
        pos_min = numpy.array([numpy.inf]*3)
        pos_max = numpy.array([-numpy.inf]*3)
    
        Position = self['randoms.Position']
        N = max(self.comm.allgather(len(Position)))
        
        chunksize = 1024 ** 2
        for i in range(0, N, chunksize):
            s = slice(i, i + chunksize)
            
            if len(Position) != 0:
                pos = self.compute(Position[s])

                # global min/max of cartesian coordinates
                pos_min = numpy.minimum(pos_min, pos.min(axis=0))
                pos_max = numpy.maximum(pos_max, pos.max(axis=0))

        # gather everything to root
        pos_min = self.comm.gather(pos_min)
        pos_max = self.comm.gather(pos_max)
    
        # rank 0 setups up the box and computes nbar (if needed)
        if self.comm.rank == 0:
        
            # find the global coordinate minimum and maximum
            pos_min = numpy.amin(pos_min, axis=0)
            pos_max = numpy.amax(pos_max, axis=0)
        
            # used to center the data in the first cartesian quadrant
            delta = abs(pos_max - pos_min)
            self.attrs['BoxCenter'] = 0.5 * (pos_min + pos_max)
    
            # BoxSize is padded diff of min/max coordinates
            if self.attrs['BoxSize'] is None:
                delta *= 1.0 + self.attrs['BoxPad']
                self.attrs['BoxSize'] = numpy.ceil(delta) # round up to nearest integer
        else:
            self.attrs['BoxCenter'] = None
        
        # broadcast the results that rank 0 computed
        self.attrs['BoxSize'] = self.comm.bcast(self.attrs['BoxSize'])
        self.attrs['BoxCenter'] = self.comm.bcast(self.attrs['BoxCenter'])

        # log some info
        if self.comm.rank == 0:
            self.logger.info("BoxSize = %s" %str(self.attrs['BoxSize']))
            self.logger.info("cartesian coordinate range: %s : %s" %(str(pos_min), str(pos_max)))
            self.logger.info("BoxCenter = %s" %str(self.attrs['BoxCenter']))
        
    def to_mesh(self, Nmesh=None, BoxSize=None, dtype='f4', interlaced=False, compensated=False, 
                window='cic', fkp_weight='FKPWeight', comp_weight='Weight', selection='Selection',
                nbar='NZ'):
                
        """
        Convert the FKPCatalog to a mesh, which knows how to "paint" the catalog
                
        Additional keywords to the :func:`to_mesh` function include the FKP weight column,
        completeness weight column, and the column specifying the number density as a 
        function of redshift.
        
        Parameters
        ----------
        Nmesh : int, 3-vector; optional
            the number of cells per box side; if not specified in `attrs`, this
            must be provided
        BoxSize : float, 3-vector; optional
            the size of the box; if provided, this will use the default value in `attrs`
        dtype : str, dtype; optional
            the data type of the mesh when painting
        interlaced : bool; optional
            whether to use interlacing to reduce aliasing when painting the particles 
            on the mesh
        compensated : bool; optional
            whether to apply a Fourier-space transfer function to account for the 
            effects of the gridding + aliasing
        window : str; optional
            the string name of the window to use when interpolating the particles 
            to the mesh; see ``pmesh.window.methods`` for choices
        fkp_weight : str; optional
            the name of the column in the source specifying the FKP weight; this 
            weight is applied to the FKP density field: ``n_data - alpha*n_randoms``
        comp_weight : str; optional
            the name of the column in the source specifying the completeness weight; 
            this weight is applied to the individual fields, either ``n_data``  or ``n_random``
        selection : str; optional
            the name of the column used to select a subset of the source when painting
        nbar : str; optional
            the name of the column specifying the number density as a function of redshift
        """
        # verify that all of the required columns exist    
        for name in ['data', 'randoms']:
            for col in ['Position', fkp_weight, comp_weight, selection, nbar]:
                col = name+'.'+col
                if col not in self:
                    raise ValueError("missing '%s' column; try using `source[column] = array` syntax" %col)
        
        if BoxSize is None:
            BoxSize = self.attrs['BoxSize']

        if Nmesh is None:
            try:
                Nmesh = self.attrs['Nmesh']
            except KeyError:
                raise ValueError("cannot convert FKP source to a mesh; 'Nmesh' keyword is not "
                                 "supplied and the FKP source does not define one in 'attrs'.")
                
        # initialize the FKP mesh
        kws = {'Nmesh':Nmesh, 'BoxSize':BoxSize, 'dtype':dtype, 'selection':selection}
        mesh = FKPMeshSource(self,  nbar=nbar, weight='TotalWeight', comp_weight=comp_weight, fkp_weight=fkp_weight, **kws)
        mesh.interlaced = interlaced
        mesh.compensated = compensated
        mesh.window = window
        
        # add some additional internal columns to the mesh
        for name in ['data', 'randoms']:
            
            # total weight for the mesh is completeness weight x FKP weight
            mesh[name+'.TotalWeight'] = self[name+'.'+comp_weight] * self[name+'.'+fkp_weight]
            
            # position on the mesh is re-centered to [-BoxSize/2, BoxSize/2]
            mesh[name+'.Position']   = self[name+'.Position'] - self.attrs['BoxCenter']
        
        return mesh

