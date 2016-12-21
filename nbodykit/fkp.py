import os
import numpy
import logging
import warnings
from contextlib import contextmanager
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from nbodykit.extern import six
from nbodykit.core import DataSource, Painter, Algorithm
from nbodykit.distributedarray import GatherArray

logger = logging.getLogger('FKPCatalog')

def is_float(s):
    """
    Determine if a string can be cast safely to a float
    """
    try: float(s); return True
    except ValueError: return False

class FKPCatalog(object):
    """
    A `DataSource` representing a catalog of tracer objects, 
    designed to be used in analysis similar to that first outlined 
    by Feldman, Kaiser, and Peacock (FKP) 1994 (astro-ph/9304022)
    
    In particular, the `FKPCatalog` uses a catalog of random objects
    to define the mean density of the survey, in addition to the catalog
    of data objects.
    
    
    Attributes
    ----------
    data : DataSource
        a `DataSource` that returns the position, weight, etc of the 
        true tracer objects, whose intrinsic clustering is non-zero
    randoms : DataSource
        a `DataSource` that returns the position, weight, etc 
        of a catalog of objects generated randomly to match the
        survey geometry and whose instrinsic clustering is zero
    BoxSize : array_like (3,)
        the size of the cartesian box -- the Cartesian coordinates
        of the input objects are computed using the input cosmology,
        and then placed into the box
    mean_coordinate_offset : array_like, (3,) 
        the average coordinate value in each dimension -- this offset
        is used to return cartesian coordinates translated into the
        domain of [-BoxSize/2, BoxSize/2]
    """
    def __init__(self,  data, 
                        randoms, 
                        BoxSize=None, 
                        BoxPad=0.02, 
                        compute_fkp_weights=False, 
                        P0_fkp=None, 
                        nbar=None, 
                        fsky=None):
        """
        Parameters
        ----------
        data : DataSource
            the DataSource that corresponds to the 'data' catalog
        randoms : DataSource
            the DataSource that corresponds to the 'randoms' catalog
        BoxSize : {float, array_like (3,)}, optional
            the size of the Cartesian to box when gridding the data and randoms; 
            if not provided, the box will automatically be computed from the
            maximum extent of the randoms catalog
        BoxPad : float, optional
            if BoxSize is not provided, apply this padding to the box that
            is automatically created from the randoms catalog; default is ``0.02``
        compute_fkp_weights : bool, optional
            if ``True``, compute and apply FKP weights using `P0_fkp` and 
            the number density n(z) column from the input data sources; 
            default is ``False``
        P0_fkp : float, optional
            if ``compute_fkp_weights=True``, use this value in in the FKP 
            weights, which are defined as, :math:`w_\mathrm{FKP} = 1 / (1 + n_g(x) * P_0)`
        nbar : {str, float}, optional
            either the name of a file, giving ``(z, n(z))`` in columns, or a scalar float,
            which gives the constant n(z) value to use
        fsky : float, optional
            if ``nbar = None``, then the n(z) is computed from the randoms catalog, and
            the sky fraction covered by the survey is required to properly normalize
            the number density (for the volume calculation)
        """
        # set the cosmology
        self.cosmo = data.cosmo
        if self.cosmo is None:
            raise ValueError("FKPCatalog requires a cosmology")
        if data.cosmo is not randoms.cosmo:
            raise ValueError("mismatch between cosmology instances of `data` and `randoms` in `FKPCatalog`")
            
        # set the comm
        self.comm = data.comm
        if data.comm is not randoms.comm:
            raise ValueError("mismatch between communicators of `data` and `randoms` in `FKPCatalog`")
        
        # data and randoms datasources
        self.data    = data
        self.randoms = randoms

        # set the BoxSize and BoxPad
        self.BoxSize = BoxSize
        if self.BoxSize is not None:
            self.BoxSize = DataSource.BoxSizeParser(self.BoxSize)
        self.BoxPad = BoxPad
        
        # weights configuration
        self.compute_fkp_weights = compute_fkp_weights
        self.P0_fkp              = P0_fkp
        
        # n(z) configuration
        self.nbar = nbar
        self.fsky = fsky
        
        # use the default painter when painting
        self.painter = Painter.create('DefaultPainter')

    #--------------------------------------------------------------------------
    # properties
    #--------------------------------------------------------------------------
    @property
    def default_columns(self):
        """
        Return a dictionary of the default columns to be read from the :attr:`data`
        and :attr:`randoms` attributes. 
        """
        return {'Redshift':-1., 'Nbar':-1., 'FKPWeight':1., 'Weight':1.}
    
    @property
    def fsky(self):
        r"""
        The sky area fraction (relative to :math:`4 \pi` steradians) 
        
        .. note::
        
            If :math:`n(z)` is not provided, then it will be computed from the 
            randoms catalog. This attribute is needed for the volume 
            normalization in that calculation, and must be set; otherwise, 
            the code will crash
        """
        try:
            return self._fsky
        except AttributeError:
            cls = self.__class__.__name__
            raise AttributeError("'%s' object has no attribute 'fsky'" %cls)
            
    @fsky.setter
    def fsky(self, val):
        """
        Set the sky fraction, only if not None
        """
        if val is not None:
            self._fsky = val
        
    @property
    def data(self):
        """
        Explicitly keep track of the `data` DataSource, in order to track 
        the total number of objects
        """
        return self._data

    @data.setter
    def data(self, val):
        """
        Set the data, 
        """
        if not hasattr(self, '_data'):
            self._data = val
        else:
            # open the new data automatically
            if not self.closed:
                
                # close the old data
                if hasattr(self, 'data_stream'):
                    self.data_stream.close()
                    del self.data_stream
                    del self.alpha
                    
                self._data = val
                self.data_stream = self._data.open(defaults=self.default_columns)
            else:
                self._data = val
    
    @property
    def alpha(self):
        r"""
        Return the ratio of the data number density to the randoms number density
        
        This is computed using the "completeness weights" : 
        
        .. math::
        
            \alpha = \sum_\mathrm{gal} w_{c,i} / \sum_\mathrm{ran} w_{c,i}
        
        where :math:`w_{c,i}` is the completeness weight for the ith galaxy. This
        is specified via the ``Weight`` column, and has a default value of 1.
        """
        try:
            return self._alpha
        except AttributeError:
            
            if self.closed:
                cls = self.__class__.__name__
                raise ValueError("please open the %s before accessing the ``alpha`` attribute" %cls)
                
            # the sum comp weights for data
            W_data = 0.
            for [comp_weight] in self.data_stream.read(['Weight'], full=False):
                W_data += comp_weight.sum()
            W_data = self.comm.allreduce(W_data)
              
            # sum comp weights for randoms
            W_ran = 0.
            for [comp_weight] in self.randoms_stream.read(['Weight'], full=False):
                W_ran += comp_weight.sum()
            W_ran = self.comm.allreduce(W_ran) 
            
            self._alpha = 1.*W_data / W_ran   
            logger.debug("alpha computed as %.4f" %self._alpha) 
            
            return self._alpha
            
    @alpha.deleter
    def alpha(self):
        if hasattr(self, '_alpha'): del self._alpha
                                
    @property
    def closed(self):
        """
        Return `True` if the catalog has been setup and the
        data and random streams are open
        """
        if not hasattr(self, 'data_stream'):
            return True
        if not hasattr(self, 'randoms_stream'):
            return True
            
        return False
        
    @property
    def nbar(self):
        """
        A callable function that returns the number density ``n(z)`` 
        as a function of redshift (provided via argument)
        """
        return self._nbar
        
    @nbar.setter
    def nbar(self, val):
        """
        Set the number density as a function of redshift, :math:`n(z)`
        
        This should be either:
            
            1.  a string, specifying the name of the file to read :math:`n(z)` from; 
                the file should contain two columns, specifying ``(z, nz)``
            2.  a float, specifying the constant :math:`n(z)` value to use
        """
        if val is not None:
            
            # input is an existing file
            if isinstance(val, six.string_types) and os.path.exists(val):
                    
                # read from the file
                try:
                    d = numpy.loadtxt(val)
                    
                    # columns are z, n(z)
                    if d.shape[1] == 2:
                        self._nbar = spline(d[:,0], d[:,1])
                
                    # columns are z_min, z_max, z_cen, n(z)
                    elif d.shape[1] == 4:
                        self._nbar = spline(d[:,-2], d[:,-1])
                    
                    # wrong number of columns
                    else:
                        raise
                except:
                    raise ValueError("n(z) file should specify either: [z, n(z)] or [z_min, z_max, z_cen, n(z)]")
            
                # redshift is required for the n(z) spline
                self._nbar.need_redshift = True
            
            # input is a float, specifying the constant nbar
            elif isinstance(val, six.string_types) and is_float(val) or numpy.isscalar(val):
                val = float(val)
                
                def constant_nbar(redshift):
                    nbar = numpy.array(val)
                    nbar = numpy.lib.stride_tricks.as_strided(nbar, (len(redshift), nbar.size), (0, nbar.itemsize))
                    return nbar.squeeze()
                self._nbar = constant_nbar # return a constant n(z)
                self._nbar.need_redshift = False
            else:
                msg = ("error setting ``nbar`` parameter from input value; should "
                       " be either the name of a file, or a float value if n(z) is constant")
                raise TypeError(msg)
        else:
            self._nbar = None
                     
    def _compute_randoms_nbar(self):
        """
        Internal function to compute the n(z) from the :attr:`randoms` DataSource
        
        This uses :class:`~nbodykit.plugins.algorithms.RedshiftHistogram.RedshiftHistogramAlgorithm`
        to compute the n(z) from the randoms catalog
        
        When computing the histogram, it uses the completeness weights, as specified
        by the ``Weight`` column, which defaults to unity. 
        """
        # only need to compute this if nbar wasn't provided
        if self.nbar is None:
            
            # initialize the RedshiftHistogram algorithm
            # run with fsky = 1.0, if no fsky is given because we might not 
            # need this n(z)
            fsky = 1. if not hasattr(self, 'fsky') else self.fsky
            nz_computer = Algorithm.create('RedshiftHistogram', datasource=self.randoms, weight_col='Weight', fsky=fsky)
            
            # run the algorithm
            # exceptions might not be fatal, if Nbar is provided in 
            # the input DataSources
            try:
                zbins, z_cen, nz = nz_computer.run()
                self.randoms_nbar = spline(z_cen, nz)
            except DataSource.MissingColumn:
                self.randoms_nbar = None
            except Exception as e:
                warn = "an exception occurred while computing randoms n(z)"
                logger.warning(warn)
                warnings.warn(warn, RuntimeWarning)
                self.randoms_nbar = None
    
    def _put_randoms_in_box(self):
        """
        Internal function to put the :attr:`randoms` DataSource in a Cartesian box
        
        This function updates two necessary attribues:
        
        1. :attr:`BoxSize` : array_like, (3,)
            if not input the user, the BoxSize in each direction is computed from
            the maximum extent of the Cartesian coordinates of the :attr:`randoms`
            DataSource
        2. :attr:`mean_coordinate_offset`: array_like, (3,)
            the mean coordinate value in each direction; this is used to re-center
            the Cartesian coordinates of the :attr:`data` and :attr:`randoms`
            to the range of ``[-BoxSize/2, BoxSize/2]``
        """
        # need to compute cartesian min/max
        pos_min = numpy.array([numpy.inf]*3)
        pos_max = numpy.array([-numpy.inf]*3)
        
        # compute global min/max of randoms
        for [pos] in self.randoms_stream.read(['Position'], full=False):
            if len(pos):
                
                # global min/max of cartesian coordinates
                pos_min = numpy.minimum(pos_min, pos.min(axis=0))
                pos_max = numpy.maximum(pos_max, pos.max(axis=0))

        # gather everything to root
        pos_min = self.comm.gather(pos_min)
        pos_max = self.comm.gather(pos_max)
        
        # rank 0 setups up the box and computes nbar (if needed)
        if self.comm.rank == 0:
            
            # find the global coordinate minimum and maximum
            pos_min   = numpy.amin(pos_min, axis=0)
            pos_max   = numpy.amax(pos_max, axis=0)
            
            # used to center the data in the first cartesian quadrant
            delta = abs(pos_max - pos_min)
            self.mean_coordinate_offset = 0.5 * (pos_min + pos_max)
        
            # BoxSize is padded diff of min/max coordinates
            if self.BoxSize is None:
                delta *= 1.0 + self.BoxPad
                self.BoxSize = numpy.ceil(delta) # round up to nearest integer
        else:
            self.mean_coordinate_offset = None
            
        # broadcast the results that rank 0 computed
        self.BoxSize                = self.comm.bcast(self.BoxSize)
        self.mean_coordinate_offset = self.comm.bcast(self.mean_coordinate_offset)
    
        # log some info
        if self.comm.rank == 0:
            logger.info("BoxSize = %s" %str(self.BoxSize))
            logger.info("cartesian coordinate range: %s : %s" %(str(pos_min), str(pos_max)))
            logger.info("mean coordinate offset = %s" %str(self.mean_coordinate_offset))
    
    def _get_nbar_array(self, name, stream, redshift, nbar):
        """
        Get the number density array, sorting through the various possibilities. 
        
        The possibilities are:
        
            1. :attr:`nbar` is not ``None``, and n(z) will be computed from 
            the input function, using the 'Redshift' column, or just returning
            a constant n(z) value
            2. otherwise, the n(z) is computed from the randoms galaxies
        
        Parameters
        ----------
        name : {'data', 'randoms'}
            the name of the stream we are reading from 
        stream : DataStream
            the actual stream we are reading from
        redshift : array_like
            the "Redshift" array that has been read already; this could be an
            array of default values 
        nbar : array_like
            the "Nbar" array that has been read already; this could be an array
            of default values
        
        Returns
        -------
        nbar : array_like
            the final number density array
        """
        # determine if we have default redshift, nbar arrays
        default_z = stream.isdefault('Redshift', redshift)
        default_nbar = stream.isdefault('Nbar', nbar)
        
        # need to compute n(z) from redshift
        if self.nbar is not None:
            
            # fail if we need redshift and don't have it
            if self.nbar.need_redshift and default_z:
                cls = self.__class__.__name__
                raise DataSource.MissingColumn("`n(z)` calculation requires redshift "
                                                "but '%s' DataSource in %s does not support `Redshift` column" %(name, cls))
            
            # get the array of nbar (len == len(redshift))
            nbar = self.nbar(redshift)
                
        # no number density provided -- so use n(z) from randoms
        elif default_nbar:
            
            # crash if we don't have redshift
            if default_z:
                cls = self.__class__.__name__
                raise DataSource.MissingColumn("`n(z)` calculation requires redshift "
                                                "but '%s' DataSource in %s does not support `Redshift` column" %(name, cls))
        
            # need an explicit 'fsky' to compute the volume in n(z)
            if not hasattr(self, 'fsky'):
                raise ValueError("computing `n(z)` from 'randoms' DataSource, but no `fsky` "
                                 "provided for volume calculation")
            
            # if randoms nbar should have been computed already
            if self.randoms_nbar is None:
                raise ValueError("error computing `n(z)` from 'randoms' DataSource -- "
                                 "perhaps 'randoms' DataSource does not support `Redshift` column?")
                
            # normalize the randoms number density to match data number density
            nbar = self.randoms_nbar(redshift) * self.alpha
    
        # if we read number density column for randoms, scale by alpha
        elif name == 'randoms':
            
            # print a warning if we read n(z) for randoms directly
            if self.comm.rank == 0:
                logger.warning(('n(z) read from randoms file; this should give the expected n(z) for the data, '
                                'i.e., we are NOT scaling this n(z) by alpha'))
            #nbar = nbar * self.alpha
            
        
        return nbar
        
        
    #--------------------------------------------------------------------------
    # main functions
    #--------------------------------------------------------------------------
    def __enter__ (self):
        """
        This function is called when the `with` statement is used
        """
        self.open()
        
    def __exit__ (self, exc_type, exc_value, traceback):
        """
        This function is called when a code block using the `with` statement ends
        """
        self.close()
        
    def open(self):
        """
        The FKPCatalog class is designed to be used as a context manager, and 
        this function serves to 'open' the catalog.
        
        This function performs the following actions:
        
            -   open the streams for the :attr:`data` and :attr:`randoms` DataSources,
                which allows them to be read from
            -   if needed, compute the Cartesian BoxSize using the maximum extent of the
                Cartesian coordinates of the :attr:`randoms` DataSource
            -   compute the :attr:`mean_coordinate_offset` from the :attr:`randoms` 
                DataSource, which is used to re-center the Cartesian coordinates of 
                the :attr:`data` and :attr:`randoms` to the range of ``[-BoxSize/2, BoxSize/2]``
            -   if :attr:`nbar` is ``None``, compute the :math:`n(z)` for the :attr:`randoms`, 
                using the :class:`~nbodykit.plugins.algorithms.RedshiftHistogram.RedshiftHistogramAlgorithm`
        
        The desired usage for this function is
        
        .. code:: python
            
            # initialize the catalog
            catalog = FKPCatalog(data, randoms, **kwargs)
            
            # open the catalog
            with catalog:
                ...
        
        In the above snippet, the ``with`` statement automatically calls 
        the ``open`` function. 
        """
        # do nothing if we are already open
        if not self.closed: return 
            
        # open the input data and random streams
        self.data_stream = self.data.open(defaults=self.default_columns)
        self.randoms_stream = self.randoms.open(defaults=self.default_columns)
                
        # compute coordinate statistics of the randoms
        self._put_randoms_in_box()
        
        # compute the n(z) of the randoms
        self._compute_randoms_nbar()
        
    def close(self):
        """
        Close an 'open' FKPCatalog
        
        This performs the following actions: 
            
            1.  close the streams of the :attr:`data` and :attr:`randoms` DataSource objects
            2.  delete the stream attributes, so any cache of the DataSource objects will 
                be automatically deleted, to free up memory
        """
        if hasattr(self, 'data_stream'):
            self.data_stream.close()
            del self.data_stream
            del self.alpha
            
        if hasattr(self, 'randoms_stream'):
            self.randoms_stream.close()
            del self.randoms_stream
            del self.alpha
                                    
    def read(self, name, columns, full=False):
        """
        Read columns from either :attr:`data` or :attr:`randoms`, which is 
        specified by the `name` argument
        
        Parameters
        ----------
        name : {'data', 'randoms'}
            which DataSource to read the columns from
        columns : list
            the list of the names of the columns to read
        full : bool, optional
            if `True`, ignore any `bunchsize` parameters when reading
            from the specified DataSource
        
        Returns
        -------
        list of arrays
            the list of the data arrays for each column in ``columns``
        """ 
        if self.closed:
            raise ValueError("'read' operation on a closed FKPCatalog")
              
        # check valid columns
        valid = ['Position'] + list(self.default_columns.keys())
        if any(col not in valid for col in columns):
            cls = self.__class__.__name__
            raise DataSource.MissingColumn("valid `columns` to read from %s: %s" % (cls, str(valid)))
             
        # set the stream we want to read from                 
        if name == 'data':
            stream = self.data_stream
        elif name == 'randoms':
            stream = self.randoms_stream
        else:
            cls = self.__class__.__name__
            raise ValueError("stream name for %s must be 'data' or 'randoms'" %cls)
    
        # read from the stream
        columns0 = ['Position', 'Redshift', 'Nbar', 'FKPWeight', 'Weight']
        for [coords, redshift, nbar, fkp_weight, comp_weight] in stream.read(columns0, full=full):
        
            # re-centered cartesian coordinates (between -BoxSize/2 and BoxSize/2)
            pos = coords - self.mean_coordinate_offset
            
            # enforce that position is between (-L/2, L/2)
            lim = (pos < -0.5*self.BoxSize)|(pos > 0.5*self.BoxSize)
            if lim.any():
                out_of_bounds = lim.any(axis=1).sum()
                args = (out_of_bounds, name, self.BoxSize)
                warn = ("%d '%s' particles have positions outside of the box when using a BoxSize of %s" %args)
                logger.warning(warn)
                logger.warning(("the positions of out-of-bounds particles are periodically wrapped into "
                                 "the box domain -- the resulting behavior is undefined"))
                warnings.warn(warn, RuntimeWarning)

            # get the number density array
            nbar = self._get_nbar_array(name, stream, redshift, nbar)

            # update the weights with new FKP
            if self.compute_fkp_weights:
                if self.P0_fkp is None:
                    raise ValueError("if 'compute_fkp_weights' is set, please specify a value for 'P0_fkp'")
                fkp_weight = 1. / (1. + nbar*self.P0_fkp)
            
            P = {}
            P['Position']  = pos
            P['Nbar']      = nbar
            P['Redshift']  = redshift
            P['FKPWeight'] = fkp_weight
            P['Weight']    = comp_weight
            
            yield [P[key] for key in columns]
            
    def paint(self, pm, paintbrush='cic'):
        """
        Paint the FKP weighted density field: ``data - alpha*randoms`` using
        the input `ParticleMesh`
        
        The are two different weights that enter into the painting procedure:
        
            1.  **completeness weights**: these weight each number density 
                field for data and randoms seperately
            2.  **FKP weights**: these weight the total FKP density field, i.e., 
                they weight ``data - alpha*randoms``
        
        Parameters
        ----------
        pm : ParticleMesh
            the particle mesh instance to paint the density field to
        
        Returns
        -------
        stats : dict
            a dictionary of FKP statistics, including total number, normalization,
            and shot noise parameters (see equations 13-15 of Beutler et al. 2013)
        """  
        from pmesh.pm import RealField
        
        if self.closed:
            raise ValueError("'paint' operation on a closed FKPCatalog")
            
        # setup
        columns = ['Position', 'Nbar', 'FKPWeight', 'Weight']
        stats = {}
        A_ran = A_data = 0.
        S_ran = S_data = 0.
        N_ran = N_data = 0.
        W_ran = W_data = 0.
        
        # clear the density mesh
        real = RealField(pm)
        real[:] = 0
        
        # alpha determined from completeness weights
        alpha = self.alpha
                
        # paint -1.0*alpha*N_randoms
        for [position, nbar, fkp_weight, comp_weight] in self.read('randoms', columns):
            
            # total weight each galaxy gets is the product of FKP 
            # weight and completeness weight
            weight = fkp_weight * comp_weight
            
            self.painter.basepaint(real, position, weight=weight, paintbrush=paintbrush)
            A_ran += (nbar*comp_weight*fkp_weight**2).sum()
            N_ran += len(position)
            S_ran += (weight**2).sum()
            W_ran += comp_weight.sum()
            
        # randoms get -alpha factor
        real[:] *= -alpha

        A_ran = self.comm.allreduce(A_ran)
        N_ran = self.comm.allreduce(N_ran)
        S_ran = self.comm.allreduce(S_ran)
        W_ran = self.comm.allreduce(W_ran)

        # paint the data
        for [position, nbar, fkp_weight, comp_weight] in self.read('data', columns):
            
            # total weight each galaxy gets is the product of FKP 
            # weight and completeness weight
            weight = fkp_weight * comp_weight
            
            self.painter.basepaint(real, position, weight=weight, paintbrush=paintbrush)
            A_data += (nbar*comp_weight*fkp_weight**2).sum()
            N_data += len(position) 
            S_data += (weight**2).sum()
            W_data += comp_weight.sum()
            
        A_data = self.comm.allreduce(A_data)
        N_data = self.comm.allreduce(N_data)
        S_data = self.comm.allreduce(S_data)
        W_data = self.comm.allreduce(W_data)

        # store the stats (see equations 13-15 of Beutler et al 2013)
        # see equations 13-15 of Beutler et al 2013
        stats['W_data'] = W_data; stats['W_ran'] = W_ran
        stats['N_data'] = N_data; stats['N_ran'] = N_ran
        stats['A_data'] = A_data; stats['A_ran'] = A_ran
        stats['S_data'] = S_data; stats['S_ran'] = S_ran
        stats['alpha'] = alpha
        
        stats['A_ran'] *= alpha
        stats['S_ran'] *= alpha**2
        stats['shot_noise'] = (stats['S_ran'] + stats['S_data'])/stats['A_ran'] # the final shot noise estimate for monopole
        
        # go from number to number density
        real[:] *= numpy.product(pm.Nmesh/pm.BoxSize)
        
        return real, stats
    
        
