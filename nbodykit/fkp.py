import numpy
import logging
import os
from contextlib import contextmanager
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from nbodykit.extensionpoints import DataSource, Painter, algorithms
from nbodykit.distributedarray import GatherArray

logger = logging.getLogger('FKPCatalog')

class FKPCatalog(object):
    """
    A `DataSource` representing a catalog of tracer objects, 
    designed to be used in analysis similar to that first outlined 
    by Feldman, Kaiser, and Peacock (FKP) 1994 (astro-ph/9304022)
    
    In particular, the `FKPCatalog` uses a catalog of random objects
    to define the mean density of the survey, in addition to the catalog
    of data objects
    
    
    Attributes
    ----------
    data: DataSource
        a `DataSource` that returns the position, weight, etc of the 
        true tracer objects, whose intrinsic clustering is non-zero
    randoms: DataSource
        a `DataSource` that returns the position, weight, etc 
        of a catalog of objects generated randomly to match the
        survey geometry and whose instrinsic clustering is zero
    BoxSize:
        the size of the cartesian box -- the Cartesian coordinates
        of the input objects are computed using the input cosmology,
        and then placed into the box
    mean_coordinate_offset: 
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
        Finalize by performing several steps:
        
            1. if `BoxSize` not provided, infer the value 
               from the Cartesian coordinates of the `data` catalog
            2. compute the mean coordinate offset for each 
               Cartesian dimension -- used to re-center the 
               coordinates to the [-BoxSize/2, BoxSize/2] domain
            3. compute the number density as a function of redshift
               from the `data` and store a spline
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

        # optional configuration
        self.BoxSize             = BoxSize
        self.BoxPad              = BoxPad
        self.compute_fkp_weights = compute_fkp_weights
        self.P0_fkp              = P0_fkp
        self.nbar                = nbar
        self.fsky                = fsky
        
        # default painter
        self.painter = Painter.create('DefaultPainter')

    @property
    def fsky(self):
        """
        The sky area fraction (relative to 4pi str); needed for the volume 
        computation when computing `n(z)`
        """
        try:
            return self._fsky
        except:
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
        Update the ``data`` attribute, keeping track of the total number 
        of objects
        """
        try:
            return self._data
        except:
            cls = self.__class__.__name__
            raise AttributeError("'%s' object has no attribute 'data'" %cls)
        
    @data.setter
    def data(self, val):
        """
        Set the data
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
                
                # set and open the new data
                self._data = val
                defaults = {'Redshift':-1., 'Nbar':-1., 'Weight':1.}
                self.data_stream = self.data.open(defaults)
                self.verify_data_size() # verify the size
            
            else:
                self._data = val
    
    def verify_data_size(self):
        """
        Verify the size of the data, setting it if need be
        """            
        # make sure the size is set properly
        try:
            size = self.data.size
        except:
            if hasattr(self, 'data_stream') and not self.data_stream.closed:            
                # compute the total number
                for [Position] in self.data_stream.read(['Position'], full=False):
                    continue
                self.data.size = self.data_stream.nread
                logger.debug("setting `data` size to %d" %self.data.size)
                        
    @property
    def closed(self):
        """
        Return `True` if the catalog has been setup and the
        data and random streams are open
        """
        if not hasattr(self, 'data_stream'):
            return True
        elif self.data_stream.closed:
            return True
            
        if not hasattr(self, 'randoms_stream'):
            return True
        elif self.randoms_stream.closed:
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
        Set the number density as a function of redshift n(z)
        """
        if val is not None:
            if isinstance(val, str) and os.path.exists(val):
                try:
                    d = numpy.loadtxt(val)
                    
                    # columns are z, n(z)
                    if d.shape[1] == 2:
                        self._nbar = spline(d[:,0], d[:,1])
                    # columns are z_min, z_max, z_cen, n(z)
                    elif d.shape[1] == 4:
                        self._nbar = spline(d[:,-2], d[:,-1])
                    self._nbar.need_redshift = True
                except:
                    raise ValueError("cannot initialize `n(z)` spline from file '%s'" %val)
            elif numpy.isscalar(val):
                val = float(val)
                self._nbar = lambda: val # return a constant n(z)
                self._nbar.need_redshift = False
            else:
                raise ValueError("error setting n(z) from input value")
        else:
            self._nbar = None
             
    def _define_box(self, coords_min, coords_max):
        """
        Define the Cartesian box by:
        
            * computing the Cartesian coordinates for all objects
            * setting the `BoxSize` attribute, if not provided
            * computing the coorindate offset needed to center the
              data onto the [-BoxSize/2, BoxSize/2] domain
        """   
        # center the data in the first cartesian quadrant
        delta = abs(coords_max - coords_min)
        self.mean_coordinate_offset = 0.5 * (coords_min + coords_max)
        
        # set the box size automatically
        if self.BoxSize is None:
            delta *= 1.0 + self.BoxPad
            self.BoxSize = numpy.ceil(delta) # round up to nearest integer
        
    def open(self):
        """
        Open the catalog by defining the Cartesian box
        and opening the `data` and `randoms` streams
        """
        # open the streams
        defaults = {'Redshift':-1., 'Nbar':-1., 'Weight':1.}
        self.data_stream = self.data.open(defaults=defaults)
        self.randoms_stream = self.randoms.open(defaults=defaults)
                
        # verify data size
        self.verify_data_size()
    
        # need to compute cartesian min/max
        pos_min = numpy.array([numpy.inf]*3)
        pos_max = numpy.array([-numpy.inf]*3)
        
        # loop over the randoms to make the box
        for [pos] in self.randoms_stream.read(['Position'], full=False):
            if len(pos):
                
                # global min/max of cartesian coordinates
                pos_min = numpy.minimum(pos_min, pos.min(axis=0))
                pos_max = numpy.maximum(pos_max, pos.max(axis=0))

        # set the size, if not set already
        if not hasattr(self.randoms, 'size'):
            self.randoms.size = self.randoms_stream.nread
                
        # gather everything to root
        pos_min   = self.comm.gather(pos_min)
        pos_max   = self.comm.gather(pos_max)
        
        # compute the n(z) of the randoms to get n(z)
        if self.nbar is None:
            fsky = 1. if not hasattr(self, 'fsky') else self.fsky
            nz_computer = algorithms.RedshiftHistogram(self.randoms, fsky=fsky)
            try:
                zbins, z_cen, nz = nz_computer.run()
                self.randoms_nbar = spline(z_cen, nz)
            except DataSource.MissingColumn:
                self.randoms_nbar = None
            except Exception as e:
                logger.warning("exception occurred while computing randoms n(z)")
                self.randoms_nbar = None
        
        # rank 0 setups up the box and computes nbar (if needed)
        if self.comm.rank == 0:
            
            # find the global
            pos_min   = numpy.amin(pos_min, axis=0)
            pos_max   = numpy.amax(pos_max, axis=0)
            
            # setup the box, using randoms to define it
            self._define_box(pos_min, pos_max)
        else:
            self.mean_coordinate_offset = None
            
        # broadcast the results that rank 0 computed
        self.BoxSize                = self.comm.bcast(self.BoxSize)
        self.mean_coordinate_offset = self.comm.bcast(self.mean_coordinate_offset)
        
        if self.comm.rank == 0:
            logger.info("BoxSize = %s" %str(self.BoxSize))
            logger.info("cartesian coordinate range: %s : %s" %(str(pos_min), str(pos_max)))
            logger.info("mean coordinate offset = %s" %str(self.mean_coordinate_offset))
            
    def close(self):
        """
        Close the FKPCatalog by close the `data` and `randoms` streams
        """
        if hasattr(self, 'data_stream'):
            self.data_stream.close()
            del self.data_stream
        if hasattr(self, 'randoms_stream'):
            self.randoms_stream.close()
            del self.randoms_stream
                
    def __enter__ (self):
        if self.closed:
            self.open()
        
    def __exit__ (self, exc_type, exc_value, traceback):
        self.close()
                    
    def read(self, name, columns, full=False):
        """
        Read data from `stream`, which is specified by the `name` argument
        """   
        # check valid columns
        valid = ['Position', 'Weight', 'Nbar']
        if any(col not in valid for col in columns):
            raise DataSource.MissingColumn("valid `columns` to read from FKPCatalog: %s" %str(valid))
                             
        if name == 'data':
            stream = self.data_stream
        elif name == 'randoms':
            stream = self.randoms_stream
        else:
            raise ValueError("stream name for FKPCatalog must be 'data' or 'randoms'")
    
        # read position, redshift, and weights from the stream
        columns0 = ['Position', 'Redshift', 'Weight', 'Nbar']
        for [coords, redshift, weight, nbar] in stream.read(columns0, full=full):
        
            # determine if we have unique redshift, nbar arrays
            default_z = stream.isdefault('Redshift', redshift)
            default_nbar = stream.isdefault('Nbar', nbar)
        
            # recentered cartesian coordinates (between -BoxSize/2 and BoxSize/2)
            pos = coords - self.mean_coordinate_offset
            
            # enforce that position is between (-L/2, L/2)
            lim = (pos < -0.5*self.BoxSize)|(pos > 0.5*self.BoxSize)
            if lim.any():
                out_of_bounds = lim.any(axis=1).sum()
                args = (out_of_bounds, name, self.BoxSize)
                errmsg = ("%d '%s' particles have positions outside of the box when using a BoxSize of %s" %args)
                logger.warning(errmsg)
                logger.warning(("the positions of out-of-bounds particles are periodically wrapped into "
                                 "the box domain -- the resulting behavior is undefined"))

            # number density from redshift
            if self.nbar is not None:
                if self.nbar.need_redshift:
                    if default_z:
                        raise ValueError("`n(z)` calculation requires redshift "
                                         "but '%s' DataSource does not support `Redshift` column" %name)
                    nbar = self.nbar(redshift)
                else:
                    # nbar is a single value
                    nbar = numpy.array(self.nbar())
                    nbar = numpy.lib.stride_tricks.as_strided(nbar, (len(pos), nbar.size), (0, nbar.itemsize))
                    nbar = nbar.squeeze()
                    
            elif default_nbar:
                if default_z:
                    raise ValueError("`n(z)` calculation requires redshift "
                                     "but '%s' DataSource does not support `Redshift` column" %name)
            
                if not hasattr(self, 'fsky'):
                    raise ValueError("computing `n(z)` from 'randoms' DataSource, but no `fsky` "
                                     "provided for volume calculation")
                
                if self.randoms_nbar is None:
                    raise ValueError("`n(z)` calculation requires redshift "
                                     "but 'randoms' DataSource does not support `Redshift` column")
                    
                alpha = 1.*self.data.size/self.randoms.size
                nbar = self.randoms_nbar(redshift) * alpha
        
            elif name == 'randoms':
                alpha = 1.*self.data.size/self.randoms.size
                nbar = nbar * alpha
                
            # update the weights with new FKP
            if self.compute_fkp_weights:
                if self.P0_fkp is None:
                    raise ValueError("if 'compute_fkp_weights' is set, please specify a value for 'P0_fkp'")
                weight = 1. / (1. + nbar*self.P0_fkp)
            
            P = {}
            P['Position'] = pos
            P['Weight']   = weight
            P['Nbar']     = nbar
        
            yield [P[key] for key in columns]
            
    def paint(self, pm):
        """
        Paint the FKP weighted density field: ``data - alpha*randoms`` using
        the input `ParticleMesh`
        
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
        if self.closed:
            raise ValueError("'paint' operation on a closed FKPCatalog")
            
        # setup
        columns = ['Position', 'Weight', 'Nbar']
        stats = {}
        A_ran = A_data = 0.
        S_ran = S_data = 0.
        N_ran = N_data = 0
        
        # clear the density mesh
        pm.clear()
        
        # alpha determined from size of data sources
        alpha = 1.*self.data.size/self.randoms.size
        
        # paint -1.0*alpha*N_randoms
        for [position, weight, nbar] in self.read('randoms', columns):
            Nlocal = self.painter.basepaint(pm, position, -alpha*weight)
            A_ran += (nbar*weight**2).sum()
            N_ran += Nlocal
            S_ran += (weight**2).sum()

        A_ran = self.comm.allreduce(A_ran)
        N_ran = self.comm.allreduce(N_ran)
        S_ran = self.comm.allreduce(S_ran)
        
        if N_ran != self.randoms.size:
            args = (N_ran, self.randoms.size)
            raise ValueError("`size` mismatch when painting: `N_ran` = %d, `randoms.size` = %d" %args)

        # paint the data
        for [position, weight, nbar] in self.read('data', columns):
            Nlocal = self.painter.basepaint(pm, position, weight)
            A_data += (nbar*weight**2).sum()
            N_data += Nlocal 
            S_data += (weight**2).sum()
            
        A_data = self.comm.allreduce(A_data)
        N_data = self.comm.allreduce(N_data)
        S_data = self.comm.allreduce(S_data)
        
        if N_data != self.data.size:
            args = (N_data, self.data.size)
            raise ValueError("`size` mismatch when painting: `N_data` = %d, `data.size` = %d" %args)

        # store the stats (see equations 13-15 of Beutler et al 2013)
        # see equations 13-15 of Beutler et al 2013
        stats['N_data'] = N_data; stats['N_ran'] = N_ran
        stats['A_data'] = A_data; stats['A_ran'] = A_ran
        stats['S_data'] = S_data; stats['S_ran'] = S_ran
        stats['alpha'] = alpha
        
        stats['A_ran'] *= alpha
        stats['S_ran'] *= alpha**2
        stats['shot_noise'] = (S_ran + S_data)/A_ran # the final shot noise estimate for monopole
        
        return stats
    
        
