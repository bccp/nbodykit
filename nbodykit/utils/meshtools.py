import numpy

class MeshSlab(object):
    """
    A convenience class to represent a specific slab of a mesh, 
    which is denoted as a ``slab``
    """
    def __init__(self, islab, coords, axis, symmetry_axis):
        """
        Parameters
        ----------
        islab : int, [0, x[0].shape[0]]
            the index of the slab, which indexes the first dimension of 
            the mesh (the `x` coordinate), thus producing a y-z plane
        coords : list of arrays
            the coordinate arrays of the mesh, with proper 3D shapes
            for easy broadcasting; if the mesh has size (Nx, Ny, Nz), 
            then the shapes of `x` are: [(Nx, 1, 1), (1, Ny, 1), (1, 1, Nz)]
        axis : int, {0, 1, 2}
            the index of the mesh axis to iterate over
        symmetry_axis : int, optional
            if provided, the axis that has been compressed due to Hermitian symmetry
        """
        self.index       = islab
        self.axis        = axis
        self.symmetry_axis = symmetry_axis
        self._coords     = coords
        
    def __str__(self):
        name = self.__class__.__name__
        return "<%s: axis=%d, index=%d>" %(name, self.axis, self.index)
        
    def __repr__(self):
        return self.__str__()
    
    @property
    def meshshape(self):
        """
        Return the local shape of the mesh on this rank, as determined
        by the input coordinates array
        """
        return [numpy.shape(self._coords[i])[i] for i in [0, 1, 2]]
    
    @property
    def shape(self):
        """
        Return the shape of the slab
        """
        return [s for i, s in enumerate(self.meshshape) if i != self.axis]

    @property
    def hermitian_symmetric(self):
        """
        Whether the slab is Hermitian-symmetric
        """
        return self.symmetry_axis is not None
    
    def coords(self, i):
        """
        Return the coordinate array for dimension ``i`` on this slab, 
        
        .. note:: 
            
            The return value will be properly squeezed for easy
            broadcasting, i.e., if `i` is `self.axis`, then an array of 
            shape `(1,1)` array is returned, otherwise, the shape is 
            `(N_i, 1)` or `(1, N_i)`
        
        Parameters
        ----------
        i : int, {0,1,2}
            the index of the desired dimension
        
        Returns
        -------
        array_like
            the coordinate array for dimension `i` on the slab; see the 
            note about the shape of the return array for details
        """
        if i != self.axis:
            return self._coords[i][self.axis]
        else:
            return self._coords[i][self.index]
            
    def norm2(self):    
        """
        The square of coordinate grid norm defined at each 
        point on the slab. 
        
        This broadcasts the coordinate arrays along each dimension
        to compute the norm at each point in the slab.
                
        Returns
        -------
        array_like, (slab.shape)
            the square of coordinate mesh at each point in the slab
        """
        return self.coords(0)**2 + self.coords(1)**2 + self.coords(2)**2

    def mu(self, los_index):
        """
        The `mu` value defined at each point on the slab for the
        specified line-of-sight index
        
        
        Parameters
        ----------
        los_index: int, {0, 1, 2}
            the index defining the line-of-sight, which `mu` is defined
            with respect to
        
        Returns
        -------
        array_like, (slab.shape)
            the `mu` value at each point in the slab
        """
        with numpy.errstate(invalid='ignore'):            
            return self.coords(los_index) / self.norm2()**0.5

    @property
    def nonsingular(self):
        """
        Return the indices on the slab of the positive frequencies
        along the dimension specified by `symmetry_axis`
        """
        try: 
            return self._nonsingular
        except AttributeError:
            
            # initially, return slice that includes all elements 
            all_slice = slice(None); empty_slice = slice(0, 0)
            idx = [all_slice, all_slice]
            
            # iteration axis is symmetry axis
            if self.symmetry_axis == self.axis:
                
                # check if current iteration value is positive
                if numpy.float(self.coords(self.axis)) <= 0.:
                    idx = [empty_slice, empty_slice] 
                       
            # one of slab dimensions is symmetry axis
            else:
                
                # return indices that have positive freq along symmetry axis
                nonsingular = numpy.squeeze(self._coords[self.symmetry_axis] > 0.)
                idx[self.symmetry_axis-1] = nonsingular
        
            self._nonsingular = idx
            return self._nonsingular

    @property
    def hermitian_weights(self):
        """
        Weights to be applied to quantities on the slab in order
        to account for Hermitian symmetry
        
        These weights double-count the positive frequencies along
        the `symmetry_axis`. 
        """
        try:
            return self._weights
        except AttributeError:
            
            # if not Hermitian symmetric, weights are 1
            if not self.hermitian_symmetric:
                toret = 1.
            # iteration axis is symmetry axis
            elif self.axis == self.symmetry_axis:
                toret = 1.
                if numpy.float(self.coords[self.symmetry_axis]) > 0.:
                    toret = 2.
            # only nonsingular plane gets factor of 2
            else:
                toret = numpy.ones(self.shape, dtype='f4')
                toret[self.nonsingular] = 2.
            
            self._weights = toret
            return self._weights
        
def SlabIterator(coords, axis=0, symmetry_axis=None):
    """
    Iterate over the specified dimension of the coordinate mesh,
    returning a :class:`MeshSlab` for each iteration
    
    Parameters
    ----------
    coords : list of arrays
        the coordinate arrays of the mesh, with proper 3D shapes
        for easy broadcasting; if the mesh has size (Nx, Ny, Nz), 
        then the shapes of `x3d` should be: ``[(Nx, 1, 1), (1, Ny, 1), (1, 1, Nz)]``
    axis : int, optional
        the index of the mesh axis to iterate over
    symmetry_axis : int, optional
        if provided, the axis that has been compressed due to Hermitian symmetry
    """        
    # account for negative axes
    if axis < 0: axis += 3
    if symmetry_axis is not None and symmetry_axis < 0: symmetry_axis += 3
    
    # iterate over the specified axis, slab by slab
    for islab in range(len(coords[axis])):
        yield MeshSlab(islab, coords, axis, symmetry_axis)

    
