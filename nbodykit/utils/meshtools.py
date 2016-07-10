import numpy

class MeshSlab(object):
    """
    A convenience class to represent a specific y-z plane of a mesh, 
    which is denoted as a ``slab``
    """
    def __init__(self, islab, x):
        """
        Parameters
        ----------
        islab : int, [0, x[0].shape[0]]
            the index of the slab, which indexes the first dimension of 
            the mesh (the `x` coordinate), thus producing a y-z plane
        x : list of arrays
            the coordinate arrays of the mesh, with proper 3D shapes
            for easy broadcasting; if the mesh has size (Nx, Ny, Nz), 
            then the shapes of `x` are: [(Nx, 1, 1), (1, Ny, 1), (1, 1, Nz)]
        """
        self.index = islab
        self._x = x
        
    def __str__(self):
        name = self.__class__.__name__
        return "<%s: index=%d>" %(name, self.index)
        
    def __repr__(self):
        return self.__str__()
    
    def xi(self, i):
        """
        Return the ``x`` array for dimension ``i`` on this slab, 
        
        .. note:: 
            
            The return value will be properly squeezed for easy
            broadcasting, i.e., if `i` is 0, then an array of shape `(1,1)`
            array is returned, otherwise, the shape is `(Ny, 1)` or
            `(1, Nz)`
        
        Parameters
        ----------
        i : int, {0,1,2}
            the index of the desired dimension
        
        Returns
        -------
        x : array_like
            the coordinate array for dimension `i` on the slab; see the 
            note about the shape of the return array for details
        """
        return self._x[i][0] if i != 0 else self._x[i][self.index]
    
    def x(self):
        """
        The `x` value defined at each point on the slab, where 
        `x` defines the three coordinate arrays of the mesh 
        
        Returns
        -------
        x : array_like, (Ny, Nz)
            the `x` value at each point in the slab
        """
        return self.xsq()**0.5
        
    def xsq(self):    
        """
        The square of :func:`x` defined at each point on the slab. 
        
        This broadcasts the coordinate arrays along each dimension
        to return the square of `x` at each point in the plane:
        
        ..code:: 
            
            xsq = slab.xi(0)**2 + slab.xi(1)**2 + slab.xi(2)**2
        
        Returns
        -------
        xsq : array_like, (Ny, Nz)
            the square of `x` at each point in the slab
        """
        return self.xi(0)**2 + self.xi(1)**2 + self.xi(2)**2


    def mu(self, los_index):
        """
        The `mu` value defined at each point on the slab for the
        specified line-of-sight index
        
        This is defined such that it returns:
        
        ..code:: 
            
            mu = slab.xi(los_index) / slab.x()
        
        Parameters
        ----------
        los_index: int, {0, 1, 2}
            the index defining the line-of-sight, which `mu` is defined
            with respect to
        
        Returns
        -------
        mu : array_like, (Ny, Nz)
            the `mu` value at each point in the slab
        """
        with numpy.errstate(invalid='ignore'):            
            return self.xi(los_index) / self.x()


class MeshWorker(object):
    """
    A helper class to iterate over a coordinate mesh, applying a specified
    function slab-by-slab while iterating
    
    The class is iterable and returns a `MeshSlab` for each index in the
    first dimension of the mesh, i.e.,
    
    ..code:: 
    
        for slab in MeshWorker(x):
            ...
    """        
    def __init__(self, x):
        """
        Parameters
        ----------
        x : list of arrays
            the coordinate arrays of the mesh, with proper 3D shapes
            for easy broadcasting; if the mesh has size (Nx, Ny, Nz), 
            then the shapes of `x` are: [(Nx, 1, 1), (1, Ny, 1), (1, 1, Nz)]
        """    
        self.x = x
        
    def __iter__(self):
        """
        Iterate over the mesh, returning a :class:`MeshSlab` instance
        for each index in the 1st dimension of the mesh
        """
        for islab in range(len(self.x[0])):
            yield MeshSlab(islab, self.x)
            
    def apply(self, f):
        """
        Apply the input function to the mesh, by iterating over
        each slab in the mesh and calling `f`
        
        While iterating, the `MeshSlab` instance will be passed
        to the function `f`
        
        Parameters
        ----------
        f : callable
            a function that should take a MeshSlab instance as its only 
            positional argument
        """
        for slab in self: f(slab)
            
    