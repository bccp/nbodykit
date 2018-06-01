import numpy
import logging
from pmesh.pm import ParticleMesh, RealField, BaseComplexField
import warnings

class MeshSource(object):
    """
    Base class for a source in the form of data on an input grid.

    The MeshSource object remembers the original source together with
    a sequence of transformations (added via the apply method).

    ``dtype`` is the type of the real numbers, either 'f4' or 'f8'.

    .. note:

        Subclasses of this class must implement either :func:`to_real_field`
        or :func:`to_complex_field`. The methods must return either a
        :class:`pmesh.pm.RealField` or a :class:`pmesh.pm.ComplexField` object.

    Parameters
    ----------
    comm :
        the global MPI communicator
    Nmesh : int, array_like
        the number of cells per grid size
    BoxSize : array_like
        the size of the box
    dtype : str
        the desired data type of the grid
    """
    logger = logging.getLogger('MeshSource')

    def __init__(self, comm, Nmesh, BoxSize, dtype):

        # ensure self.comm is set, though usually already set by the child.
        self.comm = comm
        self.dtype = dtype

        if Nmesh is None or BoxSize is None:
            raise ValueError("both Nmesh and BoxSize must not be None to initialize ParticleMesh")

        Nmesh = numpy.array(Nmesh)
        if Nmesh.ndim == 0:
            ndim = 3
        else:
            ndim = len(Nmesh)
        _Nmesh = numpy.empty(ndim, dtype='i8')
        _Nmesh[:] = Nmesh
        self.pm = ParticleMesh(BoxSize=BoxSize, Nmesh=_Nmesh,
                                dtype=self.dtype, comm=self.comm)

        self.attrs['BoxSize'] = self.pm.BoxSize.copy()
        self.attrs['Nmesh'] = self.pm.Nmesh.copy()

        # modify the underlying method
        # actions may have been overriden!
        self._actions = []
        self.base = None

    def __finalize__(self, other):
        """
        Finalize the creation of a MeshSource object by copying over
        attributes from a second MeshSource.

        Parameters
        ----------
        other : MeshSource
            the second MeshSource to copy over attributes from
        """
        if isinstance(other, MeshSource):
            self.comm = other.comm
            self.dtype = other.dtype
            self.pm = other.pm
            self.attrs.update(other.attrs)
            self._actions = []
            self._actions.extend(other.actions)

        return self


    def view(self):
        """
        Return a "view" of the MeshSource, in the spirit of
        numpy's ndarray view.

        This returns a new MeshSource whose memory is owned by ``self``.

        Note that for CatalogMesh objects, this is overidden by the
        ``CatalogSource.view`` function.
        """
        view = object.__new__(MeshSource)
        view.base = self
        return view.__finalize__(self)

    @property
    def attrs(self):
        """
        A dictionary storing relevant meta-data about the CatalogSource.
        """
        try:
            return self._attrs
        except AttributeError:
            self._attrs = {}
            return self._attrs

    @property
    def actions(self):
        """
        A list of actions to apply to the density field when interpolating
        to the mesh.

        This stores tuples of ``(mode, func, kind)``; see :func:`apply`
        for more details.
        """
        return self._actions

    def apply(self, func, kind='wavenumber', mode='complex'):
        """
        Return a view of the mesh, with :attr:`actions` updated to
        apply the specified function, either in Fourier space or
        configuration space, based on ``mode``

        Parameters
        ----------
        func : callable
            func(x, y) where x is a list of ``r`` (``k``) values that broadcasts
            into a full array, when ``mode`` is 'real' ('complex');
            the value of x depends on ``kind``. ``y`` is the value of
            the mesh field on the corresponding locations.
        kind : string, optional
            The kind of value in x.

            - When ``mode`` is 'complex':

              - 'wavenumber' means wavenumber from [- 2 pi / L * N / 2, 2 pi / L * N / 2).
              - 'circular' means circular frequency from [- pi, pi).
              - 'index' means [0, Nmesh )

            - When ``mode`` is 'real':

              - 'relative' means distance from [-0.5 Boxsize, 0.5 BoxSize).
              - 'index' means [0, Nmesh )
        mode : 'complex' or 'real', optional
            whether to apply the function to the mesh in configuration space
            or Fourier space

        Returns
        -------
        MeshSource :
            a view of the mesh object with the :attr:`actions` attribute
            updated to include the new action
        """
        assert mode in ['complex', 'real'], "``mode`` should be 'complex' or 'real'"
        if mode == 'real':
            assert kind in ['relative', 'index']
        else:
            assert kind in ['wavenumber', 'circular', 'index']

        view = self.view()
        # modify the underlying method
        # actions may have been overriden!
        view._actions.append((mode, func, kind))
        return view

    def __len__(self):
        """
        Length of a mesh source is zero
        """
        return 0

    def to_real_field(self, out=None, normalize=True):
        """
        Convert the mesh source to the configuration-space field,
        returning a :class:`pmesh.pm.RealField` object.

        Not implemented in the base class, unless object is a view.
        """
        if isinstance(self.base, MeshSource): return self.base.to_real_field()
        return NotImplemented

    def to_complex_field(self, out=None):
        """
        Convert the mesh source to the Fourier-space field,
        returning a :class:`pmesh.pm.ComplexField` object.

        Not implemented in the base class, unless object is a view.
        """
        if isinstance(self.base, MeshSource): return self.base.to_complex_field()
        return NotImplemented

    def to_field(self, mode='real', out=None):
        """
        Return the mesh as a :mod:`pmesh` Field object, either in Fourier
        space or configuration space, based on ``mode``.

        This will call :func:`to_real_field` or :func:`to_complex_field`
        based on ``mode``.

        Parameters
        ----------
        mode : 'real' or 'complex'
            the return type of the field

        Returns
        -------
        :class:`~pmesh.pm.RealField`, :class:`~pmesh.pm.ComplexField` :
            either a RealField of ComplexField, storing the value of the field
            on the mesh
        """
        if mode == 'real':
            real = self.to_real_field()
            if real is NotImplemented:
                complex = self.to_complex_field()
                assert complex is not NotImplemented
                real = complex.c2r(out=Ellipsis)
                if hasattr(complex, 'attrs'):
                    real.attrs = complex.attrs
            var = real
        elif mode == 'complex':
            complex = self.to_complex_field()
            if complex is NotImplemented:
                real = self.to_real_field()
                assert real is not NotImplemented
                complex = real.r2c(out=Ellipsis)
                if hasattr(real, 'attrs'):
                    complex.attrs = real.attrs
            var = complex
        else:
            raise ValueError("mode is either real or complex, %s given" % mode)

        return var

    def compute(self, mode='real', Nmesh=None):
        """
            Compute / Fetch the mesh object into memory as a RealField or ComplexField object.
        """
        return self._paint_XXX(mode=mode, Nmesh=Nmesh)

    def paint(self, mode="real", Nmesh=None):
        warnings.warn("the paint method is deprecated from the Public API. Use .compute() instead.", DeprecationWarning)
        return self._paint_XXX(mode=mode, Nmesh=Nmesh)

    def _paint_XXX(self, mode="real", Nmesh=None):
        """
        Paint the density on the mesh and apply
        any transformation functions specified in :attr:`actions`.

        The return type of the :mod:`pmesh` Field object is specified by
        ``mode``. This calls :func:`to_field` to convert the mesh to a Field.

        See the :ref:`documentation <painting-mesh>` on painting for more
        details on painting catalogs to a mesh.

        Parameters
        ----------
        mode : 'real' or 'complex'
            the type of the returned Field object, either a RealField or
            ComplexField
        Nmesh : int or array_like, or None
            If given and different from the intrinsic Nmesh of the source,
            resample the mesh to the given resolution

        Returns
        -------
        :class:`~pmesh.pm.RealField`, :class:`~pmesh.pm.ComplexField` :
            either a RealField of ComplexField, with the functions in
            :attr:`actions` applied to it
        """
        if not mode in ['real', 'complex']:
            raise ValueError('mode must be "real" or "complex"')

        # add a dummy action to ensure the right mode of return value
        actions = self.actions + [(mode, )]

        # if we expect complex, be smart and use complex directly.
        var = self.to_field(mode=actions[0][0])

        if not hasattr(var, 'attrs'):
            attrs = {}
        else:
            attrs = var.attrs

        for action in actions:
            # ensure var is the right mode

            if action[0] == 'complex':
                if not isinstance(var, BaseComplexField):
                    var = var.r2c(out=Ellipsis)
            if action[0] == 'real':
                if not isinstance(var, RealField):
                    var = var.c2r(out=Ellipsis)

            if len(action) > 1:
                # there is a filter function
                kwargs = {}
                kwargs['func'] = action[1]
                if action[2] is not None:
                    kwargs['kind'] = action[2]
                kwargs['out'] = Ellipsis
                var.apply(**kwargs)

        # FIXME: get rid of this after pmesh 0.1.45+
        # cast to the correct type (transposed complex or real)
        from pmesh.pm import _typestr_to_type

        var = var.cast(type=_typestr_to_type(mode), out=var)
        pm = self.pm.resize(Nmesh)

        if any(pm.Nmesh != self.pm.Nmesh):
            # resample if the output mesh mismatches
            # XXX: this could be done more efficiently.
            var1 = pm.create(type=mode)
            var.resample(out=var1)
            var = var1

            if self.comm.rank == 0:
                self.logger.info('%s resampling from %s to %s done' % (str(self), str(self.pm.Nmesh), str(pm.Nmesh)))

        var.attrs = attrs
        var.attrs.update(self.attrs)

        if self.comm.rank == 0:
            self.logger.info('field: %s painting done' % str(self))

        return var

    def preview(self, axes=None, Nmesh=None, root=0):
        """
        Gather the mesh into as a numpy array, with
        (reduced) resolution. The result is broadcast to
        all ranks, so this uses :math:`\mathrm{Nmesh}^3` per rank.

        Parameters
        ----------
        Nmesh : int, array_like
            The desired Nmesh of the result. Be aware this function
            allocates memory to hold a full ``Nmesh`` on each rank.
        axes : int, array_like
            The axes to project the preview onto., e.g. (0, 1)
        root : int, optional
            the rank number to treat as root when gathering to a single rank

        Returns
        -------
        out : array_like
            An numpy array holding the real density field.
        """
        field = self.to_field(mode='real')
        if Nmesh is None:
            Nmesh = self.pm.Nmesh

        return field.preview(Nmesh, axes=axes)

    def save(self, output, dataset='Field', mode='real'):
        """
        Save the mesh as a :class:`~nbodykit.source.mesh.bigfile.BigFileMesh`
        on disk, either in real or complex space.

        Parameters
        ----------
        output : str
            name of the bigfile file
        dataset : str, optional
            name of the bigfile data set where the field is stored
        mode : str, optional
            real or complex; the form of the field to store
        """
        import bigfile
        import warnings
        import json
        from nbodykit.utils import JSONEncoder

        field = self.compute(mode=mode)

        with bigfile.BigFileMPI(self.pm.comm, output, create=True) as ff:
            data = numpy.empty(shape=field.size, dtype=field.dtype)
            field.ravel(out=data)
            with ff.create_from_array(dataset, data) as bb:
                if isinstance(field, RealField):
                    bb.attrs['ndarray.shape'] = field.pm.Nmesh
                    bb.attrs['BoxSize'] = field.pm.BoxSize
                    bb.attrs['Nmesh'] = field.pm.Nmesh
                elif isinstance(field, BaseComplexField):
                    bb.attrs['ndarray.shape'] = field.Nmesh, field.Nmesh, field.Nmesh // 2 + 1
                    bb.attrs['BoxSize'] = field.pm.BoxSize
                    bb.attrs['Nmesh'] = field.pm.Nmesh

                for key in field.attrs:
                    # do not override the above values -- they are vectors (from pm)
                    if key in bb.attrs: continue
                    value = field.attrs[key]
                    try:
                        bb.attrs[key] = value
                    except ValueError:
                        try:
                            json_str = 'json://'+json.dumps(value, cls=JSONEncoder)
                            bb.attrs[key] = json_str
                        except:
                            warnings.warn("attribute %s of type %s is unsupported and lost while saving MeshSource" % (key, type(value)))
