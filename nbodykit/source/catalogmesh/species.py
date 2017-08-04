from nbodykit.base.catalogmesh import CatalogMesh
from nbodykit.utils import attrs_to_dict

import numpy
import logging
import contextlib

class MultipleSpeciesCatalogMesh(CatalogMesh):
    """
    A subclass of :class:`~nbodykit.base.catalogmesh.CatalogMesh`
    designed to paint the density field from a sum of multiple types
    of particles.

    The :func:`paint` function paints the density field summed over
    all particle species.

    Parameters
    ----------
    source : CatalogSource
        the input catalog that we wish to interpolate to a mesh
    BoxSize :
        the size of the box
    Nmesh : int, 3-vector
        the number of cells per mesh side
    dtype : str
        the data type of the values stored on mesh
    weight : str
        column in ``source`` that specifies the weight value for each
        particle in the ``source`` to use when gridding
    value : str
        column in ``source`` that specifies the field value for each particle;
        the mesh stores a weighted average of this column
    selection : str
        column in ``source`` that selects the subset of particles to grid
    position : str, optional
        column in ``source`` specifying the position coordinates; default
        is ``Position``
    """
    logger = logging.getLogger('MultipleSpeciesCatalogMesh')

    def __init__(self, source, BoxSize, Nmesh, dtype,
                    weight, value, selection, position='Position'):

        from nbodykit.source.catalog import MultipleSpeciesCatalog
        if not isinstance(source, MultipleSpeciesCatalog):
            raise TypeError(("the input source for MultipleSpeciesCatalogMesh "
                             "must be a MultipleSpeciesCatalog"))

        CatalogMesh.__init__(self, source, BoxSize, Nmesh, dtype, weight,
                            value, selection, position=position)

        # store the column names
        self._colnames = {'position':position, 'weight':weight, 'value':value,
                          'selection':selection}

    @contextlib.contextmanager
    def _set_species(self, name):
        """
        Internal context manager to set the appropriate
        column names needed for painting, based on the input ``name``.

        Parameters
        ----------
        name : str
            the name of one of the species, which determines which columns
            to use when painting to the mesh.
        """
        # simple sanity check
        if name not in self.source.species:
            raise ValueError("'name' should be one of: %s" %str(self.source.species))

        # update the column names to "species/name"
        for col in self._colnames:
            setattr(self, col, name+'/'+self._colnames[col])

        yield # do not need to yield anything

        # restore the original values
        for col in self._colnames:
            setattr(self, col, self._colnames[col])

    def to_real_field(self, normalize=True):
        """
        Paint the density field holding the sum of all particle species,
        returning a :class:`~pmesh.pm.RealField` object.

        Meta-data computed for each particle is stored in the :attr:`attrs`
        attribute of the returned RealField, with keys that are prefixed by
        the species name.

        Parameters
        ----------
        normalize : bool, optional
            if ``True``, normalize the density field as :math:`1+\delta`,
            dividing by the total mean number of objects per cell, as given
            by the ``num_per_cell`` meta-data value in :attr:`attrs`

        Returns
        -------
        RealField :
            the RealField holding the painted density field
        """
        # track the sum of the mean number of objects per cell across species
        attrs = {'num_per_cell':0.}

        # initialize an empty real field
        real = self.pm.create(mode='real', zeros=True)

        # loop over each species
        for name in self.source.species:
            with self._set_species(name):

                # paint the un-normalized density field for this species
                real = CatalogMesh.to_real_field(self, out=real, normalize=False)

                # add to the mean number of objects per cell
                attrs['num_per_cell'] += real.attrs['num_per_cell']

                # store the meta-data for this species, with a prefix
                attrs.update(attrs_to_dict(real, name+'.'))

        # # normalize the field by nbar -> this is now 1+delta
        if normalize:
            real[:] /= attrs['num_per_cell']

        # update the meta-data
        real.attrs.clear()
        real.attrs.update(attrs)

        # add the total shot noise
        nbar = attrs['num_per_cell'] * numpy.prod(self.pm.Nmesh)
        real.attrs['shotnoise'] = numpy.prod(self.pm.BoxSize) / nbar

        return real
