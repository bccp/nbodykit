.. _cookbook:

The Cookbook
============

Here, we provide a set of recipes detailing a broad selection of the
functionality available in nbodykit. Ranging from simple tasks to
more complex work flows, we hope that these recipes help users become acclimated
with nbodykit as well as illustrate the power of nbodykit for large-scale
structure data analysis.

For users who wish to dive right into the examples, an interactive environment
containing the cookbook recipes is available to users
via the `BinderHub service <https://github.com/jupyterhub/binderhub>`_.
Just click the launch button below to get started!

.. figure:: http://mybinder.org/badge.svg
    :alt: binder
    :target: https://mybinder.org/v2/gh/bccp/nbodykit-cookbook/master?filepath=recipes

.. add a hidden toctree so sphinx doesnt complain
.. toctree::
  :maxdepth: 1
  :hidden:
  :glob:

  *

--------

The (static) recipes below are provided as
`Jupyter notebooks <http://jupyter-notebook.rtfd.io>`_
and are available for download by clicking the "Source" link in the navigation
bar at the top of the page.

Data Recipes
------------

* | :ref:`Generating catalogs from a log-normal density field mocks <cookbook/lognormal-mocks.ipynb>`
  | Demonstrates how to use the :class:`~nbodykit.source.catalog.lognormal.LogNormalCatalog` class, which
    Poisson samples a log-normal density field and applies the Zel'dovich approximation.
* | :ref:`Generating catalogs using a halo occupation distribution <cookbook/hod-mocks.ipynb>`
  | Demonstrates how to use the :class:`~nbodykit.source.catalog.hod.HODCatalog`, which uses the
    :mod:`halotools` package to populate a halo catalog with objects.


.. _cookbook-painting:

Painting Recipes
----------------

* | :ref:`cookbook/painting.ipynb#Painting-the-Overdensity-Field`
  | Demonstrates how to create a mesh object from a catalog and paint the overdensity
    field to the mesh.
* | :ref:`cookbook/painting.ipynb#Painting-the-Line-of-sight-Momentum-Field`
  | Demonstrates how to paint mass-weighted quantities (in this case, the line-of-sight
    momentum field) to a mesh.
* | :ref:`cookbook/painting.ipynb#Painting-Multiple-Species-of-Particles`
  | Demonstrates how to paint the combined overdensity field from different types of
    particles to the same mesh.

Algorithm Recipes
-----------------

* | :ref:`cookbook/fftpower.ipynb`
  | Demonstrates how to use :class:`~nbodykit.algorithms.fftpower.FFTPower` to compute the
    power spectrum of objects in a simulation box.

* | :ref:`cookbook/interlacing.ipynb`
  | Demonstrates the use and effects of the interlacing technique when painting
    density fields to a mesh.

* | :ref:`cookbook/interpolation-windows.ipynb`
  | Demonstrates the different interpolation windows available to use when painting
    and their accuracy.

* | :ref:`cookbook/convpower.ipynb`
  | Demonstrates how to use :class:`~nbodykit.algorithms.convpower.ConvolvedFFTPower` to
    compute the power spectrum multipoles of observational data.

* | :ref:`cookbook/boss-dr12-data.ipynb`
  | Computes the power spectrum multipoles of the DR12 BOSS LOWZ galaxy sample.

* | :ref:`cookbook/angular-paircount.ipynb`
  | Demonstrates how to use :class:`~nbodykit.algorithms.survey_paircount.AngularPairCount`
    to compute the angular correlation function.

Contributing
------------

If you have an application of nbodykit that is concise and interesting,
please consider adding it to our cookbook. We also welcome feedback and
improvements for these recipes. Users can submit issues or open a pull request
on the `nbodykit cookbook repo on GitHub <https://github.com/bccp/nbodykit-cookbook>`_.

Cookbook recipes should be in the form of Jupyter notebooks. See the
`existing recipes <https://github.com/bccp/nbodykit-cookbook/recipes>`_
for examples. The recipes are designed to illustrate interesting uses of
nbodykit for other users to learn from.

We appreciate any and all contributions!
