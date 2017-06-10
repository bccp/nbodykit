.. _extending-nbodykit:

Extending nbodykit
==================

One of the goals of the extension point and plugin framework used by nbodykit
is to allow the user to easily extend the main code base with new plugins. In this
section, we'll describe the details of implementing plugins for the 4 built-in
extension points: :class:`~nbodykit.core.Algorithm`,
:class:`~nbodykit.core.DataSource`, :class:`~nbodykit.core.Painter`,
and :class:`~nbodykit.core.Transfer`.

To define a plugin:

    1.  Subclass from the desired extension point class.

    2.  Define a class method `fill_schema` that declares the relevant attributes by calling :func:`~nbodykit.plugins.fromfile.ConstructorSchema.add_argument` of the class's :class:`~nbodykit.plugins.fromfile.ConstructorSchema`, which is stored as the `schema` attribute.

    3.  Define a `plugin_name` class attribute.

    4.  Define the functions relevant for that extension point interface.

.. ipython:: python
    :suppress:

    from nbodykit.core import Algorithm, DataSource, Painter, Transfer

Registering plugins
-------------------

All plugin classes must define a :func:`fill_schema` function, which is necessary for the
core of the nbodykit code to be aware of and use the plugin class. Each plugin carries a
`schema` attribute which is a :class:`nbodykit.plugins.fromfile.ConstructorSchema` that is
responsible for storing information regarding the parameters needed to initialize
the plugin. The main purpose of :func:`fill_schema` is to update this schema object
for each argument of a plugin's :func:`__init__`.

As an example of how this is done, we can examine :func:`__init__` and
:func:`fill_schema` for the
:class:`PlainText <nbodykit.core.datasource.PlainText.PlainTextDataSource>` DataSource:


.. literalinclude:: ../nbodykit/core/datasource/PlainText.py
    :pyobject: PlainTextDataSource.__init__
    :linenos:
    :lineno-match:

.. literalinclude:: ../nbodykit/core/datasource/PlainText.py
    :pyobject: PlainTextDataSource.fill_schema
    :linenos:
    :lineno-match:


A few things to note in this example:

    1. All arguments of :func:`__init__` are added to the class schema via the :func:`~nbodykit.plugins.fromfile.ConstructorSchema.add_argument` function in the `fill_schema` function.
    2. The :func:`~nbodykit.plugins.fromfile.ConstructorSchema.add_argument` function has a calling signature similar to :meth:`argparse.ArgumentParser.add_argument`. The user can specify default values, parameter choices, and type functions used for casting parsed values.
    3. Any default values and whether or not the parameter is required will be directly inferred from the :func:`__init__` calling signature.
    4. Parameters in a plugin's schema will be automatically attached to the class instance before the body of :func:`__init__` is executed -- the user does not need to reattach these attributes. As such, the body of :func:`__init__` in this example is empty. However, additional initialization-related computations could also be performed here.



Extension point interfaces
--------------------------

Below, we provide the help messages for each of the functions that are
required to implement plugins for the 4 built-in extension point types.

Algorithm
~~~~~~~~~

The functions required to implement an Algorithm plugin are:

.. ipython:: python

    # run the algorithm
    help(Algorithm.run)

    # save the result to an output file
    help(Algorithm.save)

DataSource
~~~~~~~~~~

The functions required to implement a DataSource plugin are:

.. ipython:: python

    # read and return all available data columns (recommended for typical users)
    help(DataSource.readall)

    # read data columns, reading data in parallel across MPI ranks
    help(DataSource.parallel_read)


Painter
~~~~~~~

The functions required to implement a Painter plugin are:

.. ipython:: python

    # do the painting procedure
    help(Painter.paint)

Transfer
~~~~~~~~

The functions required to implement a Transfer plugin are:

.. ipython:: python

    # apply the Fourier-space kernel
    help(Transfer.__call__)
