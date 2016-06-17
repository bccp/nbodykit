Overview
========

nbodykit aims to take advantage of the wealth of large-scale computing 
resources by providing a massively-parallel
toolkit to tackle a wide range of problems that arise in the 
analysis of large-scale structure datasets. 

A major goal of the project is to provide a unified treatment of 
both simulation and observational datasets, allowing nbodykit to 
be used in the analysis of not only N-body simulations, but also
data from current and future large-scale structure surveys.

nbodykit implements a framework that
insulates analysis algorithms from data containers by relying
on **plugins** that interact with the core of the code base through
distinct **extension points**. Such a framework allows the
user to create plugins designed for a specific task,
which can then be easily loaded by nbodykit, provided that the 
plugin implements the minimal interface required by the 
desired extension point.  

We provide several built-in extension points and plugins, which we outline
below. For more detailed instructions on how to add new plugins to 
nbodykit, see :ref:`extending-nbodykit`.

Extension Points
----------------

There are several built-in extension points, which can be found
in the :mod:`nbodykit.extensionpoints` module. These classes serve as 
the mount point for plugins, connecting the core of the nbodykit
package to the individual plugin classes. Each extension point defines
a specific interface that all plugins of that type must implement. 

There are four built-in extension points. Each extension point carries a 
`registry`, which stores all plugins of that type that have been succesfully
loaded by the main nbodykit code.

1. **Algorithm**

    - **location**: :class:`nbodykit.extensionpoints.Algorithm`
    - **registry**: :data:`nbodykit.extensionpoints.algorithms`, :attr:`Algorithm.registry <nbodykit.extensionpoints.Algorithm.registry>`
    - **description**: the mount point for plugins that run one of the high-level algorithms, i.e, a power spectrum calculation or Friends-of-Friends halo finder
    
2. **DataSource**

    - **location**: :class:`nbodykit.extensionpoints.DataSource`
    - **registry**: :data:`nbodykit.extensionpoints.datasources`, :attr:`DataSource.registry <nbodykit.extensionpoints.DataSource.registry>`
    - **description**: the mount point for plugins that refer to the reading of input data files
    
3. **Painter** 

    - **location**: :class:`nbodykit.extensionpoints.Painter`
    - **registry**: :data:`nbodykit.extensionpoints.painters`, :attr:`Painter.registry <nbodykit.extensionpoints.Painter.registry>`
    - **description**: the mount point for plugins that "paint" input data files, where painting refers to the process of gridding a desired quantity on a mesh; the most common example is gridding the density field of a catalog of objects
    
4. **Transfer** 

    - **location**: :class:`nbodykit.extensionpoints.Transfer`
    - **registry**: :data:`nbodykit.extensionpoints.transfers`, :attr:`Transfer.registry <nbodykit.extensionpoints.Transfer.registry>`
    - **description**: the mount point for plugins that apply a kernel to the painted field in Fourier space during power spectrum calculations
    
Plugins
-------

Plugins are subclasses of an extension point that are designed
to handle a specific task, such as reading a certain type of data
file, or computing a specific type of algorithm. 

The core of the nbodykit functionality comes from the built-in plugins, of
which there are numerous. Below, we list each of the built-in plugins and
a brief desciption of the class. For further details, the
name of each plugin provides a link to the API reference for each class.

1. **Algorithms**

.. include:: plugins-list/Algorithm.rst

2. **DataSource**

.. include:: plugins-list/DataSource.rst

3. **Painter**

.. include:: plugins-list/Painter.rst

4. **Transfer**

.. include:: plugins-list/Transfer.rst