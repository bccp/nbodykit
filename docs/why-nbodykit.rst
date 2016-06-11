Overview: why nbodykit?
=======================

nbodykit aims to take advantage of the abundance and availability 
of large-scale computing resources by providing a massively-parallel
toolkit to tackle a wide range of problems that arise in the 
analysis of large-scale structure datasets. 

A major goal of the project is to provide a unified treatment of 
both simulation and observational datasets, allowing nbodykit to 
be used in the analysis of not only N-body simulations, but also
survey data from current and future large-scale structure 
experiments.

nbodykit implements a framework that
insulates analysis algorithms from data containers by relying
on **plugins** that interact with the core of the code base through
distinct **extension points**. Such a framework allows the
user to create plugins designed for a specific task,
which can then be easily loaded by nbodykit, provided the 
plugin implements the minimal interface required by the 
desired extension point.  

We provide several built-in plugins for data containers and
algorithms, as well as more-detailed instructions on how
users can write their own plugins.

Extension Points
----------------

The extension points defined so far are:


**DataSource** (:class:`nbodykit.extensionpoints.DataSource`)
    Data containers
**Algorithm** (:class:`nbodykit.extensionpoints.Algorithm`)
    Algorithms
**Painter** (:class:`nbodykit.extensionpoints.Painter`)
    Painters
**Transfer** (:class:`nbodykit.extensionpoints.Transfer`)
    Transfer functions

Plugins
-------