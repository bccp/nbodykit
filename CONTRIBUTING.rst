Contributing Guidelines
=======================

We welcome user contributions to nbodykit, but be sure to read this guide first
to make the process as smooth as possible!

.. _local-dev:

Requesting a Feature
--------------------

You can use `GitHub issues <https://github.com/bccp/nbodykit/issues>`_
or send mail to nbodykit-issues@fire.fundersclub.com
to request features you would like to see in nbodykit.

1. Provide a clear and detailed explanation of the feature you want to add
   and its use case in large-scale structure data analysis.
   Keep in mind that the goal is to include features that are useful for
   a wide set of users.

2. If you are able, start writing some code and attempt a pull
   request. Be sure to read the :ref:`submission guidelines <PR-guide>`.
   There are often several features competing for time
   commitments, and user pull requests are greatly appreciated!


Bug Reporting
-------------

If you think you've found a bug in nbodykit, follow these steps to submit a
report:

1. First, double check that your bug isn't already fixed.
   Follow our :ref:`instructions <local-dev>`
   for setting up a local development environment and make sure you've installed
   the tip of the master branch.

2. Search for similar issues `on GitHub <https://github.com/bccp/nbodykit/issues>`_.
   Make sure to delete `is:open` on the issue search to find solved tickets as
   well. It is possible that this bug has been encountered before.

3. Open up an issue on GitHub or send an email to nbodykit-issues@fire.fundersclub.com.

4. Please include the versions of Python, nbodykit, and other dependency
   libraries if they are applicable (numpy, scipy, etc)

5. Provide us with the logging output of the script from when the bug was encountered,
   including a traceback if appropriate. If at all possible, provide us with
   a standalone script that will reproduce the issue. Issues have a much higher chance
   of being resolved quickly if we can easily reproduce the bug.

6. Take a stab at fixing the bug yourself! The :mod:`runtests` module used by
   nbodykit supports on-line debugging via the
   `PDB interface <https://docs.python.org/3/library/pdb.html>`_.  It also
   supports drop-in replacements for PDB such as
   `PDB++ <https://pypi.python.org/pypi/pdbpp/>`_. A common debugging route
   is to add a regression unit test to the nbodykit test suite that fails due
   the bug and then run the test in debugging mode:

   .. code:: bash

      $ python run-tests.py nbodykit/path/to/your/test --pdb

   :ref:`Pull requests <PR-guide>` for bug fixes are always welcome!

We strongly recommend following the above steps and providing as much information
as possible when bugs are encountered. This will help us resolve issues faster --
and get everyone back to doing more accurate science!

Setting up for Local Development
--------------------------------

1. Fork nbodykit_ on GitHub:

   .. code:: bash

      $ git clone https://github.com/bccp/nbodykit.git
      $ cd nbodykit

2. Install the dependencies. We recommend using the
   `Anaconda environment manager <https://www.continuum.io/downloads>`_.
   To create a new environment for nbodykit development, use:

   .. code:: bash

       # create and activate a new conda environment
       $ conda create -n nbodykit-dev python=3
       $ source activate nbodykit-dev

       # install requirements
       $ conda install -c bccp --file requirements.txt --file requirements-extras.txt

3. Install nbodykit in develop mode using ``pip``:

   .. code:: bash

       # install in develop mode
       $ pip install -e .


.. _PR-guide:

Opening a Pull Request
----------------------

1. Write the code implementing your bug fix or feature, making sure to use
   detailed commit messages.

2. Ensure that any new code is properly documented, with docstrings following
   the `NumPy/Scipy documentation style guide <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

3. Write tests of the new code, ensuring that it has full unit test coverage.
   This is a crucial step as new pull requests will not be merged if they
   reduce the overall test coverage of nbodykit.

4. Run the test suite locally. From the main nbodykit directory, run:

   .. code:: bash

      $ python run-tests.py --with-coverage --html-cov

   This will also output the test coverage statistics to ``build/coverage/index.html``.

5. Make sure all of the tests have passed and that the coverage statistics
   indicate that any new code is fully covered by the test suite.

6. Be sure to update the
   `changelog <https://github.com/bccp/nbodykit/blob/master/CHANGES.rst>`_
   to indicate what was added/modified.

7. Submit your pull request to ``nbodykit:master``.
   The `Travis CI <https://travis-ci.org/bccp/nbodykit>`_ build must be passing
   before your pull request can be merged. Additionally, the overall
   coverage of the test suite must not decrease for the pull request to be merged.


.. _contributing_examples:

Contributing to the Cookbook
----------------------------

If you have an application of nbodykit that is concise and interesting,
please consider adding it to our :ref:`cookbook of recipes <cookbook>`.
We also welcome feedback and improvements for these recipes. Users can
submit issues or open a pull request on the
`nbodykit cookbook repo on GitHub <https://github.com/bccp/nbodykit-cookbook>`_.

Cookbook recipes should be in the form of Jupyter notebooks. See the
`existing recipes <https://github.com/bccp/nbodykit-cookbook/tree/master/recipes>`_
for examples. The recipes are designed to illustrate interesting uses of
nbodykit for other users to learn from.

We appreciate any and all contributions!

.. _nbodykit: https://github.com/bccp/nbodykit
