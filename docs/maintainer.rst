
Maintainer's corner
===================

Here, we outline notes intended for the maintainers of nbodykit, includuing details on running
the unit test suite and tagging and uploading releases. 

Dependency of tests
-------------------

- `pytest`_
- `pytest-pipeline`_
- `pytest-cov`_
- `coveralls`_

.. _`pytest`: http://pytest.org/latest/
.. _`pytest-pipeline`: https://github.com/bow/pytest-pipeline
.. _`pytest-cov`: https://pytest-cov.readthedocs.io/en/latest/
.. _`coveralls`: https://pypi.python.org/pypi/coveralls

To run the tests use

.. code:: 

    py.test nbodykit


or a specific test

.. code::

    py.test nbodykit/test/test_batch.py::TestStdin


Steps to tag a release
----------------------

For a stable release
++++++++++++++++++++

1. Bump the version, in :code:`nbodykit/__init__.py`, removing :code:`.dev0` postfix;

2. Add a git commit, add the version tag, push to master.

3. :code:`python setup.py sdist`

4. Upload the .tar.gz file in :code:`dist/` to pypi.

5. Bump the version, in :code:`nbodykit/__init__.py`, raise the version number,
   add `.dev0` postfix

6. Add a git commit, push to master.


For a development release
+++++++++++++++++++++++++

1. Bump the version, in :code:`nbodykit/__init__.py`, increase the release revision number in :code:`.dev0` postfix;

2. Add a git commit, add the version tag, push to master.

3. :code:`python setup.py sdist`

4. Add a git commit, push to master.

We shall have a nightly build on NERSC that builds three bundles in 
``/usr/common/contrib/bccp/nbodykit``: 

    - ``nbodykit-latest.tar.gz`` : built from the HEAD of the master branch 
    - ``nbodykit-stable.tar.gz`` : built from the latest version of nbodykit uploaded to pypi, which will only be built if the current bundle is out of date
    - ``nbodykit-dep.tar.gz`` : build from the dependencies in ``requirements.txt``, which will only be built if one of the current dependencies is out of date

