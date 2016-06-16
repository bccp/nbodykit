
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

3. Reset the stable branch to the tag, do a forced push to replace the old stable branch
   This will trigger a new nightly stable build. We may want to add a github hook to use
   NIM service and trigger a rebuild on NERSC automatically.

4. :code:`python setup.py sdist`

5. Upload the .tar.gz file in :code:`dist/` to pip.

6. Bump the version, in :code:`nbodykit/__init__.py`, raise the version number,
   add `.dev0` postfix

7. Add a git commit, push to master.


For a development release
+++++++++++++++++++++++++

1. Bump the version, in :code:`nbodykit/__init__.py`, increase the release revision number in :code:`.dev0` postfix;

2. Add a git commit, add the version tag, push to master.

3. Reset the unstable branch to the tag, do a forced push to replace the old unstable branch
   This will trigger a new nightly unstable build. We may want to add a github hook to use
   NIM service and trigger a rebuild on NERSC automatically.

4. :code:`python setup.py sdist`

5. Add a git commit, push to master.

.. todo::

    Add a git push / commit hook and scan for version bumps,
    show this instruction if it looks like tagging a release.

In the future we may want to use a version branch instead of a version tag;
that means we will get rid of all developing branches from bccp/nbodykit,
and strictly maintain to-merged branches on
forks.

Also we shall have a nightly build on NERSC that tracks master. So three versions on nersc,

stable, unstable, nightly (master branch).

