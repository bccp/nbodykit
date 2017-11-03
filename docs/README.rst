Build Instructions
------------------

To build a local copy of the nbodykit docs, install the programs in
environment.yml and run 'make html'. If you use the conda package manager
these commands suffice::

  git clone git@github.com:bccp/nbodykit.git
  cd nbodykit/docs
  conda env create --name nbodykitdocs -f environment.yml
  source activate nbodykitdocs
  make html
  open build/html/index.html

Developer Instructions
----------------------

The Cookbook notebooks
^^^^^^^^^^^^^^^^^^^^^^

Notebooks in ``source/cookbook`` are downloaded at build time from the
`bccp/nbodykit-cookbook <https://github.com/bccp/nbodykit-cookbook>`_
repository. Developers should modify and add new recipes to this
repository to update the cookbook in the docs.

Notebooks in this directory tree should NOT be modified. All
``.ipynb`` files in ``source/cookbook`` are ignored by git.

Executing notebooks
^^^^^^^^^^^^^^^^^^^

To execute all notebooks in the documentation that do not live
in ``source/cookbook``, use::

    python helper_scripts/run_notebooks.py

or equivalently, you can use ``make ipynb``.
