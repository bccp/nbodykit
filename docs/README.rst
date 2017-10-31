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

Executing notebooks
^^^^^^^^^^^^^^^^^^^

We often want to execute two commands. First, execute all notebooks not in
the ``source/cookbook`` submodule::

    python helper_scripts/run_notebooks.py -e source/cookbook/recipes

or equivalently, use ``make ipynb``.

Second, we can execute only the cookbook submodule::

    python helper_scripts/run_notebooks.py source/cookbook/recipes/

or equivalently, use ``make cookbook``.

Updating the cookbook submodule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With changes in the cookbook submodule, be sure to checkout the master branch
before committing any changes. From the ``docs`` directory::

    cd source/cookbook
    git checkout master
    git commit ....

To update the cookbook submodule, use::

    git submodule update --remote
