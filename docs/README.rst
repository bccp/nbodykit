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

To execute all jupyter notebooks in the docs before committing, use::

    make notebooks

This will execute all Jupyter notebooks by searching for ``*.ipynb`` files
in the ``source`` directory. Individual notebooks can be executed using::

    python helper_scripts/run_notebooks.py example.ipynb
