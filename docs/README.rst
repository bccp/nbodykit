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

Notebooks in this directory tree should **NOT** be modified. All
``.ipynb`` files in ``source/cookbook`` are ignored by git.

Executing notebooks
^^^^^^^^^^^^^^^^^^^

To execute all notebooks in the documentation that do not live
in ``source/cookbook``, use::

    python helper_scripts/run_notebooks.py

or equivalently, you can use ``make ipynb``.

Workflow for editing notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use the ``nbsphinx`` extension to include ``.ipynb`` files in the docs. A few useful things to know:

1. **Hidden Cells**: You can include hidden cells in the notebook, which will not show up in the HTML output, by editing the cell metadata. Use View -> Cell Toolbar -> Edit Metadata, and then go to the cell you want, click edit metadata, and insert into the metadata dictionary the key/value pair: ``"nbsphinx" : "hidden"``.
2. **ReST Cells**: You can include raw restructured text cells in the notebook, which will be converted properly. You must set the cell type to raw ReST. This can be done using:  View -> Cell Toolbar -> Raw Cell Format, and then going to the cell you want, and setting the type to ReST. 
3. **Linking to notebook files**: From other RST files, you can link to the notebooks via the ``ref`` directive. The important thing to remember is that you must use the full notebook file path, as if the top level directory is ``source``. So you can link to a notebook in the cookbook directory, using the path "cookbook/lognormal-mocks.ipynb".
