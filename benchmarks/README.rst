Running the nbodykit benchmarks
--------------------------------

We can run the benchmarking suite by executing (from
the top-level directory):

.. code:: bash

    python run-tests.py BENCHNAME -m SAMPLE --no-build --bench  --bench-dir BENCHDIR

where

- ``BENCHNAME``: the name of the test file/function to run, e.g.,
  ``nbodykit/benchmarks/test_fftpower.py``.

- ``BENCHDIR``: the directory where the benchmarking results will be saved

- ``SAMPLE``: one of the samples to run, "test", "boss_like", or "desi_like".
  The ``-m`` option ensures only tests marked with that name will be executed.
