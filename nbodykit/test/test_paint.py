from .pipeline import RunAlgorithm, add_run_fixture
from . import os, asserts, unittest, pytest
from .. import examples_dir

class RunPaintAlgorithm(RunAlgorithm):
    run_dir = os.path.join(examples_dir, 'paint')
           

@add_run_fixture(__name__, RunPaintAlgorithm, 'PaintGrid')
class TestPaint(unittest.TestCase):
    param_file  = "test_fastpm.params"
    output_file = "test_paint_fastpm"
    datasources = ['fastpm_1.0000']

    def test_exit_code(self):
        asserts.test_exit_code(self)

    def test_exception(self):
        asserts.test_exception(self)

    def test_result(self):
        asserts.test_bigfile_result(self, 'PaintGrid')

@add_run_fixture(__name__, RunPaintAlgorithm, 'PaintGrid')
class TestPaintGrid(unittest.TestCase):
    param_file  = "test_grid.params"
    output_file = "test_paint_grid"
    datasources = ['bigfile_grid']

    def test_exit_code(self):
        asserts.test_exit_code(self)

    def test_exception(self):
        asserts.test_exception(self)

    def test_result(self):
        asserts.test_bigfile_result(self, 'PaintGrid')
