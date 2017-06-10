from .utils.pipeline import RunAlgorithm, add_run_fixture
from .utils import asserts
from . import os, unittest
from .. import examples_dir
import pytest

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

    @pytest.mark.xfail(reason='numpy compiling differences')
    def test_result(self):
        asserts.test_bigfile_result(self, 'PaintGrid', rtol=1e-3, atol=1e-3)

@add_run_fixture(__name__, RunPaintAlgorithm, 'PaintGrid')
class TestInterlacedPaint(unittest.TestCase):
    param_file  = "test_fastpm_interlaced.params"
    output_file = "test_paint_fastpm_interlaced"
    datasources = ['fastpm_1.0000']

    def test_exit_code(self):
        asserts.test_exit_code(self)

    def test_exception(self):
        asserts.test_exception(self)

    @pytest.mark.xfail(reason='numpy compiling differences')
    def test_result(self):
        asserts.test_bigfile_result(self, 'PaintGrid', rtol=1e-3, atol=1e-3)

@add_run_fixture(__name__, RunPaintAlgorithm, 'PaintGrid')
class TestPaintGrid(unittest.TestCase):
    param_file  = "test_grid.params"
    output_file = "test_paint_grid"
    datasources = ['bigfile_grid']

    def test_exit_code(self):
        asserts.test_exit_code(self)

    def test_exception(self):
        asserts.test_exception(self)

    @pytest.mark.xfail(reason='numpy compiling differences')
    def test_result(self):
        asserts.test_bigfile_result(self, 'PaintGrid', rtol=1e-3, atol=1e-3)

@add_run_fixture(__name__, RunPaintAlgorithm, 'PaintGrid')
class TestPaintGridK(unittest.TestCase):
    param_file  = "test_gridk.params"
    output_file = "test_paint_gridk"
    datasources = ['bigfile_gridk']

    def test_exit_code(self):
        asserts.test_exit_code(self)

    def test_exception(self):
        asserts.test_exception(self)

    def test_result(self):
        asserts.test_bigfile_result(self, 'PaintGrid', rtol=1e-3, atol=1e-3)
