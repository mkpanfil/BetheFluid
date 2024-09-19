import pytest
import sys
import os
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from BetheFluid import solver


def test_diffusion_time_evolution():
    """Test the diffusion time evolution against expected grid."""

    expected_object = solver.Solver.load('./tests/fixtures/diffusion.pkl')
    expected_grid = expected_object.grid

    tested_obj = solver.Solver()
    tested_obj.solve_equation()
    tested_grid = tested_obj.grid

    assert np.allclose(tested_grid, expected_grid, rtol=1e-5, atol=1e-8)