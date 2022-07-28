import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py_src/"))

from mk_modelling import MkModeller
import numpy as np
from numpy.random import default_rng
import unittest
import matplotlib.pyplot as plt

class TestMkModelling(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rng = default_rng(34875223)
    
    def test_dimensionality(self):
        n_timesteps = 100
        mk_modeller = MkModeller()
        mk_modeller.y_mask[:] = np.nan

        sol = mk_modeller.solve(n_timesteps)
        
        assert sol.shape[0] == n_timesteps
        assert sol.shape[1] == 9
        assert np.allclose(sol, 0.), "Did not stay all 0"

    def test_constraints(self):
        n_timesteps = 100
        mk_modeller = MkModeller()
        mk_modeller.y_mask[:] = 1.

        mk_modeller.y_mask[:] = 1.
        sol = mk_modeller.solve(n_timesteps)

        assert sol.shape[0] == n_timesteps
        assert sol.shape[1] == 9
        assert np.allclose(sol, 1.), "Did not stay all 1"

        for ii_test in range(9):
            mk_modeller = MkModeller()
            mk_modeller.y_mask[:] = np.nan 
            mk_modeller.y_mask[ii_test] = 30.

            sol = mk_modeller.solve(30)

            assert np.allclose(sol[:, ii_test]) == 30., "Not close to 30 in %u."%ii_test

    def test_equilibria(self):
        for ii_test in range(9):
            pass # TODO check proper implementation of all k values

if __name__ == '__main__':
    unittest.main()