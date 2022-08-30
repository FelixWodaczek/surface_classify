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

        mk_modeller.k[...] = self.rng.random((8, 2))
        sol = mk_modeller.solve(n_timesteps)
        
        assert sol.shape[0] == n_timesteps
        assert sol.shape[1] == 9
        assert np.allclose(sol, 0.), "Did not stay all 0"

    def test_constraints(self):
        n_timesteps = 100
        mk_modeller = MkModeller()
        mk_modeller.y_mask[:] = 1.

        mk_modeller.y_mask[:] = 1.
        mk_modeller.k[...] = self.rng.random((8, 2))
        sol = mk_modeller.solve(n_timesteps)

        assert sol.shape[0] == n_timesteps
        assert sol.shape[1] == 9
        assert np.allclose(sol, 1.), "Did not stay all 1"

        for ii_test in range(9):
            mk_modeller = MkModeller()
            mk_modeller.y_mask[:] = np.nan 
            mk_modeller.y_mask[ii_test] = 0.3
            mk_modeller.k[...] = self.rng.random((8, 2))

            sol = mk_modeller.solve(30)

            assert np.allclose(sol[:, ii_test], 0.3), "Not close to 0.3 in %u."%ii_test

    def test_equilibria(self):
        target_indices = [
            [0, 0, 1],
            [1, 1, 2],
            [2, 1, 3],
            [3, 2, 4],
            [4, 3, 4],
            [5, 4, 5],
            [5, 4, 6], 
            [6, 5, 7],
            [6, 6, 7],
        ]
        mk_modeller = MkModeller()
        y_0 = np.zeros((9, ), dtype=np.float64)
        
        for index_triple in target_indices:
            index_pair = np.array(index_triple[1:])
            mk_modeller.y_mask[:] = np.nan
            if 5 in index_pair:
                mk_modeller.y_mask[6] = 1.
            elif 6 in index_pair:
                mk_modeller.y_mask[5] = 1.
            mk_modeller.k[...] = 0.
            mk_modeller.k[index_triple[0], :] = np.array([0.2, 0.4])
            y_0[:] = 0.
            y_0[index_pair] = 1.

            sol = mk_modeller.solve(100, y_0=y_0, max_step=1e-2)

            final_values = sol[-1, :]
            assert np.allclose(np.delete(final_values, index_pair), 0.), "Leakage at %s"%(
                " ".join(map(str, np.argwhere(np.delete(final_values, index_pair)!=0).tolist()))
            )
            frac = final_values[index_pair[1]]/final_values[index_pair[0]]
            assert np.allclose(frac, 2.), "Fraction error. At index [%s %s] the fraction gives %f but should be 2 to 1"%(index_pair[0], index_pair[1], frac)

if __name__ == '__main__':
    unittest.main()