import numpy as np
import scipy as sp
from scipy.integrate import ode
from scipy.constants import Boltzmann

class MkModeller:
    def __init__(self):
        self.diffusion = np.zeros((9, ), dtype=np.float64)
        self.k = np.zeros((9, 2), dtype=np.float64)

        self.y_mask = np.empty((9, ), np.float64)
        self.y_mask[0] = 1.
        self.y_mask[6] = 0.1
        self.y_mask[8] = 0.1
        self.mask_cond = np.logical_not(np.isnan(self.y_mask))

        self.y_inputs = np.zeros((9, ), np.float64)

    def init_mask_cond(self):
        self.mask_cond = np.logical_not(np.isnan(self.y_mask))

    @staticmethod
    def k_from_energy(temperature, energy=None, delta_h=None, delta_s=None):
        if energy is None:
            return np.exp(-delta_h/(temperature * Boltzmann)) * np.exp(delta_s/Boltzmann)
        else:
            return np.exp(-energy/(temperature * Boltzmann))


    def system_of_equations(self, t, y):
        self.diffusion[:] = 0.

        star = 1 - np.sum(y[1:6]) - y[7] # TODO: check if this is actually correct
        self.diffusion[0] = self.k[0, 0] * y[1] - self.k[0, 1] * y[0] 
        self.diffusion[1] = self.k[0, 1] * y[0] - self.k[1, 1] * y[1] + self.k[1, 0] * y[2] - self.k[2, 1] * y[1] + self.k[2, 0] * y[3] - self.k[0, 0] * y[1]
        self.diffusion[2] = self.k[1, 1] * y[1] - self.k[1, 0] * y[2] + self.k[3, 0] * y[4] - self.k[3, 1] * y[2]
        self.diffusion[3] = self.k[2, 1] * y[1] - self.k[2, 0] * y[2] + self.k[4, 0] * y[4] - self.k[4, 1] * y[3]
        self.diffusion[4] = self.k[3, 1] * y[2] + self.k[4, 1] * y[3] + self.k[5, 0] * y[5] * y[6] - self.k[3, 0] * y[4] - self.k[4, 0] * y[4] - self.k[5, 1] * y[4]
        self.diffusion[5] = self.k[5, 1] * y[4] + self.k[6, 0] * y[7] - self.k[5, 0] * y[5] * y[6] - self.k[6, 1] * y[5]
        self.diffusion[6] = self.k[5, 1] * y[4] - self.k[5, 0] * y[5] * y[6]
        self.diffusion[7] = self.k[6, 1] * y[5] - self.k[7, 1] * y[7] - self.k[6, 0] * y[7] + self.k[7, 0] * star * y[8]
        self.diffusion[8] = self.k[7, 1] * y[7] - self.k[7, 0] * star * y[8]

        return self.diffusion

    
    def constrained_soe(self, t, y):
        self.y_inputs[:] = np.where(self.mask_cond, self.y_mask, y)
        
        self.system_of_equations(t, self.y_inputs)
        self.diffusion[self.mask_cond] = 0.
        
        return self.diffusion


    def solve(self, n_steps, y_0=None, target_function=None):
        self.init_mask_cond()

        if y_0 is None:
            y_0 = np.zeros((9, ), dtype=np.float64)
            y_0[self.mask_cond] = self.y_mask[self.mask_cond]

        if target_function is None:
            target_function = self.constrained_soe

        solver = ode(target_function).set_integrator('vode', method='adams')
        solver.set_initial_value(y=y_0, t=0)
        
        solution = np.zeros((n_steps, y_0.shape[0]))
        for ii_step, step in enumerate(np.arange(1, n_steps+1)):
            solution[ii_step, :] = solver.integrate(step)

        return solution

if __name__ == '__main__':
    pass