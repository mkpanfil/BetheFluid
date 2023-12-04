import BetheFluid
import numpy as np


def foo1(momenta, x):
    foo = (2 + 0.25 * np.cos(2 * x * np.pi)) * (
            1 / (1 + np.exp(2 * (momenta + 2.4) ** 2 - 3)) + 1 / (1 + np.exp(2 * (momenta - 2.4) ** 2 - 3))) / (
                  2 * np.pi) * 0.25 / 2

    return foo


def potential(x):
    pot = 1 / 5 * np.cos(2 * np.pi * x)

    return pot


x_grid, int_x = np.linspace(-0.5, 0.5, 25, endpoint=False, retstep=True)
l_grid, int_l = np.linspace(-5, 5, 40, endpoint=False, retstep=True)
t_diff = np.arange(0, 0.005, 0.001)
c = 3

rho = foo1

diffusion = BetheFluid.Solver(t=t_diff, l=l_grid, x=x_grid, rho0=rho, c=c, diff=True, potential=potential)
diffusion.solve_equation()


obs = BetheFluid.Observable(diffusion)

energy = obs.energy('theta')
obs.plot_entropy('local', N = 1)

