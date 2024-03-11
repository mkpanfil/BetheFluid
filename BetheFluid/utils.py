import numpy as np


# initial state

def foo1(l, x):
    rho = (2 + 0.25 * np.cos(2 * x * np.pi)) * (
            1 / (1 + np.exp(2 * (l + 2.4) ** 2 - 3)) + 1 / (1 + np.exp(2 * (l - 2.4) ** 2 - 3))) / (
                  2 * np.pi) * 0.25 / 2

    return rho


def potential_def(x):
    potential = 1 / 5 * np.cos(2 * np.pi * x)

    return potential


x_grid, int_x = np.linspace(-0.5, 0.5, 25, endpoint=False, retstep=True)
l_grid, int_l = np.linspace(-5, 5, 40, endpoint=False, retstep=True)
t_diff = np.arange(0, 0.005, 0.001)

t_diff2 = np.arange(0.006, 0.011, 0.001)

c_def = 3

rho = foo1

