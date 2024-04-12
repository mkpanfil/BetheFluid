import numpy as np


##################################################################################
#                   Data for creating default Solver object
##################################################################################
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

##################################################################################
#                   Basic math functions used in modules
##################################################################################

def x_der(arr, dx):
    '''
    Calculates the derivative in position space
    Parameters
    ----------
    arr : numpy array of the grid at given time
    dx : numpy array of the spacing of the grid
    Returns
    -------
    numpy array which is a derivative in x dimension of inputted arr
    '''
    der = 1 / (2 * dx) * (np.roll(arr, -1, axis=-1) - np.roll(arr, 1, axis=-1))

    return der


def lambda_der(arr, dl):
    '''
    Calculates the derivative in momentum space

    Parameters
    ----------
    arr : numpy array of the grid at given time
    dl: spacing of the grid
    Returns
    -------
    numpy array which is a derivative in l dimension of inputted arr
    '''
    der = 1 / (2 * dl) * (np.roll(arr, -1, axis=-2) - np.roll(arr, 1, axis=-2))

    return der