import matplotlib.pyplot as plt
import numpy as np
import dill
import scipy as sci
from typing import Union
from BetheFluid.solver import Solver


class Observable:

    def __init__(self, Solver_object: Union[Solver, str]):

        self.solver_object = self.get_object(Solver_object)
        # dimensions: l, x, t
        self.TBA_object = self.solver_object.get_model('TBA', self.solver_object.grid, self.solver_object.l,
                                                       self.solver_object.c)

        self.T = self.TBA_object.T
        self.rho_tot = np.einsum('xlt -> lxt', self.TBA_object.rho_tot)
        self.n = np.einsum('xlt -> lxt', self.TBA_object.n)
        self.rho_h = self.get_rho_h()

    def get_object(self, inp: Union[Solver, str]) -> Solver:
        '''
        loading Solver object into observable class
        Parameters
        ----------
        inp

        Returns
        -------

        '''
        if isinstance(inp, str):

            with open(inp, 'rb') as file:
                arr = dill.load(file)

            return arr

        else:
            return inp

    def get_rho_h(self):

        rho_h = self.rho_tot - self.solver_object.grid

        return rho_h

    def __calc_template(self, observable, option):

        option_mapping = {
            'local': (np.sum(observable, axis=0) * self.solver_object.int_l),
            'theta': (np.sum(observable, axis=1) * self.solver_object.int_x),
            'total': (np.sum(observable, axis=(0, 1)) * self.solver_object.int_l * self.solver_object.int_x)
        }

        if option not in option_mapping:
            raise ValueError('Incorrect argument, choose between: local, theta, total')

        result = option_mapping[option]

        return result

    def calc_energy(self, option='total'):
        # Dimensions l, x, t
        l = self.solver_object.l[:, np.newaxis, np.newaxis]
        energy_grid = self.solver_object.grid * l ** 2

        return self.__calc_template(energy_grid, option)

    # # here I can make it better if the index is negative
    def calc_n(self, option='total'):
        return self.__calc_template(self.n, option)

    def calc_entropy(self, option='total'):

        # Dimensions N, l, x, t

        rho = self.solver_object.grid

        rho[rho < 0] = 0

        S_grid = self.rho_tot * np.log(self.rho_tot) - sci.special.xlogy(rho, rho) - sci.special.xlogy(self.rho_h,
                                                                                                       self.rho_h)
        return self.__calc_template(S_grid, option)

    def __plot_template(self, observable, option='local', frames=(0, -1), path=None, name='', style='-'):

        main_options_dictionairy = {
            'n': {
                'local': {'x_axis': self.solver_object.x, 'y_axis': self.calc_n('local'), 'x_label': 'x',
                          'y_label': 'n'},
                'theta': {'x_axis': self.solver_object.l, 'y_axis': self.calc_n('theta'), 'x_label': 'theta',
                          'y_label': 'n'},
                'total': {'x_axis': self.solver_object.t, 'y_axis': self.calc_n('total'), 'x_label': 'time',
                          'y_label': 'n'}},

            'energy': {
                'local': {'x_axis': self.solver_object.x, 'y_axis': self.calc_energy('local'), 'x_label': 'x',
                          'y_label': 'energy'},
                'theta': {'x_axis': self.solver_object.l, 'y_axis': self.calc_energy('theta'), 'x_label': 'theta',
                          'y_label': 'energy'},
                'total': {'x_axis': self.solver_object.t, 'y_axis': self.calc_energy('total'), 'x_label': 'time',
                          'y_label': 'energy'}},

            'entropy': {
                'local': {'x_axis': self.solver_object.x, 'y_axis': self.calc_entropy('local'), 'x_label': 'x',
                          'y_label': 'entropy'},
                'theta': {'x_axis': self.solver_object.l, 'y_axis': self.calc_entropy('theta'), 'x_label': 'theta',
                          'y_label': 'entropy'},
                'total': {'x_axis': self.solver_object.t, 'y_axis': self.calc_entropy('total'), 'x_label': 'time',
                          'y_label': 'entropy'}}
        }

        option_mapping = main_options_dictionairy[observable]

        if option not in option_mapping:
            raise ValueError('Incorrect argument, choose between: local, theta or total')

        if option == 'total':
            plt.plot(option_mapping[option]['x_axis'], option_mapping[option]['y_axis'], style,
                     )
            plt.xlabel('time')
            plt.ylabel(option_mapping[option]['y_label'])
            plt.show()

        else:
            for item in frames:
                plt.plot(option_mapping[option]['x_axis'], option_mapping[option]['y_axis'][Ellipsis, item], style,
                         label='{} t = {}'.format(name, round(self.solver_object.t[item], 3)))
            plt.xlabel(option_mapping[option]['x_label'])
            plt.ylabel(option_mapping[option]['y_label'])
            plt.legend()
            plt.show()

        if path is not None:
            plt.savefig(path)

    def plot_n(self, option='local', frames=(0, -1), path=None, name='', style='-'):

        self.__plot_template(observable='n', option=option, frames=frames, path=path, name=name, style=style)

    def plot_energy(self, option='total', frames=(0, -1), path=None, name='', style='-'):

        self.__plot_template(observable='energy', option=option, frames=frames, path=path, name=name, style=style)

    def plot_entropy(self, option='total', frames=(0, -1), path=None, name='', style='-'):

        self.__plot_template(observable='entropy', option=option, frames=frames, path=path, name=name, style=style)
