import matplotlib.pyplot as plt
import numpy as np
import dill
import scipy as sci


class Observable:

    def __init__(self, Solver_object):

        self.object = self.get_object(Solver_object)
        # dimensions: N, l, x, t       
        self.T = self.get_T()
        self.rho_tot = self.get_rho_tot()
        self.rho_h = self.get_rho_h()
        # self.n = self.get_n()

    def get_object(self, inp) -> object:
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

    def get_T(self):

        # numba doesn't support meshgrid
        l, u = np.meshgrid(self.object.l, self.object.l, indexing='ij')

        T_grid = []

        for value in self.object.c:
            T = value / np.pi * 1 / ((l - u) ** 2 + value ** 2)

            T_grid.append(T)

        # dimensions N, l, u   
        T = np.stack(T_grid)

        return T

    def get_rho_tot(self):

        # new indices are rho: N, x, l

        rho_tot = 1 / (2 * np.pi) + np.einsum('Nlu, Nuxt -> Nlxt', self.T, self.object.grid,
                                              optimize=True) * self.object.int_l

        return rho_tot

    def get_rho_h(self):

        rho_h = self.rho_tot - self.object.grid

        return rho_h

    def energy(self, option='total'):
        # Dimensions N, l, x, t
        l = self.object.l[np.newaxis, :, np.newaxis, np.newaxis]
        energy_grid = self.object.grid * l ** 2

        option_mapping = {
            'local': (np.sum(energy_grid, axis=1) * self.object.int_l),
            'theta': (np.sum(energy_grid, axis=2) * self.object.int_x),
            'total': (np.sum(energy_grid, axis=(1, 2)) * self.object.int_l * self.object.int_x)
        }

        if option not in option_mapping:
            raise ValueError('Incorrect argument, choose between: local, theta, total')

        energy = option_mapping[option]

        if energy.shape[0] == 1:
            energy = energy[0, Ellipsis]

        return energy

    # # here I can make it better if the index is negative
    def n(self, option='total'):
        # Dimensions N, l, x, t

        n_grid = self.object.grid / self.rho_tot

        option_mapping = {
            'local': (np.sum(n_grid, axis=1) * self.object.int_l),
            'theta': (np.sum(n_grid, axis=2) * self.object.int_x),
            'total': (np.sum(n_grid, axis=(1, 2)) * self.object.int_l * self.object.int_x)
        }

        if option not in option_mapping:
            raise ValueError('Incorrect argument, choose between: local, theta, total')

        n = option_mapping[option]

        if n.shape[0] == 1:
            n = n[0, Ellipsis]

        return n

    def entropy(self, option='total'):

        # Dimensions N, l, x, t

        rho = self.object.grid

        rho[rho < 0] = 0

        S_grid = self.rho_tot * np.log(self.rho_tot) - sci.special.xlogy(rho, rho) - sci.special.xlogy(self.rho_h,
                                                                                                       self.rho_h)
        option_mapping = {
            'local': (np.sum(S_grid, axis=1) * self.object.int_l),
            'theta': (np.sum(S_grid, axis=2) * self.object.int_x),
            'total': (np.sum(S_grid, axis=(1, 2)) * self.object.int_l * self.object.int_x)
        }

        if option not in option_mapping:
            raise ValueError('Incorrect argument, choose between: local, theta, total')

        entropy = option_mapping[option]

        if entropy.shape[0] == 1:
            entropy = entropy[0, Ellipsis]

        return entropy

    def plot_template(self, observable, option='local', frames=(0, -1), N=0, path=None, name='', style='-'):

        main_options_dictionairy = {
            'n': {
                'local': {'x_axis': self.object.x, 'y_axis': self.n('local'), 'x_label': 'x', 'y_label': 'n'},
                'theta': {'x_axis': self.object.l, 'y_axis': self.n('theta'), 'x_label': 'theta', 'y_label': 'n'},
                'total': {'x_axis': self.object.t, 'y_axis': self.n('total'), 'x_label': 'time', 'y_label': 'n'}},

            'energy': {
                'local': {'x_axis': self.object.x, 'y_axis': self.energy('local'), 'x_label': 'x', 'y_label': 'energy'},
                'theta': {'x_axis': self.object.l, 'y_axis': self.energy('theta'), 'x_label': 'theta',
                          'y_label': 'energy'},
                'total': {'x_axis': self.object.t, 'y_axis': self.energy('total'), 'x_label': 'time',
                          'y_label': 'energy'}},

            'entropy': {
                'local': {'x_axis': self.object.x, 'y_axis': self.entropy('local'), 'x_label': 'x', 'y_label': 'entropy'},
                'theta': {'x_axis': self.object.l, 'y_axis': self.entropy('theta'), 'x_label': 'theta',
                          'y_label': 'entropy'},
                'total': {'x_axis': self.object.t, 'y_axis': self.entropy('total'), 'x_label': 'time',
                          'y_label': 'entropy'}}
        }

        option_mapping = main_options_dictionairy[observable]

        if option not in option_mapping:
            raise ValueError('Incorrect argument, choose between: local, theta, total')

        if option == 'total':
            plt.plot(option_mapping[option]['x_axis'], option_mapping[option]['y_axis'][Ellipsis, :], style,
                     )
            plt.xlabel('time')
            plt.ylabel(option_mapping[option]['y_label'])
            plt.show()

        else:
            for item in frames:
                plt.plot(option_mapping[option]['x_axis'], option_mapping[option]['y_axis'][Ellipsis, item], style,
                         label='{} t = {}'.format(name, round(item * self.object.int_t, 3)))
            plt.xlabel(option_mapping[option]['x_label'])
            plt.ylabel(option_mapping[option]['y_label'])
            plt.legend()
            plt.show()

        if path is not None:
            plt.savefig(path)

    def plot_n(self, option='local', frames=(0, -1), N=0, path=None, name='', style='-'):

        self.plot_template(observable='n', option=option, frames=frames, N=N, path=path, name=name, style=style)

    def plot_energy(self, option='total', frames=(0, -1), N=0, path=None, name='', style='-'):

        self.plot_template(observable='energy', option=option, frames=frames, N=N, path=path, name=name, style=style)

    def plot_entropy(self, option='total', frames=(0, -1), N=0, path=None, name='', style='-'):

        self.plot_template(observable='entropy', option=option, frames=frames, N=N, path=path, name=name, style=style)
