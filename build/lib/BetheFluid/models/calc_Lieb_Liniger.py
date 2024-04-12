import numpy as np
from BetheFluid.calc import TBA, CalcV, CalcD



class TBA_LiebLiniger(TBA):
    def create_T(self):
        '''
        Returns
        -------
        numpy array, integral kernel of GHD
        '''
        l, u = np.meshgrid(self.miu_grid, self.miu_grid, indexing='ij')

        T = self.coupling / np.pi * 1 / ((l - u) ** 2 + self.coupling ** 2)

        return T

    def calc_n_rho_tot(self):
        '''
        Calculates n and rho total
        Returns
        -------
        n : numpy array
        rho_tot : numpy array
        '''
        # new indices are rho: N, x, l

        rho_tot = 1 / (2 * np.pi) + np.einsum('lu..., xu... -> xl...', self.T, self.rho, optimize=True) * self.dl

        n = self.rho / rho_tot

        return n, rho_tot

class VelocityLiebLiniger(CalcV, TBA_LiebLiniger):
    '''
    Class calculating effective velocity of given state rho
    '''

    def get_operator(self):
        '''
        Creates 1 -Tn operator required for velocity calculations
        Returns
        -------
        operator : numpy array
        '''

        # dimensions : T(l,u) , n(x, u) -> Tn (x,l,u)
        # Tn = self.T[np.newaxis, :, :] * self.n[:, np.newaxis, :]

        Tn = np.einsum('lu, xu... -> xlu...', self.T, self.n, optimize=True)

        # create delta l,u for each x
        # dimensions x, l, u

        delta = np.identity(self.miu_grid.size)

        ones = np.ones_like(Tn)

        delta = np.einsum('xlu..., lu -> xlu...', ones, delta)

        operator = delta - Tn * self.dl

        # dimensions x, l, u

        operator = np.einsum('xlu... -> ...xlu', operator)

        operator = np.linalg.inv(operator)

        return operator


    def get_V(self):
        '''
        Calculates effective velocity
        Returns
        -------
        V : numpy array
        '''
        # dimensions x, l

        u = 2 * self.miu_grid

        k_dr = np.sum(self.operator, axis=-1)

        omega_dr = np.einsum('...xlu, u -> ...xl', self.operator, u)

        V = omega_dr / k_dr

        return V


class DiffusionLiebLiniger(VelocityLiebLiniger, CalcD):
    '''
    Class calculating diffusion operator for given state rho, derived class of CalcV
    '''

    def get_W(self):
        '''
        Calculates W operatos
        Returns
        -------
        W : numpy array
        '''
        T_dr = np.einsum('xlu, uo -> xlo', self.operator, self.T, optimize=True)

        # Now order of indices is x, l, u

        rho = self.rho[Ellipsis, np.newaxis]

        n = self.n[Ellipsis, np.newaxis]

        W = rho * (1 - n) * T_dr ** 2 * np.abs(self.V[Ellipsis, np.newaxis] - self.V[Ellipsis, np.newaxis, :])

        return W

    def get_D_ker(self):
        '''
        Calculates D_ker operator, where D operator is  (1 -Tn)-1 rho D_ker rho-1 (1 -Tn)
        Returns
        -------
        D_ker : numpy array
        '''
        delta = np.identity(self.miu_grid.size)[np.newaxis, Ellipsis]

        rho_tot = self.rho_tot[Ellipsis, np.newaxis]

        # dimensions x, l, u
        D_ker = (delta * self.w[Ellipsis, np.newaxis] - self.W * self.dl) / rho_tot ** 2

        return D_ker

    def get_D(self):
        '''
        Calculates diffusion operator
        Returns
        -------
        D : numpy array
        '''
        Tn = self.T[np.newaxis, :, :] * self.n[:, np.newaxis, :]

        delta = np.identity(self.miu_grid.size)[np.newaxis, Ellipsis]

        op_ker = delta - Tn * self.dl

        rho_factor = self.rho_tot[Ellipsis, np.newaxis] / self.rho_tot[Ellipsis, np.newaxis, :]

        D_ker = self.D_ker * rho_factor

        D = np.einsum('xou, xul , xls -> xos', self.operator, D_ker, op_ker, optimize=True)

        return D




