import numpy as np


from BetheFluid.calc import CalcV, CalcD


class VelocityLiebLiniger(CalcV):
    '''
    Class calculating effective velocity of given state rho
    '''

    def create_T(self):
        '''
        Returns
        -------
        numpy array, integral kernel of GHD
        '''
        l, u = np.meshgrid(self.l, self.l, indexing='ij')

        T = self.c / np.pi * 1 / ((l - u) ** 2 + self.c ** 2)

        return T


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
        delta = np.identity(self.l.size)[np.newaxis, Ellipsis]

        rho_tot = self.rho_tot[Ellipsis, np.newaxis]

        # dimensions x, l, u
        D_ker = (delta * self.w[Ellipsis, np.newaxis] - self.W * self.int_l) / rho_tot ** 2

        return D_ker

    def get_D(self):
        '''
        Calculates diffusion operator
        Returns
        -------
        D : numpy array
        '''
        Tn = self.T[np.newaxis, :, :] * self.n[:, np.newaxis, :]

        delta = np.identity(self.l.size)[np.newaxis, Ellipsis]

        op_ker = delta - Tn * self.int_l

        rho_factor = self.rho_tot[Ellipsis, np.newaxis] / self.rho_tot[Ellipsis, np.newaxis, :]

        D_ker = self.D_ker * rho_factor

        D = np.einsum('xou, xul , xls -> xos', self.operator, D_ker, op_ker, optimize=True)

        return D


# Changes:  rho_factor is in different place and D_ker is devided by rho_tot**2



