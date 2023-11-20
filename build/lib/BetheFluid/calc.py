import numpy as np


class CalcV:
    '''
    Class calculating effective velocity of given state rho
    '''

    def __init__(self, rho, l, c):
        '''
        Constructor of a class
        Parameters
        ----------
        rho : numpy array, state
        l : numpy array, momenta
        c : tuple of floats, coupling constants
        '''
        # dimensions of rho N, x, l
        self.rho = self.get_rho(rho)
        self.l = l
        self.c = c
        self.int_l = np.diff(self.l).mean()
        self.T = self.create_T()
        self.n, self.rho_tot = self.get_n_rho_tot()
        self.operator = self.get_operator()
        self.V = self.get_V()

    def get_rho(self, rho):
        '''
        Changes dimensions order, to more useful in further calculations
        Parameters
        ----------
        rho : numpy array

        Returns
        -------
        numpy array, rho with dimensions: N, x, l

        '''
        rho = np.einsum('Nlx -> Nxl', rho)

        return rho

    def create_T(self):
        '''
        Returns
        -------
        numpy array, integral kernel of GHD
        '''
        l, u = np.meshgrid(self.l, self.l, indexing='ij')

        T_grid = []

        for value in self.c:
            T = value / np.pi * 1 / ((l - u) ** 2 + value ** 2)

            T_grid.append(T)

        # dimensions N, l, u   
        T = np.stack(T_grid)

        return T

    def get_n_rho_tot(self):
        '''
        Calculates n and rho total
        Returns
        -------
        n : numpy array
        rho_tot : numpy array
        '''
        # new indices are rho: N, x, l

        rho_tot = 1 / (2 * np.pi) + np.einsum('Nlu, Nxu -> Nxl', self.T, self.rho) * self.int_l

        n = self.rho / rho_tot

        return n, rho_tot

    def get_operator(self):
        '''
        Creates 1 -Tn operator required for velocity calculations
        Returns
        -------
        operator : numpy array
        '''

        # dimensions : T(N,l,u) , n(N, x, u)
        Tn = self.T[:, np.newaxis, :, :] * self.n[:, :, np.newaxis, :]

        # create delta l,u for each N and x
        # dimensions N, x, l, u

        delta = np.identity(self.l.size)[np.newaxis, np.newaxis, :, :]

        operator = delta - Tn * self.int_l

        # dimensions N,x, l, u
        operator = np.linalg.inv(operator)

        return operator

    def get_V(self):
        '''
        Calculates effective velocity
        Returns
        -------
        V : numpy array
        '''
        # dimensions N, x, l

        u = 2 * self.l

        k_dr = np.sum(self.operator, axis=-1)

        omega_dr = np.einsum('Nxlu, u -> Nxl', self.operator, u)

        V = omega_dr / k_dr

        return V


class CalcD(CalcV):
    '''
    Class calculating diffusion operator for given state rho, derived class of CalcV
    '''

    def __init__(self, rho, l, c):
        '''
        Constructor of this class
        Parameters
        ----------
        rho : numpy array, state at given time
        l : numpy array, momenta dimension
        c : tuple of floats, coupling constants
        '''
        super().__init__(rho, l, c)

        # dimensions N, x, l    
        self.W = self.get_W()
        self.w = np.sum(self.W, axis=-2) * self.int_l
        # self.w = self.get_w()
        self.D_ker = self.get_D_ker()
        self.D = self.get_D()

    def get_W(self):
        '''
        Calculates W operatos
        Returns
        -------
        W : numpy array
        '''
        T_dr = np.einsum('Nxlu, Nuo -> Nxlo', self.operator, self.T, optimize=True)

        # Now order of indices is N, x, l, u

        rho = self.rho[Ellipsis, np.newaxis]

        n = self.n[Ellipsis, np.newaxis]

        # rho_tot = self.rho_tot[Ellipsis, np.newaxis]

        W = rho * (1 - n) * T_dr ** 2 * np.abs(self.V[Ellipsis, np.newaxis] - self.V[Ellipsis, np.newaxis, :])

        return W

    def get_D_ker(self):
        '''
        Calculates D_ker operator, where D operator is  (1 -Tn)-1 rho D_ker rho-1 (1 -Tn)
        Returns
        -------
        D_ker : numpy array
        '''
        delta = np.identity(self.l.size)[np.newaxis, np.newaxis, Ellipsis]

        rho_tot = self.rho_tot[Ellipsis, np.newaxis]

        # dimensions N, x, l, u
        D_ker = (delta * self.w[Ellipsis, np.newaxis] - self.W * self.int_l) / rho_tot ** 2

        return D_ker

    def get_D(self):
        '''
        Calculates diffusion operator
        Returns
        -------
        D : numpy array
        '''
        Tn = self.T[:, np.newaxis, :, :] * self.n[:, :, np.newaxis, :]

        delta = np.identity(self.l.size)[np.newaxis, np.newaxis, Ellipsis]

        op_ker = delta - Tn * self.int_l

        rho_factor = self.rho_tot[Ellipsis, np.newaxis] / self.rho_tot[Ellipsis, np.newaxis, :]

        D_ker = self.D_ker * rho_factor

        D = np.einsum('Nxou, Nxul , Nxls -> Nxos', self.operator, D_ker, op_ker, optimize=True)

        return D

# Changes:  rho_factor is in different place and D_ker is devided by rho_tot**2
