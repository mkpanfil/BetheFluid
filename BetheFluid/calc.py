import numpy as np
from abc import ABC, abstractmethod


###############################################################################################################
#          Module containing abstract classes for calculating TBA, effective velocity and diffusion operator
###############################################################################################################

class TBA(ABC):
    def __init__(self, rho, l, c):
        self.rho = self.calc_rho(rho)
        self.miu_grid = l
        self.coupling = c
        self.dl = np.diff(self.miu_grid).mean()
        self.T = self.create_T()
        self.n, self.rho_tot = self.calc_n_rho_tot()


    def calc_rho(self, rho):
        '''
        Changes dimensions order, to more useful in further calculations
        Parameters
        ----------
        rho : numpy array

        Returns
        -------
        numpy array, rho with dimensions: x, l

        '''
        rho = np.einsum('lx... -> xl...', rho)

        return rho

    @abstractmethod
    def create_T(self):
        pass

    @abstractmethod
    def calc_n_rho_tot(self):
        pass

class CalcV(TBA):

    def __init__(self, rho, l, c):
        super().__init__(rho, l, c)
        self.operator = self.get_operator()
        self.V = self.get_V()

    @abstractmethod
    def get_operator(self):
        pass

    @abstractmethod
    def get_V(self):
        pass



class CalcD(CalcV):

    def __init__(self, rho, l, c):

        super().__init__(rho, l, c)

        # dimensions N, x, l
        self.W = self.get_W()
        self.w = np.sum(self.W, axis=-2) * self.dl
        self.D_ker = self.get_D_ker()
        self.D = self.get_D()

    @abstractmethod
    def get_W(self):
        pass

    @abstractmethod
    def get_D_ker(self):
        pass

    @abstractmethod
    def get_D_ker(self):
        pass

    @abstractmethod
    def get_D(self):
        pass
