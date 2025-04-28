import numpy as np
import math
from scipy.optimize import root
from BetheFluid.calc import TBA, CalcV, CalcD
from scipy.interpolate import NearestNDInterpolator

import warnings
class Therodynamic_Limit_LiebLiniger(TBA):
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


class VelocityLiebLiniger(CalcV, Therodynamic_Limit_LiebLiniger):
    '''
    Class calculating effective velocity of given state rho
    '''

    def get_operator(self, n):
        '''
        Creates 1 -Tn operator required for velocity calculations
        Returns
        -------
        operator : numpy array
        '''

        # dimensions : T(l,u) , n(x, u) -> Tn (x,l,u)
        # Tn = self.T[np.newaxis, :, :] * self.n[:, np.newaxis, :]

        Tn = np.einsum('lu, xu... -> xlu...', self.T, n, optimize=True)

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



class TBA_LiebLiniger(VelocityLiebLiniger):

    def __init__(self, rho, l, c, potential):
        super().__init__(rho, l, c)

        self.potential = potential

    def calc_particle_density(self, rho_p):
        integrated_density = np.sum(rho_p, axis=-1) * self.dl

        return integrated_density

    def calc_momentum(self, rho_p):
        momentum = self.miu_grid[np.newaxis, :] * rho_p

        integrated_momentum = np.sum(momentum, axis=-1) * self.dl

        return integrated_momentum

    def calc_energy(self, rho_p):
        potential = np.einsum('ij -> ji', self.potential)

        energy = (self.miu_grid[np.newaxis, :] ** 2 + potential) * rho_p

        # energy = (self.miu_grid[np.newaxis, :] ** 2) * rho_p

        integrated_energy = np.sum(energy, axis=-1) * self.dl

        return integrated_energy

    def calc_equillibrium_state(self, eps0):
        eps = eps0

        error = 1

        while error > 1e-6:
            kernel = self.T[np.newaxis, Ellipsis] * np.log(1.0 + np.exp(-eps))[:, np.newaxis, :]
            new_eps = eps0 - np.sum(kernel, axis=-1) * self.dl
            error = np.mean(np.sum(np.abs(new_eps - eps), axis=1) * self.dl)
            eps = new_eps

        return eps

    def calc_rho_from_eps(self, eps):
        n = 1 / (1 + np.exp(eps))

        operator = self.get_operator(n)

        rho_tot = 1 / (2 * np.pi) * np.sum(operator, axis=-1)

        rho_calculated = n * rho_tot

        return rho_calculated



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


class Calc_Potentials_Matrix(TBA_LiebLiniger):

    def __init__(self, rho, l, c, potential, tau):
        super().__init__(rho, l, c, potential)

        self.tau = tau
        self.potentials_matrix = self.calculate_potentials_matrix()


    def calc_potentials_for_rho_boosted(self):
        # Precompute target density, momentum, and energy from self.rho
        target_density = self.calc_particle_density(self.rho)

        momentum = self.calc_momentum(self.rho)
        energy = self.calc_energy(self.rho)

        target_energy = energy - momentum ** 2 / (2 * target_density)

        def equation_to_solve(params):
            params = params.reshape(2, self.rho.shape[0])

            # Calculate epsilon
            eps0 = params[0, :, np.newaxis] + 0.5 * params[1, :, np.newaxis] * self.miu_grid[np.newaxis, :] ** 2

            eps = self.calc_equillibrium_state(eps0)

            # Calculate rho and derived quantities
            rho_calculated = self.calc_rho_from_eps(eps)

            density_calculated = self.calc_particle_density(rho_calculated)
            energy_calculated = self.calc_energy(rho_calculated)

            residual_density = (density_calculated - target_density)
            residual_energy = (energy_calculated - target_energy)
            #print(residual_density)
            return np.concatenate([residual_density.ravel(), residual_energy.ravel()])

        beta_0 = np.ones_like(self.rho[:, 0])
        beta_1 = np.ones_like(self.rho[:, 0])
        # Solve using root
        initial_guess = np.stack((beta_0, beta_1)).reshape(-1)
        result = root(equation_to_solve, initial_guess,
                      method='hybr', tol=10 ** (-7))  # Here maybe it is worth considering the different methods

        if not result.success:
            raise ValueError(f"Root finding failed: {result.message}")

        arrays_for_matrix = (target_density, target_energy, result.x.reshape(2, self.rho.shape[0]))

        return arrays_for_matrix

    def calculate_potentials_matrix(self):
        transf_density, transf_energy, potentials = self.calc_potentials_for_rho_boosted()

        # Check for NaNs in input data
        # assert not np.isnan(transf_density).any(), "NaNs in transf_density"
        # assert not np.isnan(transf_energy).any(), "NaNs in transf_energy"
        # assert not np.isnan(susceptibilities).any(), "NaNs in susceptibilities"

        points = np.column_stack((transf_density, transf_energy))

        # Reuse triangulation for both interpolators
        #interp0 = LinearNDInterpolator(points, susceptibilities[0, :], fill_value=44)
        #interp1 = LinearNDInterpolator(points, susceptibilities[1, :], fill_value=44)
        interp0 = NearestNDInterpolator(points, potentials[0, :])
        interp1 = NearestNDInterpolator(points, potentials[1, :])

        def interpolator(new_density, new_energy):
            points_new = np.column_stack((new_density, new_energy))

            s0 = interp0(points_new)
            s1 = interp1(points_new)
            result = np.stack((s0, s1), axis=-1)

            return result

        return interpolator



class RTA_approximation(TBA_LiebLiniger):
    def __init__(self, rho, l, c, potential, tau, potentials_matrix):
        super().__init__(rho, l, c, potential)

        self.tau = tau
        self.potentials_matrix = potentials_matrix
        self.charges = self.calc_charges()
        self.C_operator = self.calc_C_operator()
        self.C_matrix = self.calc_C_matrix()
        self.rho_thermal = self.calc_rho_thermal()
        self.linearized_collision_integral = self.calc_linearized_collision_integral()
        self.collision_integral = self.calc_collision_integral()

    def calc_C_operator(self):

        Tn = self.T[np.newaxis, :, :] * self.n[:, np.newaxis, :]

        delta = np.identity(self.miu_grid.size)[np.newaxis, Ellipsis]

        op_inv = np.linalg.inv(delta / self.dl - Tn)

        middle = self.rho_tot * self.n * (1 - self.n)

        C_op = np.einsum('xul,xl, xlm -> xum', op_inv, middle, op_inv)

        return C_op

    def calc_C_matrix(self):

        C_matrix = np.zeros((self.rho.shape[0], 3, 3))
        for a in range(3):
            for b in range(3):
                miu_b = self.miu_grid ** (b) / math.factorial(b)
                miu_a = self.miu_grid ** (a) / math.factorial(a)

                C_mat_elem = np.einsum('l, xlu, u -> x', miu_b, self.C_operator, miu_a) * self.dl ** 2

                C_matrix[:, b, a] = C_mat_elem

        return C_matrix

    def calc_charges(self):
        # transf_density = self.calc_particle_density(self.rho)

        density = self.calc_particle_density(self.rho)
        momentum = self.calc_momentum(self.rho)
        energy = self.calc_energy(self.rho)

        transf_energy = energy - momentum ** 2 / (2 * density)

        #return (transf_density, transf_energy)
        return (density, momentum, transf_energy)

    def calc_rho_thermal(self):

        params = self.potentials_matrix(self.charges[0], self.charges[2])
        u = self.charges[1]/self.charges[0]

        beta0 = params[:, 0] + 0.5 * params[:, 1] * u**2
        beta1 = - params[:, 1] * u
        beta2 = params[:, 1]
        eps0 = beta0[:, np.newaxis] + beta1[:, np.newaxis] * self.miu_grid[np.newaxis, :]
        eps0 += 0.5 * beta2[:, np.newaxis] * self.miu_grid[np.newaxis, :] ** 2

        eps = self.calc_equillibrium_state(eps0)

        rho_thermal = self.calc_rho_from_eps(eps)

        return rho_thermal


    def calc_linearized_collision_integral(self):

        delta = np.identity(self.miu_grid.size)[np.newaxis, Ellipsis]
        C_inv = np.linalg.inv(self.C_matrix)

        sum_ab = np.zeros_like(self.C_operator)
        for a in range(3):
            for b in range(3):
                miu_b = self.miu_grid ** (b) / math.factorial(b)
                miu_a = self.miu_grid ** (a) / math.factorial(a)

                C_inv_mat_elem = C_inv[:, b, a]
                C_lambda = np.einsum('xlu, u -> xl', self.C_operator, miu_b) * self.dl

                sum_ab += C_lambda[:, :, np.newaxis] * C_inv_mat_elem[:, np.newaxis, np.newaxis] * miu_a[np.newaxis,
                                                                                                   np.newaxis, :]

        colision_integral = 1 / self.tau * (sum_ab - delta / self.dl)

        return colision_integral

    def calc_collision_integral(self):

        colision_integral = 1 / self.tau * (self.rho_thermal - self.rho)

        return colision_integral


### Now the onlny thing to do is to change the Solver object, to generate the class RTA in the loop for the next
### time step

if __name__ == '__main__':
    from BetheFluid import solver
    import matplotlib.pyplot as plt

    # path = '../../tests/fixtures/diffusion.pkl'

    l = np.linspace(-10, 10)

    x = np.linspace(-5,5 ,8)

    object = solver.Solver(miu_grid=l, x_grid=x)

    rho = object.grid[:, :, 0]

    relax = Calc_Potentials_Matrix(rho, object.miu_grid, object.coupling, object.potential, 5)

    sus_matrix = relax.calculate_potentials_matrix()

    sus_evaluation = sus_matrix(0.17, 1.2)



    # colision_int = relax.calc_linearized_collision_integral()
    #
    # density_check = np.sum(colision_int, axis=1) * object.dl
    #
    # momentum_check = np.sum(object.miu_grid[np.newaxis, :, np.newaxis] * colision_int, axis=1) * object.dl
    #
    # energy_check = np.sum(object.miu_grid[np.newaxis, :, np.newaxis] ** 2 * colision_int, axis=1) * object.dl
    #
    # colision_int_einv = np.linalg.eig(colision_int * object.dl)[0]
    #
    # plt.hist(colision_int_einv[10])
    # plt.show()

    #susceptibiliteis = relax.calc_potentials_for_rho_boosted()

    # eps0 = susceptibiliteis[0, :, np.newaxis] + susceptibiliteis[1, :, np.newaxis] * object.miu_grid[np.newaxis, :] ** 2
    #
    # eps = relax.calc_equillibrium_state(eps0)
    #
    # rho_boosted = relax.calc_rho_from_eps(eps)
    #
    # rho_energy = relax.calc_energy(rho.T)
    # rho_boosted_energy = relax.calc_energy(rho_boosted)
    #
    # rho_density = relax.calc_particle_density(rho)
    # rho_boosted_density = relax.calc_particle_density(rho_boosted)
    #
    #
    # plt.plot(object.miu_grid, rho[0, :], '--', label='rho x=1')
    # plt.plot(object.miu_grid, rho[5, :], '--', label='rho x=5')
    # plt.plot(object.miu_grid, rho[15, :], '--', label='rho x=15')
    #
    # plt.plot(object.miu_grid, rho_boosted[0, :], label='rho boost x=1')
    # plt.plot(object.miu_grid, rho_boosted[5, :], label='rho boost x=5')
    # plt.plot(object.miu_grid, rho_boosted[15, :], label='rho boost x=15')
    # plt.title('States')
    # plt.xlabel('momenta')
    # plt.legend()
    # plt.savefig(os.path.join(saving_path, 'final_state.png'))
    # plt.close()
    #
    #
    # plt.plot(object.x_grid, rho_density - rho_boosted_density, label='density difference')
    # plt.plot(object.x_grid, rho_energy - rho_boosted_energy, label='energy difference')
    # plt.title('Conservations')
    # plt.xlabel('x')
    # plt.legend()
    # plt.savefig(os.path.join(saving_path, 'conservations.png'))
    # plt.close()

    # plt.show()
