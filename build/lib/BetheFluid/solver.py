import numpy as np
from BetheFluid.models.calc_Lieb_Liniger import TBA_LiebLiniger, VelocityLiebLiniger, DiffusionLiebLiniger
from tqdm import tqdm
import dill
import BetheFluid.utils as uts


class Solver:

    # x_grid, t_grid, miu_grid
    def __init__(self, t_grid=uts.t_diff, miu_grid=uts.l_grid, x_grid=uts.x_grid, rho0=uts.foo1, coupling=uts.c_def,
                 diff=True,
                 potential=uts.potential_def,
                 boundary=None, model='Lieb-Liniger'):
        '''
        constructor of the class
        Parameters
        ----------
        miu_grid : list or numpy array
        x_grid : list or numpy array
        t_grid : list or numpy array
        rho0 : function
        coupling : float
        boundary : None or tuple
        diff : bool
        '''
        self.miu_grid, self.x_grid, self.t_grid = self.correct_l_x_t(miu_grid, x_grid, t_grid)
        self.coupling = coupling
        self.rho0 = rho0
        self.boundary = self.correct_boundary(boundary)
        self.potential = self.calc_potential(potential)
        self.dx, self.dt, self.dl, self.steps_number = self.get_grid_spacing()
        self.diff = diff
        self.convergence = []
        self.model = model
        self.grid = self.create_initial_grid()

    def __str__(self):
        """
        Overloads print function to type informations about Solver Object
        """
        informations = {
            'coupling constant': self.coupling,
            'time grid': 'Length: {}, average interval {}, final step {}'.format(self.t_grid.size, np.mean(self.dt),
                                                                                 self.t_grid[-1]),
            'space grid': 'Length: {}, average interval {}, final step {}'.format(self.x_grid.size,
                                                                                  np.mean(self.x_grid),
                                                                                  self.x_grid[-1]),
            'diffusion': self.diff
        }

        printed_informations = '\n'.join([f"{key}: {value}" for key, value in informations.items()])

        return 'Solver object:\n' + printed_informations

    def __add__(self, other):
        """
        Overloading + operator concatenating grids of Solver objects
        Parameters
        ----------
        other: another solver object

        Returns
        -------
        One solver object, which combines grids of both objects
        """
        self.t_grid = np.append(self.t_grid, other.t_grid)
        self.grid = np.append(self.grid, other.grid, axis=-1)

    def correct_l_x_t(self, l, x, t):
        '''
        checking correctnes of the input and changes list into the numpy arrays
        Parameters
        ----------
        l : list or numpy array
        x : list or numpy array
        t : list or numpy array

        Returns
        -------
        l, x, t as numpy arrays
        '''
        if isinstance(x, np.ndarray) == False:
            x = np.array(x)

        if isinstance(t, np.ndarray) == False:
            t = np.array(t)

        if isinstance(l, np.ndarray) == False:
            l = np.array(l)

        if t.size < 2:
            raise ValueError('t grid size cannot be smaller than 2')

        if l.ndim != 1 or x.ndim != 1 or t.ndim != 1:
            raise ValueError('x, l and t should be 1D arrays')

        return l, x, t

    def correct_boundary(self, boundary):
        '''
        checks if boundary is inputted correctly
        Parameters
        ----------
        boundary : None or tuple

        Returns
        -------
        Raises error or unchanged boundary
        '''
        if boundary is None:
            return None

        if isinstance(boundary, tuple) == False or len(boundary) != 2:
            raise TypeError('boundary argument should be a tuple of lenght 2')

        return boundary

    def calc_potential(self, potential):
        """
        Adapts potential function to the modules requirements
        Parameters
        ----------
        potential: function

        Returns
        -------
        potential: array of the potential, which dimensions are adapted for all modules
        """

        if potential is not None:
            potential = potential(self.x_grid)[np.newaxis, :]

        return potential

    def get_grid_spacing(self):
        '''
        Extracts required spacing from x,t, momenta grids
        Returns
        -------
        dx, dl : floats, representing spacing in dimensions
        dt : array, representing spacing in time
        steps_number : int, number of time iterations
        '''
        dx = np.diff(self.x_grid).mean()

        dt = np.diff(self.t_grid)

        dl = np.diff(self.miu_grid).mean()

        steps_number = self.t_grid.size

        return dx, dt, dl, steps_number

    def get_model(self, calculation, *args):
        """
        Associates chosen integrable model with proper modules, calculating
        Parameters
        ----------
        calculation
        args

        Returns
        -------

        """
        model_classes = {
            'Lieb-Liniger': {'TBA': TBA_LiebLiniger, 'velocity': VelocityLiebLiniger, 'diffusion': DiffusionLiebLiniger}
            # Add more models and calculations as needed
        }

        if self.model in model_classes:
            model_class = model_classes[self.model].get(calculation)
            if model_class:
                return model_class(*args)
            else:
                raise ValueError(f"Invalid calculation type: {calculation}")
        else:
            raise ValueError(f"Invalid model type: {self.model}")

    def create_initial_grid(self):
        '''
        Creates initial grid filled with initial conditions
        Returns
        -------
        grid: numpy array which is a grid for whole calculation filled only with initial state
        '''
        # dimensions: l, x, t

        grid = np.zeros((self.miu_grid.size, self.x_grid.size, self.t_grid.size))

        initial_state = self.rho0(self.miu_grid[:, np.newaxis], self.x_grid[np.newaxis, :])

        grid[Ellipsis, 0] = initial_state

        return grid

    def create_matrix(self, time):
        '''

        Parameters
        ----------
        time : int, defines position in time for which the calculation will be done

        Returns
        -------
        numpy array which is a matrix required for implicit solving of GHD
        '''
        # dimensions l, x for rho and V

        rho = self.grid[Ellipsis, time]

        # calculating V using class CalcV

        V = self.get_model('velocity', rho, self.miu_grid, self.coupling).V

        # V = CalcV(rho, self.l, self.c).V

        # changing indices back to proper order

        V = np.einsum('xl -> lx', V)

        h = self.dt[time] / (2 * self.dx)

        V_V_h = h * V[Ellipsis, np.newaxis, :] * np.ones_like(V)[Ellipsis, np.newaxis]

        off_diagonal = np.zeros((V_V_h.shape[-1], V_V_h.shape[-1]))

        np.fill_diagonal(off_diagonal[:, 1:], 1)
        np.fill_diagonal(off_diagonal[1:, 0:], -1)

        off_diagonal = V_V_h * off_diagonal[np.newaxis, Ellipsis]

        diagonal = np.zeros((V_V_h.shape[-1], V_V_h.shape[-1]))

        np.fill_diagonal(diagonal, 1)

        diagonal = np.ones_like(V_V_h) * diagonal[np.newaxis, Ellipsis]

        matrix = diagonal + off_diagonal

        if self.boundary is None:

            matrix[Ellipsis, 0, -1] = -V_V_h[Ellipsis, 0, -1]
            matrix[Ellipsis, -1, 0] = V_V_h[Ellipsis, -1, 0]

        else:

            matrix[Ellipsis, 0, 0] = self.boundary[0]
            matrix[Ellipsis, 0, 1] = 0

            matrix[Ellipsis, -1, -1] = self.boundary[1]
            matrix[Ellipsis, -1, -2] = 0

        return matrix

    def diff_fixed_point_func(self, rho, rho_next, D, V, time):
        '''
        functions utilizing fixed point iteration method for solving GHD with diffusion
        Parameters
        ----------
        rho : numpy array, state for which it is calculated
        rho_next
        D : numpy  array diffusion operator
        V : numpy array, effective velocity

        Returns
        -------
        foo : array utilized for solving GHD
        diff : array containing the mean difference between initial function and returned function
        '''
        diff = rho_next

        # D, V dimensions x, momenta

        Diffusion_op = np.einsum('xos, sx -> ox', D, uts.x_der(rho_next, self.dx), optimize=True)

        V_rho = np.einsum('xl, lx -> lx', V, rho_next, optimize=True)

        if self.potential is None:

            foo = rho + self.dt[time] * (uts.x_der(Diffusion_op, self.dx) / 2 - uts.x_der(V_rho, self.dx))

        else:

            foo = rho + self.dt[time] * (
                    uts.x_der(Diffusion_op, self.dx) / 2 - uts.x_der(V_rho, self.dx) + uts.x_der(self.potential,
                                                                                                 self.dx) * uts.lambda_der(
                rho, self.dl))

        diff = np.abs(diff - foo).mean()

        return foo, diff

    def diff_equ(self, time):
        '''
        solves GHD at given time with diffusion using fixed point iteration method
        Parameters
        ----------
        time : int

        Returns
        -------
        rho_next: numpy array, state at further time
        diff : list, collects convergence of the method
        '''
        rho_next = self.grid[Ellipsis, time]

        rho = self.grid[Ellipsis, time]

        Diff = self.get_model('diffusion', rho, self.miu_grid, self.coupling)

        # Diff = CalcD(rho, self.l, self.c)

        D, V = Diff.D, Diff.V  # dimensions N, x, momenta

        diff = []

        for i in range(15):
            rho_next, difference = self.diff_fixed_point_func(rho, rho_next, D, V, time)

            diff.append(difference)

        return rho_next, diff

    def solve_equation(self, path=None, starting_point=0):
        '''
        Main function evolving the state
        Parameters
        ----------
        path : optional string, if is not None, Solver objects is saved to localization

        -------

        '''

        if self.diff is False:

            for time_step in tqdm(range(starting_point, self.grid.shape[-1] - 1)):
                matrix = self.create_matrix(time_step)

                self.grid[Ellipsis, time_step + 1] = np.linalg.solve(matrix, self.grid[Ellipsis, time_step])

        else:

            for time_step in tqdm(range(starting_point, self.grid.shape[-1] - 1)):
                rho_next, diff = self.diff_equ(time_step)

                self.grid[Ellipsis, time_step + 1] = rho_next

                self.convergence.append(diff)

        if path is not None:
            with open(path, 'wb') as file:
                dill.dump(self, file)

    def continue_calculations(self, elongation_factor, time_array=None, path=None):
        """
        Elongates the solution of GHD equations, either by extending existing time grid by a factor or by concatenating specified
        new time grid

        Parameters
        ----------
        elongation_factor : float, specifies how much the time grid should be elongated
        time_array : array in time grid which should be concatenated with the existing one
        path : sting, if not None, calculation will be saved to the localization directed by path

        -------

        """
        if time_array is None:

            new_t = np.arange(self.t_grid[-1] + np.mean(self.dt), self.t_grid[-1] * (1 + elongation_factor),
                              np.mean(self.dt))

        else:
            new_t = time_array

        new_grid = np.zeros((self.miu_grid.size, self.x_grid.size, new_t.size))

        self.grid = np.concatenate((self.grid, new_grid), axis=-1)

        starting_point = self.t_grid.size - 1

        self.t_grid = np.hstack((self.t_grid, new_t))

        self.dt = np.diff(self.t_grid)

        self.solve_equation(path=path, starting_point=starting_point)

    def save_array(self, path):
        '''
        Saves array of the solution as a binary dill file
        Parameters
        ----------
        path : string, location to which array is saved

        '''
        with open(path, 'wb') as file:
            dill.dump(self.grid, file)

    def save(self, path):
        '''
        Saves Solver object as binary dill file to path localization
        Parameters
        ----------
        path : string, location to which object is saved
        '''
        with open(path, 'wb') as file:
            dill.dump(self, file)

            # loading arrays

    @staticmethod
    def load(path):
        '''
        Loads array or Solver object from the binary file
        Parameters
        ----------
        path : string, localizaction of binary file

        Returns
        -------
        Solver object from the path localization
        '''
        with open(path, 'rb') as file:
            arr = dill.load(file)

        return arr
