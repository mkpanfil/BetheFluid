import numpy as np
import matplotlib.pyplot as plt
from CalcV import CalcV

class Solver():
    
    def __init__(self, N, l, x, t, rho0, c):
        
        
        self.N = N
        self.l = l
        self.x = x
        self.t = t
        self.c = c
        self.rho0 = rho0
        self.int_x, self.int_t, self.steps_number = self.get_ints()
        self.grid = self.create_initial_grid()
        
        
    # extracts required data from x,t grids  
    def get_ints(self):
        
        int_x = np.diff(self.x).mean()
    
        int_t = np.diff(self.t).mean()
        
        steps_number = self.t.size
        
        return int_x, int_t, steps_number
      
    #creates grid filled with initial condition   
    # order of dimensions in the grid: momentum N, x, l, t
    
    # for now N dimension is just stucking x, l, t
    
    def create_initial_grid(self):
        
        # dimensions: N, x, l, t
        
        grid = np.zeros((self.N, self.l.size, self.x.size, self.t.size))
        
        N = np.ones(self.N)
        
        N = N[:, np.newaxis, np.newaxis]
        
        rh0_matrix = self.rho0(self.l[:, np.newaxis], self.x[np.newaxis, :])
        
        initial_state = N* rh0_matrix[np.newaxis, :, :]
        
        grid[Ellipsis,0] = initial_state
        
        return grid
          
        
    
    def create_matrix(self, time):
        
        #dimensions N, l, x for rho and V
        
        rho = self.grid[Ellipsis, time]
        
        
        # calculating V using class CalcV
        
        V = CalcV(rho, self.l, self.c).V
                
        
        h = self.int_t/2*self.int_x
        
        
        V_V_h = h*V[Ellipsis,np.newaxis, :] * np.ones_like(V)[Ellipsis, np.newaxis]
        
        
        off_diagonal = np.zeros((V_V_h.shape[-1], V_V_h.shape[-1]))
             
        
        np.fill_diagonal(off_diagonal[:, 1:], 1)
        np.fill_diagonal(off_diagonal[1:, 0:], -1)
        
        off_diagonal = V_V_h * off_diagonal[np.newaxis, np.newaxis, Ellipsis]
        
        
        diagonal = np.zeros((V_V_h.shape[-1], V_V_h.shape[-1]))
        
        np.fill_diagonal(diagonal, 1)
        
        diagonal = np.ones_like(V_V_h)* diagonal[np.newaxis, np.newaxis, Ellipsis]
        
        matrix = diagonal + off_diagonal
               
        
        
        matrix[Ellipsis, 0, -1] = -V_V_h[Ellipsis, 0, -1]
        
        matrix[Ellipsis, -1, 0] = V_V_h[Ellipsis, -1, 0]
        
        return matrix
        
    def solve_equation(self):
               
        
        for time in range(self.t.size - 1):
            
            
            matrix = self.create_matrix(time)
        
            
            self.grid[Ellipsis, time + 1] = np.linalg.solve(matrix, self.grid[Ellipsis, time])      
            
       
    
    #graph solution for given N
    def do_graph(self, N, l):
    
        i = 0
        
        step = int(np.ceil(self.steps_number/10))
        
        #grid dimensions are N, l, x, t 
        
        #grid = np.sum(self.grid, axis = 1)
        
        # after summation: N, x, t
        
        grid = self.grid
        
        while i <= self.grid.shape[-1] - 1:

            plt.plot(self.x, grid[N, l, :, i], label = "t = {}".format(round(i*self.int_t, 3)))
            plt.legend()                       
            i = i + step
  
      
     