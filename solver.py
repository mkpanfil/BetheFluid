import numpy as np
import matplotlib.pyplot as plt
from CalcV import CalcV

class Solver():
    
    
    def __init__(self,steps_number,x, l, u, n0):
        
        self.steps_number = steps_number
        self.x = x
        self.l = l
        self.n0 = n0
        self.u = u
        self.int_x, self.int_t = self.create_ints()      
        self.grid = self.solve_equation()
        
    #finds appropriate time step for explicit method, given x grid  
    def create_ints(self):
        
        int_x = np.diff(self.x).mean()
    
        int_t = int_x/10
        
        return int_x, int_t
      
    #creates grid filled with initial condition   
    # order of dimensions in the grid: momentum l, x, t
    def create_initial_grid(self):
        
        grid = np.zeros((self.l.size, self.x.size, self.steps_number))
        
        l_matrix = self.l.reshape(-1,1)
        
        n0_matrix = self.n0(self.x, l_matrix, self.u)
        
        grid[:,:,0] = n0_matrix
        
        return grid
        
        
    # finds n in another time step
    def time_step(self, time, grid):
        
        #array of n values at some time step 
        value = grid[:,:,time]   
        
        # passing value and momenta array to CalcV class and getting V
        V = CalcV(value, self.l).V
        
        x_dt = self.int_t/self.int_x * (np.roll(V*value, 1, axis = 1) - np.roll(V*value, -1, axis = 1)) + value
        
        return x_dt
    
    # fulfils grid from initiall conditions of solutions, using time step
    def solve_equation(self):
              
        
        grid = self.create_initial_grid()
                
        i = 0

        while i < self.steps_number -1:
    
            grid[:,:, i + 1] = self.time_step(i, grid)
    
            i = i + 1
    
    
        return grid
        
    
    #plots a graph of solution for given momenta
    def do_graph(self, step, momenta):
    
        i = 0

        while i <= len(self.grid[momenta,0,:]) - 1:

            plt.plot(self.x, self.grid[momenta,:,i], label = "t = {}".format(round(i*self.int_t, 3)))
            plt.legend()                       
            i = i + step
       
    #used for learning
    def trying(self):
        
        V = CalcV(self.grid[0,:,0])
        
        self.grid[0,:,0] = V.V