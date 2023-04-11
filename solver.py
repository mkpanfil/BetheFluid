import numpy as np
import matplotlib.pyplot as plt

class Solver():
    
    
    def __init__(self,steps_number,x, v, n0):
        
        self.steps_number = steps_number
        self.x = x
        self.v = v
        self.n0 = n0      
        self.int_x, self.int_t = self.create_ints()      
        self.grid = self.solve_equation()
        
        
    def create_ints(self):
        
        int_x = np.diff(self.x).mean()
    
        int_t = int_x/10
        
        return int_x, int_t
              
    
    def time_step(self, time, grid):
        
        value = grid[:,time]   

        x_dt = self.int_t/self.int_x * (np.roll(value,1) - np.roll(value,-1)) + value
        
        return x_dt
    
    
    def solve_equation(self):
              
        
        grid = np.zeros((len(self.x), self.steps_number))
    
        grid[:,0] = self.n0(self.x)
        
        
        i = 0

        while i < self.steps_number -1:
    
            grid[:, i + 1] = self.time_step(i, grid)
    
            i = i + 1
    
    
        return grid
        
    
    def do_graph(self, step):
    
        i = 0

        while i <= len(self.grid[0]) - 1:

            plt.plot(self.x, self.grid[:,i], label = "t = {}".format(round(i*self.int_t, 3)))
            plt.legend()                       
            i = i + step
             