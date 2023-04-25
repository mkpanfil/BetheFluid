import numpy as np
import matplotlib.pyplot as plt
from CalcV import CalcV

class Solver():
    
    def __init__(self, t, x, l, n0):
        
        self.t = t
        self.x = x
        self.l = l
        self.n0 = n0
        self.int_x, self.int_t, self.steps_number = self.create_ints()
        self.initial_grid = self.create_initial_grid()
        self.matrix = self.create_matrix()
        self.grid = self.solve_equation()
        
    # extracts required data from x,t grids  
    def create_ints(self):
        
        int_x = np.diff(self.x).mean()
    
        int_t = np.diff(self.t).mean()
        
        steps_number = self.t.size
        
        return int_x, int_t, steps_number
      
    #creates grid filled with initial condition   
    # order of dimensions in the grid: momentum l, x, t
    def create_initial_grid(self):
        
        grid = np.zeros((self.l.size, self.x.size, self.steps_number))
        
        l_matrix = self.l.reshape(-1,1)
        
        n0_matrix = self.n0(self.x, l_matrix)       
        
        grid[:,:,0] = n0_matrix
        
        return grid
    
    def create_matrix(self):       
        
        V = CalcV(self.initial_grid, self.l).V
        
        length = self.initial_grid[0,:,0]
        
        matrices_list = []
        
        for index in range(self.l.size): 
            
            r = self.int_t/(2*self.int_x) * V[index]
    
            h_matrix = np.zeros((len(length),len(length)))

            h_matrix[0,-1], h_matrix[0,0], h_matrix[0,1]  = -r, 1, r

            matrix = np.zeros((len(length), len(length)))
    
            for index in range(len(length)):
    
                matrix = np.roll(h_matrix, (index,index), axis = (0,1)) + matrix
    
            matrices_list.append(matrix)
            
        matrix = np.stack(matrices_list)  
        
        return matrix
    
    def solve_equation(self):
        
        
        grid = self.initial_grid
        
        for t in range(self.steps_number - 1):
            
            n1 = np.linalg.solve(self.matrix, self.initial_grid[:,:, t])
            
            grid[:,:, t + 1] = n1
      
        return grid
       
    
    #graph solution for given momenta
    def do_graph(self, momenta):
    
        i = 0
        
        step = int(round(self.steps_number/10, 0))
        
        while i <= self.grid.shape[2] - 1:

            plt.plot(self.x, self.grid[momenta,:,i], label = "t = {}".format(round(i*self.int_t, 3)))
            plt.legend()                       
            i = i + step
       