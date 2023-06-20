import numpy as np
from CalcV import CalcV



class Solver():
    
    def __init__(self, l = None, x = None, t = None, rho0 = None, c = None, boundary = None):
        
               
        self.l, self.x, self.t = self.correct_l_x_t(l, x, t)
        self.c, self.rho0 = self.to_tuple(c, rho0)
        self.boundary = self.correct_boundary(boundary)
        self.int_x, self.int_t, self.steps_number = self.get_ints()
        self.grid = self.create_initial_grid()
    
    def correct_l_x_t(self, l, x, t):
        
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
        
        if boundary is None:
            return None
        
        if isinstance(boundary, tuple) == False or len(boundary) != 2:
            
            raise TypeError('boundary argument should be a tuple of lenght 2')
            
        return boundary
    
    
    # ensures that c and rho0 are tuple and have this same size
    def to_tuple(self, c, rho0):
        
        if isinstance(c, tuple) == False:
    
            c = (c,)
        
        
        if isinstance(rho0, tuple) == False:
    
            rho0 = (rho0,)
            
        if len(c) != len(rho0):
            
            raise TypeError('c and rho0 have to have this same size')
            
        return c, rho0
    
    
    # extracts required data from x,t grids  
    def get_ints(self):
        
        int_x = np.diff(self.x).mean()
    
        int_t = np.diff(self.t).mean()
        
        steps_number = self.t.size
        
        return int_x, int_t, steps_number
    

    # creates grid filled with initial condition 
    def create_initial_grid(self):
        
        # dimensions: N, l, x, t
        
        grid = np.zeros((len(self.c), self.l.size, self.x.size, self.t.size))
        
        N_dim = []      
        
        for item in self.rho0:
            
            N = item(self.l[:, np.newaxis] , self.x[np.newaxis, :])    
        
            N_dim.append(N)
        
        initial_state = np.stack(N_dim)
        
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
        
               
        if self.boundary is None:
            
            matrix[Ellipsis, 0, -1] = -V_V_h[Ellipsis, 0, -1]
            
            matrix[Ellipsis, -1, 0] = V_V_h[Ellipsis, -1, 0]
            
        else:
            
            matrix[Ellipsis, 0, 0] = self.boundary[0]
            
            matrix[Ellipsis, 0, 1] = 0
            
            matrix[Ellipsis, -1, -1] = self.boundary[1]
            matrix[Ellipsis, -1, -2] = 0
            
        
        return matrix
        
    
    
    def solve_equation(self):
               
        
        for time in range(self.t.size - 1):
            
            
            matrix = self.create_matrix(time)
        
            
            self.grid[Ellipsis, time + 1] = np.linalg.solve(matrix, self.grid[Ellipsis, time])      
            
  
      
     