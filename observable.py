import numpy as np
import matplotlib.pyplot as plt
import dill
import scipy as sci

class Observable():
       
    def __init__(self, Solver_object):
        
        self.object = self.get_object(Solver_object)
        # dimensions: N, l, x, t       
        self.T = self.get_T()
        self.rho_tot = self.get_rho_tot()
        self.rho_h = self.get_rho_h()
        #self.n = self.get_n()
     
        
    def get_object(self, inp):
        
        if isinstance(inp, str):
            
            with open(inp, 'rb') as file:
                arr = dill.load(file)

            return arr
        
        else: return inp
        
    
    def get_T(self):
        
        # numba doesn't support meshgrid
        l, u = np.meshgrid(self.object.l,self.object.l, indexing = 'ij')
              
        T_grid = []
        
        for value in self.object.c:         
        
            T = value/np.pi * 1/((l-u)**2 + value**2)
            
            T_grid.append(T)
            
        # dimensions N, l, u   
        T = np.stack(T_grid)
        
        return T
      
        
    def get_rho_tot(self):      
        
        # new indices are rho: N, x, l
        
        rho_tot = 1/(2*np.pi) + np.einsum('Nlu, Nuxt -> Nlxt', self.T, self.object.grid, optimize = True)*self.object.int_l
        
        return rho_tot
    
    
    def get_rho_h(self):
        
        rho_h = self.rho_tot - self.object.grid
        
        return rho_h
    
    
    def x_der(self, arr):
              
        der = 1/(2*self.int_x) * (np.roll(arr, -1, axis = -1) - np.roll(arr, 1, axis = -1))
        
        #der = np.gradient(arr, self.int_x, axis = -1)
        
        return der
    
    
    def energy_conservation(self, N = 0, path = None, *args, **kwargs):
        
        # Dimensions N, l, x, t
        
        grid = np.sum(self.object.grid, axis = -2) * self.object.int_x
        
        # Dimensions N, l, t
        grid = grid * self.object.l[np.newaxis, :, np.newaxis]**2
        
        grid = np.sum(grid, axis = 1) * self.object.int_l
        
        plt.plot(self.object.t, grid[N,:], *args, **kwargs)
        plt.title('Energy conservation of the N-th particle')
        plt.ylabel('energy')
        plt.xlabel('time')
        
        
        if path is not None:
            
            plt.savefig(path)
    
    
    def density_conservation(self, N = 0, path = None, *args, **kwargs):
        
        # Dimensions N, l, x, t
        
        grid = np.sum(self.object.grid, axis = (1,2)) * self.object.int_x * self.object.int_l

        plt.plot(self.object.t, grid[N, :], *args, **kwargs)
        plt.title('Density conservation of N-th particle')
        plt.ylabel(r'$\rho$')
        plt.xlabel('time')
        
        if path is not None:
            plt.savefig(path)
 
    # here I can make it better if the index is negative
    def n(self, N = 0 , frames = [0, -1], path = None, name = '', style = '-'):
        
        grid = np.sum(self.object.grid, axis = 1) * self.object.int_l 
        
        for item in frames:
            
            plt.plot(self.object.x, grid[N, :, item], style, label = '{} t = {}'.format(name, round(item * self.object.int_t, 3)))
            plt.xlabel('x')
            plt.ylabel(r'$\rho$')
            plt.legend()
    
    
    def n_lambda(self, N = 0, t = -1, path = None, *args, **kwargs):
        
        grid = np.sum(self.object.grid, axis = 2) * self.object.int_x
    
        plt.plot(self.object.l, grid[N, :, t])
        
        if path is not None:            
            plt.savefig(path)
        
          
    def enthropy(self, N = 0, path = None, *args, **kwargs):
        
        # Dimensions N, l, x, t
        
        rho = self.object.grid
        
        rho[rho < 0] = 0
        
        s = self.rho_tot * np.log(self.rho_tot) - sci.special.xlogy(rho, rho) - sci.special.xlogy(self.rho_h, self.rho_h)
               
        #S = np.sum(s, axis = (1,2)) * self.object.int_x * self.object.int_l
        
        
        return s
        
        #plt.plot(self.object.t, S[N, :], *args, **kwargs)
        #plt.plot(self.object.t, S[N, 40, :], label = r'$\lambda$ = 40', *args, **kwargs)
        #plt.plot(self.object.t, S[N, 20, :], label  = r'$\lambda$ = 20', *args, **kwargs)
        #plt.plot(self.object.t, S[N, 0, :], label  = r'$\lambda$ = 0', *args, **kwargs)
        #plt.title('Enthropy of N-th particle')
        #plt.ylabel(r'$S$')
        #plt.xlabel('time')
        #plt.legend()
        
        if path is not None:
            plt.savefig(path)
    
     
        