import numpy as np

class CalcV():
    
    def __init__(self, rho, l, c):      
              
        # dimensions of rho N, x, l
        self.rho = rho
        self.l = l
        self.c = c
        self.int_l = np.diff(self.l).mean() 
        self.T = self.create_T()
        self.n = self.get_n()
        self.operator = self.get_operator()
        self.V = self.get_V()
    
    
    def create_T(self):
        
        # numba doesn't support meshgrid
        l, u = np.meshgrid(self.l,self.l, indexing = 'ij')
        
        
        T_grid = []
        
        for value in self.c:         
        
            T = value/np.pi * 1/((l-u)**2 + value**2)
            
            T_grid.append(T)
            
        # dimensions N, l, u   
        T = np.stack(T_grid)
        
        return T


    def get_n(self):
        
        
         # change indices of rho
         
        self.rho = np.einsum('Nlx -> Nxl', self.rho)
        
        
        # new indices are rho: N, x, l
        
        rho_tot = 1/(2*np.pi) + np.einsum('nlu, Nxu -> Nxl', self.T, self.rho)*self.int_l
        
        n = self.rho/rho_tot
        
        return n
    
    def get_operator(self):      
        
        # T(N,l,u) , n(N, x, u)
        
        
        # create T*n (N,x, l, u)
        
        
        Tn = self.T[:, np.newaxis, :, :] * self.n[:,:,np.newaxis, :]
        
        
        # create delta l,u for each N and x
        #dimensions N, x, l, u
        
        delta = np.identity(self.l.size)[np.newaxis, np.newaxis, :, :]
               
        operator = delta - Tn * self.int_l
        
        # numba doesn't support in multi-D
        operator = np.linalg.inv(operator)
        
        
        return operator
        
    
    def get_V(self): 
        
        # dimensions N, x, l
        
        
        u = 2*self.l
               
        
        k_dr = np.sum(self.operator, axis = -1)
        
        
        omega_dr = np.einsum('Nxlu, u -> Nxl', self.operator, u)
        
        
        V = omega_dr/k_dr
        
        
        # changing indices back to Solver convention
        
        V = np.einsum('Nxl -> Nlx', V)
        
        
        # order of indices is Nlx
        
        return V
        
    