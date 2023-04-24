class CalcV():
    
    def __init__(self, n, l):      
        self.n = n
        self.l = l
        self.V = self.create_V()
      
    def create_V(self): 
        
        
        V = 2*self.l.reshape(-1,1)     
        
        return V