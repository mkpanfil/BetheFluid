# probably better alternative:

class CalcV():
    
    def __init__(self, n, l):      
        self.n = n
        self.l = l
        self.V = self.create_V()
      
    def modify_input(self):
        pass
    
    def create_V(self):  
        return 1