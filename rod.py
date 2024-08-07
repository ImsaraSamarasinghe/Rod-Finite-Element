import numpy as np
import matplotlib.pyplot as plt

class Solver:
    def __init__(self,K,F,node1=np.array([0,0]),node2=np.array([1,0])):
        self.K=K
        self.F=F
        self.BCs = ['Dirichlet','Neumann']
        self.node1=node1
        self.node2=node2

    
    def apply_boundary_conditions(self):
        for i, bc in enumerate(self.BCs):
            if bc == 'Dirichlet':
                self.K[i, :] = 0
                self.K[:, i] = 0
                self.K[i, i] = 1
                self.F[i] = 0
    
    def solve(self):
        self.apply_boundary_conditions()
        return np.linalg.solve(self.K, self.F)
    
    def plot(self):
        u = self.solve()
        # Calculate the displaced positions
        new_node1 = self.node1 + np.array([u[0,0], 0])
        new_node2 = self.node2 + np.array([u[1,0], 0])

        # Extract x and y coordinates for plotting
        x_original = [self.node1[0], self.node2[0]]
        y_original = [self.node1[1], self.node2[1]]

        x_displaced = [new_node1[0], new_node2[0]]
        y_displaced = [new_node1[1], new_node2[1]]

        plt.figure()
        plt.plot(x_original, y_original, linestyle='--', marker='o', label='Original')
        plt.plot(x_displaced, y_displaced, linestyle='-.', marker='o', label='Displaced')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.savefig('disp.png')
        
  

class RodElement:
    def __init__(self,E=1e6,node1=np.array([0,0]),node2=np.array([1,0])):
        self.E = E
        self.node1 = node1
        self.node2 = node2
        self.BCs = ['Dirichlet','Neumann'] # left and right
        self.Force = 1e5
        
        
    # Material Property matrix
    def D(self):
        return self.E

    # B matrix
    def B(self):
        J = 1/(self.node2[0]-self.node1[0]) # Jacobian
        return np.array([-1, 1])*J
    
    # define local K matrix
    def K(self):

        return self.B().reshape(-1,1)*self.D()*self.B()

    # Create force Vector
    def forceVec(self):
        F = np.zeros((2,1))
        
        # node 1
        if self.BCs[0]=='Dirichlet':
            F[0,0] = 0
        else:
            F[0,0] = self.Force
        
        # node 2
        if self.BCs[1]=='Dirichlet':
            F[1,0] = 0
        else:
            F[1,0] = self.Force
        
        return F
            
rod = RodElement()
solver = Solver(rod.K(),rod.forceVec())
solver.plot()