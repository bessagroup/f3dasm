from dolfin import *
from fenics import variable
from ..util import *

class Material():
    """
        GENERAL MATERIAL RELATION IMPLEMENTATION DEFORMATION (STRAIN) -> FORCE (STRESS)

        ***NOTE: F_macro is needed for multiscale formulations
    
    """
    def __init__(self, u, F_macro=None):

        d = len(u)                                      # Dimension of the problem
        
        self.I = Identity(d)                            # Identity tensor

        self.F = self.I + nabla_grad(u)                       # Deformation gradient
        
        if F_macro is not None:
            self.F += F_macro                           # Macroscopic deformation gradient
        
        self.F = variable(self.F)                       # Define deformation gradient as variable to be able to differentiate later on
        #self.J = det(self.F)

        self.I1_F, self.I2_F, self.J = Invariants(self.F)       # Calculate the Invariants of deformation gradient
        
        self.C = self.F.T * self.F                      # Right Cauchy-Green strain tensor
        self.b = self.F * self.F.T                      # Left Cauchy-Green strain tensor
        
        self.I1_C, self.I2_C, self.I3_C = Invariants(self.C)    # Calculate the Invariants of Right Cauchy-Green strain tensor
        self.I1_b, self.I2_b, self.I3_b = Invariants(self.b)    # Calculate the Invariants of Left Cauchy-Green strain tensor

        self.E = 0.5 * (self.C - self.I)                # Green strain tensor

        self.e = self.push_forward_defo(self.E)         # Almanso strain tensor

        self.psi = self.Energy()                        # Energy density function

        self.P = diff(self.psi, self.F)                 # First Piola-Kirchhoff stress tensor

        self.S = inv(self.F)*self.P                     # Second Piola-Kirchhoff stress tensor

    def push_forward_defo(self, A):
        """ Method: Push-forward operation for deformation measures """
        return inv(self.F.T) * A * inv(self.F)

    def pull_back_defo(self, A):
        """ Method: Pull-back operation for deformation measures """
        return self.F.T * A * self.F
    
    def Energy(self):
        """ Method: Energy density function definition """
        pass


