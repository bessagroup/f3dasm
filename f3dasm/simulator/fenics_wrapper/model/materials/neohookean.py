from dolfin import *

from f3dasm.simulator.fenics_wrapper.model.materials.material import Material
from f3dasm.simulator.fenics_wrapper.utils.material_tensor_ops import *


class NeoHookean(Material):
    """
        Neo-Hookean material model implementation
    """
    def __init__(self, u, F_macro, E=None, nu=None, mu=None, lmbda=None):

        """ Initialize """

        ################################
        # Initialize material properties
        ################################

        if E is not None and nu is not None:
            self.mu, self.lmbda = Lame(E,nu)
        else:
            self.mu, self.lmbda = mu, lmbda
        
        self.mu = Constant(self.mu)

        self.K = Constant(self.lmbda + 2./3. * self.mu)
        self.C1 = self.mu/2. 
        self.D1 = self.K /2.
        
        Material.__init__(self,u=u, F_macro=F_macro)    # Initialize base-class

    def Energy(self):

        """ Method: Implement energy density function """

        #psi = self.mu/2*(tr(self.b)-3-2*ln(self.J))+self.K/2*(self.J-1)**2
        psi = (self.mu/2)*(tr(self.C)- 3) - self.mu*ln(self.J) + (self.lmbda/2)*(ln(self.J))**2

        return psi