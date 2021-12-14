from dolfin import * 
from .bc import *
from ufl import algorithms
from .continuum import * 

class model():
    """ 
        GENERAL MODEL FENICS IMPLEMENTATION 

        *** Base model class where you combine your domain, boundary
                condtions information to solve a BVP
        *** Inherit from this class, but have to provide the variational
                problem and solution parameters.
    """
    def __init__(self, domain, bc, Ve=None,module_name='FEniCS'):

        """ Initialize """

        self.module_name = module_name                  # Name module
        self.domain= domain                             # Get domain
        self.bc = bc                                    # Get boundary conditions
        
        ################################
        # Define function spaces for the problem
        ################################
        if Ve is None:
            self.Ve = VectorElement("CG", domain.mesh.ufl_cell(), 1)

        ################################
        # Define function space for the pure Neumann Lagrange multiplier
        ################################
        self.Re = VectorElement("R", domain.mesh.ufl_cell(), 0)

    def problem(self):
        """ Method: Variational problem definition """
        pass

    def solver(self):
        """ Method: Solver implementation """
        pass

    def postprocess(self):
        """ Method: Postprocesssing implementation """
        pass

        
