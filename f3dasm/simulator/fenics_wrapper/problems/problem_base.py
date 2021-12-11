from f3dasm.simulator.fenics_wrapper.model.domain import Domain

import dolfin
from typing import Optional, List
import dataclasses

class ProblemBase():
    """ Base class for all problems
    Definition of the variational form of a BVP  
    Attributes;
        name: name of the model
        domain: mesh 
        Ve: function space for the problem
        Re: function space for the pure Neumann Lagrange multiplier
        # u: trial function
        # v: test function
        # bcs: list of boundary conditions

    """
    def __init__(self, options: dict, domain: Domain, name: Optional[str]=None) -> None:
        self.options = options
        self.name = name
        self.domain = domain
        # Define function spaces for the problem
        self.Ve = dolfin.VectorElement("CG", self.domain.mesh.ufl_cell(), 1)
        # Define function space for the pure Neumann Lagrange multiplier
        self.Re = dolfin.VectorElement("R", self.domain.mesh.ufl_cell(), 0)

    # bcs: Optional[List[dolfin.SubDomain]] = None
    # V: Optional[dolfin.FunctionSpace]=None
    # u: Optional[dolfin.TrialFunction] = None
    # v: Optional[dolfin.TestFunction] = None

    def _solve(self):
        return

    def postprocess(self):
        return