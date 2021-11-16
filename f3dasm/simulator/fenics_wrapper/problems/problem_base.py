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
        V: function space
        u: trial function
        v: test function
        bcs: list of boundary conditions

    """
    def __init__(self, options: dict, domain: Domain, name: Optional[str]=None) -> None:
        self.options = options
        self.name = name
        self.domain = domain
    # bcs: Optional[List[dolfin.SubDomain]] = None
    # V: Optional[dolfin.FunctionSpace]=None
    # u: Optional[dolfin.TrialFunction] = None
    # v: Optional[dolfin.TestFunction] = None