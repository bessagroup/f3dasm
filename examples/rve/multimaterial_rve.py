from f3dasm.simulator.fenics_wrapper.model.bc import PeriodicBoundary
from f3dasm.simulator.fenics_wrapper.model.domain import Domain
from f3dasm.simulator.fenics_wrapper.model.materials.neohookean import NeoHookean
from f3dasm.simulator.fenics_wrapper.model.materials.svenan_kirchhoff import SVenantKirchhoff
from f3dasm.simulator.fenics_wrapper.problems.problem_base import ProblemBase 
from f3dasm.simulator.fenics_wrapper.postprocessor.project_field import *

from dolfin import *
import numpy as np
import copy as cp
import os

class MultiMaterialRVE(ProblemBase):
    """ 
        General RVE model implementation
    """
    def __init__(self, options, domain_filename, name=None):
        domain = Domain(domain_filename)
        super().__init__(options=options, name=name, domain=domain)

        self.bc_p = PeriodicBoundary(domain,periodicity=list(range(domain.dim)),tolerance=1e-10)   # Initialize periodic boundary conditions
        # Mixed function space initialization with periodic boundary conditions
        self.W = FunctionSpace(self.domain.mesh, MixedElement([self.Ve, self.Re]), constrained_domain=self.bc_p)

        self.v_, self.lamb_ = TestFunctions(self.W)     # Define test functions 
        self.w = Function(self.W)

        self.time = 0
        
        
    def solve(self, F_macro, model_tag, work_dir):
        if not os.path.exists(work_dir):
            os.mkdir(work_dir)
        self.work_dir = work_dir
        
        self.fileResults = XDMFFile(os.path.join(work_dir, "output.xdmf"))
        self.fileResults.parameters["flush_output"] = True
        self.fileResults.parameters["functions_share_mesh"] = True

        self.convergence = True

        F_macro = Constant(F_macro)                

        u,c = split(self.w)

        ################################
        # Define materials for phases
        ################################
        model1 = [NeoHookean(u,F_macro, E=300, nu=0.1),SVenantKirchhoff(u,F_macro,E=500,nu=0.3)]        # model-1
        model2 = [NeoHookean(u,F_macro, E=300, nu=0.1),NeoHookean(u,F_macro,E=300,nu=0.1)]              # model-2
        model3 = [NeoHookean(u,F_macro, E=300, nu=0.),NeoHookean(u,F_macro,E=300,nu=0.)]                # model-3
        model4 = [NeoHookean(u,F_macro, E=300, nu=0.1),SVenantKirchhoff(u,F_macro,E=500,nu=0.1)]        # model-4
        model5 = [NeoHookean(u,F_macro, E=300, nu=0.1),SVenantKirchhoff(u,F_macro,E=800,nu=0.3)]        # model-5
        models = {1:model1, 2:model2, 3:model3, 4:model4,5:model5}

        self.material = models[model_tag]

        dx = Measure('dx')(subdomain_data=self.domain.subdomains)       # Redefine dx for subdomains

        ################################
        # Variational problem definition -> Lagrangian Linear Momentum Equation
        ################################
        self.PI = inner(self.material[0].P,nabla_grad(self.v_))*dx(1) + inner(self.material[1].P,nabla_grad(v_))*dx(2)  
        
        self.PI += dot(self.lamb_,u)*dx + dot(c,self.v_)*dx
        
        self._solve()
        
        project_u(self)


    def _solve(self):
        prm = {"newton_solver":
                {"absolute_tolerance":1e-7,'relative_tolerance':1e-7,'relaxation_parameter':1.0,'linear_solver' : 'mumps'}}
        try:
            solve(self.PI==0, self.w, [],solver_parameters=prm,form_compiler_parameters={"optimize": True},)
            (self.v, lamb) = self.w.split(True)
        except:
            self.convergence = False


    def postprocess(self):
        """ 
            Method: postprocess implementation to get the homogenized 
                    second Piola-Kirchhoff stress tensor 
        """
        if self.convergence is False:
            S = np.zeros((self.domain.dim,self.domain.dim))
        else:
            
            P = project_P(self)                          # Project first Piola-Kirchhoff stress tensor 
            F = project_F(self)                          # Project Deformation Gradient

            Piola = P.split(True)
            DG = F.split(True)
            P_hom = np.zeros(self.domain.dim**2) 
            F_hom = np.zeros(self.domain.dim**2) 
            
            for i in range(self.domain.dim**2):
                P_hom[i] = np.dot(Piola[i].vector().get_local(),self.domain.ele_vol)/self.domain.vol
                F_hom[i] = (DG[i].vector().get_local().mean())

            P_hom = P_hom.reshape(-1,self.domain.dim)
            F_hom = F_hom.reshape(-1,self.domain.dim)
            S = np.dot(np.linalg.inv(F_hom.T),P_hom)

        return S, F_hom
