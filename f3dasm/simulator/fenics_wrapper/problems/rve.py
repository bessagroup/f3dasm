from f3dasm.simulator.fenics_wrapper.model.bc import PeriodicBoundary
from f3dasm.simulator.fenics_wrapper.model.domain import Domain
from f3dasm.simulator.fenics_wrapper.model.materials.neohookean import NeoHookean
from f3dasm.simulator.fenics_wrapper.model.materials.svenan_kirchhoff import SVenantKirchhoff
from f3dasm.simulator.fenics_wrapper.problems.problem_base import ProblemBase 

from dolfin import *
import numpy as np
import copy as cp
import os


class RVE2D(ProblemBase):
    """ 
        General RVE model implementation
    """
    def __init__(self, options, domain_filename, name=None):
        # import domain from file
        domain = Domain(domain_filename)

        super().__init__(options=options, name=name, domain=domain)
        
        self.bc_p = PeriodicBoundary(self.domain,periodicity=[0,1],tolerance=1e-10)   # Initialize periodic boundary conditions
        # Mixed function space initialization with periodic boundary conditions
        self.W = FunctionSpace(self.domain.mesh, MixedElement([self.Ve, self.Re]), constrained_domain=self.bc_p)

        self.v_, self.lamb_ = TestFunctions(self.W)      # Define test functions 
        self.w = Function(self.W)
        

    def solve(self, F_macro, work_dir):
        if not os.path.exists(work_dir):
            os.mkdir(work_dir)
        self.work_dir = work_dir

        ################################
        # F_macro should be defined locally because when passing in another way
        # it gives erroneous results! So for consistancy it is defined just before
        # passing as Constant from fenics.
        ################################
        self.convergence = True

        F_macro = Constant(F_macro)                

        u,c = split(self.w)

        ################################
        # Define materials for phases
        ################################
        self.material = [NeoHookean(u,F_macro, E=300, nu=0.1),SVenantKirchhoff(u,F_macro,E=500,nu=0.3)]        # model-1

        
        dx = Measure('dx')(subdomain_data=self.domain.subdomains)       # Redefine dx for subdomains

        ################################
        # Variational problem definition -> Lagrangian Linear Momentum Equation
        ################################
        self.PI = inner(self.material[0].P,nabla_grad(self.v_))*dx(1) + inner(self.material[1].P,nabla_grad(self.v_))*dx(2)  

        self.PI += dot(self.lamb_,u)*dx + dot(c,self.v_)*dx

        """ Method: Define solver options with your solver """

        prm = {"newton_solver":
                {"absolute_tolerance":1e-7,'relative_tolerance':1e-7,'relaxation_parameter':1.0}}
        try:
            solve(self.PI==0, self.w, [],solver_parameters=prm,form_compiler_parameters={"optimize": True})
            (self.v, lamb) = self.w.split(True)
        except:
            self.convergence = False
       # self.__deformed()


    def postprocess(self):
        """ 
            Method: postprocess implementation to get the homogenized 
                    second Piola-Kirchhoff stress tensor 
        """
        if self.convergence is False:
            S = np.zeros((self.domain.dim,self.domain.dim))
        else:
            
            P = self.__project_P()                          # Project first Piola-Kirchhoff stress tensor 
            filename = File(os.path.join(self.work_dir,'stress.pvd'))
            filename << P
            F = self.__project_F()                          # Project Deformation Gradient

            Piola = P.split(True)
            DG = F.split(True)
            P_hom = np.zeros(self.domain.dim**2) 
            F_hom = np.zeros(self.domain.dim**2) 
            
            for i in range(self.domain.dim**2):
                for j in range(self.domain.ele_num):
                    P_hom[i] = np.dot(Piola[i].vector().get_local(),self.domain.ele_vol)/self.domain.vol

                    F_hom[i] = (DG[i].vector().get_local().mean())

            P_hom = P_hom.reshape(-1,self.domain.dim)
            F_hom = F_hom.reshape(-1,self.domain.dim)
            S = np.dot(np.linalg.inv(F_hom.T),P_hom)

        return S, F_hom

    def __deformed(self):

        """ Method: output the deformed state to a file """

        V = FunctionSpace(self.domain.mesh,self.Ve)
        y = SpatialCoordinate(self.domain.mesh)
        write = dot(Constant(self.F_macro),y)+self.v
        filename = File(os.path.join(self.work_dir, "deformation.pvd"))
        filename << project(write,V)

        ################################
        # Easy ploting for the 2D deformation
        ################################
        #y = SpatialCoordinate(self.domain.mesh)
        ##F = Identity(self.domain.dim) + grad(self.v) + Constant(self.F_macro)              # Deformation gradient
        #p = plot(dot(Constant(self.F_macro),y)+self.v, mode="displacement")
        ##p = plot(self.v, mode="displacement")
        ##p = plot(self.stress[0, 0])
        #plt.colorbar(p)
        #plt.savefig("rve_deformed.pdf")

    def __project_P(self):

        """ 
            Method: Projecting first Piola-Kirchhoff stress tensor.
                    Another linear variational problem has to be solved.
        """

        V = TensorFunctionSpace(self.domain.mesh, "DG",0)           # Define Discontinuous Galerkin space

        ################################
        # Similar type of problem definition inside the model
        ################################
        dx = Measure('dx')(subdomain_data=self.domain.subdomains)   
        dx = dx(metadata={'quadrature_degree': 1})
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv,v_)*dx
        b_proj = inner(self.material[0].P,v_)*dx(1) +inner(self.material[1].P,v_)*dx(2)
        P = Function(V)
        solve(a_proj==b_proj,P)
        return P

    def __project_F(self):

        """ 
            Method: Projecting deformation gradient.
                    Another linear variational problem has to be solved.
        """

        ################################
        # Similar type of problem definition inside the model
        ################################
        V = TensorFunctionSpace(self.domain.mesh, "DG",0)       # Define Discontinuous Galerkin space

        dx = Measure('dx')(subdomain_data=self.domain.subdomains)
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv,v_)*dx
        b_proj = inner(self.material[0].F,v_)*dx(1) +inner(self.material[1].F,v_)*dx(2)
        F = Function(V)
        solve(a_proj==b_proj,F)
        return F