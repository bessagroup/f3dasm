from ..src.model import *
from ..materials.hyperelastic import *
from ..src.bc import *
import matplotlib.pyplot as plt

################################################################################################################
# Notes: General RVE models with periodic boundary conditions for composite materials. 3D and 2D are seperated 
# for the sake of computational time as the cached FFC's do not need to be compiled again for the models.
# To do: Material seperation from the problem can be thought in a more clever way to enable the usage of 
# different materials easliy with the same model. But, one should not forget that the variational form of the
# problem has to match!
################################################################################################################

class RVE(model):
    """ 
        General RVE model implementation
    """
    def __init__(self, domain, Ve=None):

        """ Initialize """

        bc = PeriodicBoundary(domain,periodicity=[0,1],tolerance=1e-10)   # Initialize periodic boundary conditions
        model.__init__(self,domain,bc, Ve)              # Initialize base-class

        ################################
        # Mixed function space initialization with periodic boundary conditions
        ################################
        self.W = FunctionSpace(self.domain.mesh, MixedElement([self.Ve, self.Re]), constrained_domain=self.bc)

    def __call__(self,F_macro):
        
        """ Implement call method for function like usage """

        self.F_macro = F_macro
        self.convergence = True

    def problem(self):

        """ Method: Define the variational problem for with pure neumann boundary condition """

        v_,lamb_ = TestFunctions(self.W)                # Define test functions 
        dv, dlamb = TrialFunctions(self.W)              # Define trial functions 
        self.w = Function(self.W)
        
        ################################
        # F_macro should be defined locally because when passing in another way
        # it gives erroneous results! So for consistancy it is defined just before
        # passing as Constant from fenics.
        ################################
        d = len(u)
        I = Identity(d)             # Identity tensor
        F = I + grad(u) + Constant(self.F_macro)              # Deformation gradient
        F = variable(F)
        C = F.T*F                    # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        Ic = tr(C)
        J  = det(F)
        mu = [Constant(180.5),Constant(1.9e3)]
        lmbda = [Constant(679.5),Constant(2.73e3)]
        print(mu[0],lmbda[0])
        print(mu[1],lmbda[1])

        K = [lmbda[i] + 2./3. * mu[i] for i in range(2)]
        C1 = [mu[i]/2. for i in range(2)]
        D1 = [K[i] /2. for i in range(2)]
        # Stored strain energy density (compressible neo-Hookean model)
        a = [0.5 , 1./20., 11./1050., 19./7000., 519./673750.]
        lmbda_m=2.8
        beta = 1./lmbda_m**2
        psi_C = mu[0]/2* sum([a[i-1]*beta**(i-1)*((tr(C)**(i)-3**i)) for i in range(1,6)])
        psi_J = K[0]/2. * ((J**2-1)/2.-ln(J))
        psi2 = psi_J+psi_C

        #psi = [K[i]/2*(J-1)**2 + mu[i]/2*(tr(dot(F.T,F))*J**(-2/3)-3) for i in range(2)]
        psi = [(mu[i] /2)*(Ic - 3) - mu[i] *ln(J) + (lmbda[i] /2)*(ln(J))**2 for i in range(2)]


        #P =  mu*J**(-2./3.)*(F-1./3.*Ic*inv(F.T)) + K/2.*J*(J-1.+1./J*ln(J))*inv(F.T)
        #sig = [1./J * (mu[i] *(F*F.T-I)+lmbda[i] *ln(J)*I) for i in range(2)]
        #P = [J*sig[i]*inv(F.T) for i in range(2)]
        P = [diff(psi[0],F), diff(psi2,F) ]
        #

        self.PI = sum([inner(P[i],grad(v_))*dx(i+1) for i in range(2)])  
        #F = sum([inner(sigma(dv, i, F_macro), eps(v_))*dx(i) for i in range(nphases)])
        #a, L = lhs(F), rhs(F)
        self.PI += dot(lamb_,u)*dx + dot(c,v_)*dx




    def solver(self):

        """ Method: Define solver options with your solver """

        self.problem()
        prm = {"newton_solver":
                {"absolute_tolerance":1e-7,'relative_tolerance':1e-7,'relaxation_parameter':1.0}}
        try:
            solve(self.PI==0, self.w, [],solver_parameters=prm,form_compiler_parameters={"optimize": True})
            (self.v, lamb) = self.w.split(True)
        except:
            self.convergence = False
        self.__deformed()


    def postprocess(self):
        """ 
            Method: postprocess implementation to get the homogenized 
                    second Piola-Kirchhoff stress tensor 
        """
        P = self.__project_P()                          # Project first Piola-Kirchhoff stress tensor 
        F = self.__project_F()                          # Project Deformation Gradient

        if self.domain.dim == 3:
            ################################
            # Split for later merging the tensors
            ################################
            p11,p12,p13,p21,p22,p23,p31,p32,p33 = P.split(True) 
            f11,f12,f13,f21,f22,f23,f31,f32,f33 = F.split(True)
            P_hom = np.array([[p11.vector().get_local().mean(),p12.vector().get_local().mean(),p13.vector().get_local().mean()],
                              [p21.vector().get_local().mean(),p22.vector().get_local().mean(),p23.vector().get_local().mean()],
                              [p31.vector().get_local().mean(),p32.vector().get_local().mean(),p33.vector().get_local().mean()]])/self.domain.vol
            
            F_hom = np.array([[f11.vector().get_local().mean(),f12.vector().get_local().mean(),f13.vector().get_local().mean()],
                              [f21.vector().get_local().mean(),f22.vector().get_local().mean(),f23.vector().get_local().mean()],
                              [f31.vector().get_local().mean(),f32.vector().get_local().mean(),f33.vector().get_local().mean()]])/self.domain.vol
            
            ################################
            # If you don't have convergence just put zero
            ################################
            if self.convergence is False:
                S = np.zeros((3,3))
            else:
                S = np.linalg.inv(F_hom).dot(P_hom)

        elif self.domain.dim == 2: 
            p11,p12,p21,p22 = P.split(True)
            f11,f12,f21,f22 = F.split(True)
            P_hom= np.array([[p11.vector().get_local().mean(),p12.vector().get_local().mean()],
                                     [p21.vector().get_local().mean(),p22.vector().get_local().mean()]])/self.domain.vol
            
            F_hom = np.array([[f11.vector().get_local().mean(),f12.vector().get_local().mean()],
                                     [f21.vector().get_local().mean(),f22.vector().get_local().mean()]])#/self.domain.vol
            if self.convergence is False:
                S = np.zeros((self.domain.dim,self.domain.dim))
            else:
                S = np.linalg.inv(F_hom).dot(P_hom)
                #S = P_hom
                #S = 0.5*(F_hom.T.dot(F_hom)-np.eye(2))
                #S = 1/np.linalg.det(F_hom)*F_hom.dot(S.dot(F_hom.T))
                #S = (F_hom).dot(S)

        return S

    def __deformed(self):

        """ Method: output the deformed state to a file """

        V = FunctionSpace(self.domain.mesh,self.Ve)
        y = SpatialCoordinate(self.domain.mesh)
        write = dot(Constant(self.F_macro),y)+self.v
        filename = File("deformation.pvd")
        filename << project(write,V)

        ################################
        # Easy ploting for the 2D deformation
        ################################
        y = SpatialCoordinate(self.domain.mesh)
        #F = Identity(self.domain.dim) + grad(self.v) + Constant(self.F_macro)              # Deformation gradient
        p = plot(dot(Constant(self.F_macro),y)+self.v, mode="displacement")
        #p = plot(self.v, mode="displacement")
        #p = plot(self.stress[0, 0])
        plt.colorbar(p)
        plt.savefig("rve_deformed.pdf")

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
#######################################################################################################################
#######################################################################################################################
# PURGATORY
#######################################################################################################################
#######################################################################################################################
class RVE2D(model):
    """
        RVE for 2D models
    """
    def __init__(self, domain, Ve=None):

        """ Initialize """

        bc = PeriodicBoundary(domain,tolerance=1e-10)
        model.__init__(self,domain,bc, Ve) 
        self.W = FunctionSpace(self.domain.mesh, MixedElement([self.Ve, self.Re]), constrained_domain=self.bc)

    def __call__(self,F_macro):

        """ Implement call method for function like usage """

        self.F_macro = F_macro
        self.convergence = True

    def problem(self):
        v_,lamb_ = TestFunctions(self.W)
        dv, dlamb = TrialFunctions(self.W)
        self.w = Function(self.W)
        
        F_macro = Constant(self.F_macro)
        u,c = split(self.w)
        self.material = [NeoHookean(u,F_macro,mu=4e9,lmbda=1.9e9),ArrudaBoyce(u,F_macro, mu=800e6, lmbda=180.5, lmbda_m=2.8)]
        dx = Measure('dx')(subdomain_data=self.domain.subdomains)
        
        self.PI = inner(self.material[0].P,grad(v_))*dx(1) + inner(self.material[1].P,grad(v_))*dx(2) 
        
        self.PI += dot(lamb_,u)*dx + dot(c,v_)*dx


    def solver(self):
        self.problem()
        try:
            solve(self.PI==0, self.w, [],solver_parameters={"newton_solver":{"absolute_tolerance":1e-7}})
            (self.v, lamb) = self.w.split(True)
        except:
            self.convergence = False
        self.__deformed()


    def postprocess(self):
        P = self.__project_P()
        F = self.__project_F()
        p11,p12,p21,p22 = P.split(True)
        f11,f12,f21,f22 = F.split(True)
        P_hom= np.array([[p11.vector().get_local().sum(),p12.vector().get_local().sum()],
                                 [p21.vector().get_local().sum(),p22.vector().get_local().sum()]])/self.domain.vol
        
        F_hom = np.array([[f11.vector().get_local().sum(),f12.vector().get_local().sum()],
                                 [f21.vector().get_local().sum(),f22.vector().get_local().sum()]])/self.domain.vol
        if self.convergence is False:
            S = np.zeros((self.domain.dim,self.domain.dim))
        else:
            S = np.linalg.inv(F_hom).dot(P_hom)
        return S

    def __deformed(self):

        V = FunctionSpace(self.domain.mesh,self.Ve)
        y = SpatialCoordinate(self.domain.mesh)
        write = dot(Constant(self.F_macro),y)+self.v
        filename = File("deformation.pvd")
        filename << project(write,V)

    def __project_P(self):
        V = TensorFunctionSpace(self.domain.mesh, "DG",0)
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
        V = TensorFunctionSpace(self.domain.mesh, "DG",0)
        dx = Measure('dx')(subdomain_data=self.domain.subdomains)
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv,v_)*dx
        b_proj = inner(self.material[0].F,v_)*dx(1) +inner(self.material[1].F,v_)*dx(2)
        F = Function(V)
        solve(a_proj==b_proj,F)
        return F

         
        



                
                    
