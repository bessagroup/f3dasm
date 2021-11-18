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

class MultiMaterialRVE(model):
    """ 
        General RVE model implementation
    """
    def __init__(self, domain, Ve=None,model_tag=1):

        """ Initialize """

        bc = PeriodicBoundary(domain,periodicity=list(range(domain.dim)),tolerance=1e-10)   # Initialize periodic boundary conditions
        model.__init__(self,domain,bc, Ve)              # Initialize base-class

        self.model_tag = model_tag

        ################################
        # Mixed function space initialization with periodic boundary conditions
        ################################
        self.W = FunctionSpace(self.domain.mesh, MixedElement([self.Ve, self.Re]), constrained_domain=self.bc)

        self.fileResults = XDMFFile("output.xdmf")
        self.fileResults.parameters["flush_output"] = True
        self.fileResults.parameters["functions_share_mesh"] = True

    def __call__(self,F_macro,time=0):
        
        """ Implement call method for function like usage """

        self.F_macro = F_macro
        self.convergence = True
        self.time = time

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
        F_macro = Constant(self.F_macro)                

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

        self.material = models[self.model_tag]

        dx = Measure('dx')(subdomain_data=self.domain.subdomains)       # Redefine dx for subdomains

        ################################
        # Variational problem definition -> Lagrangian Linear Momentum Equation
        ################################
        self.PI = inner(self.material[0].P,nabla_grad(v_))*dx(1) + inner(self.material[1].P,nabla_grad(v_))*dx(2)  
        
        self.PI += dot(lamb_,u)*dx + dot(c,v_)*dx


    def solver(self):

        """ Method: Define solver options with your solver """

        self.problem()
        prm = {"newton_solver":
                {"absolute_tolerance":1e-7,'relative_tolerance':1e-7,'relaxation_parameter':1.0,'linear_solver' : 'mumps'}}
        try:
            solve(self.PI==0, self.w, [],solver_parameters=prm,form_compiler_parameters={"optimize": True},)
            (self.v, lamb) = self.w.split(True)
        except:
            self.convergence = False
        
        self.__project_u()


    def postprocess(self):
        """ 
            Method: postprocess implementation to get the homogenized 
                    second Piola-Kirchhoff stress tensor 
        """
        if self.convergence is False:
            S = np.zeros((self.domain.dim,self.domain.dim))
        else:
            
            P = self.__project_P()                          # Project first Piola-Kirchhoff stress tensor 
            F = self.__project_F()                          # Project Deformation Gradient

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
        P = Function(V,name='Piola')
        solve(a_proj==b_proj,P)
        self.fileResults.write(P,self.time)
        return P

    def __project_u(self):

        """ 
            Method: Projecting displacement.
                    Another linear variational problem has to be solved.
        """

        V = FunctionSpace(self.domain.mesh, self.Ve)           # Define Discontinuous Galerkin space

        ################################
        # Similar type of problem definition inside the model
        ################################

        y = SpatialCoordinate(self.domain.mesh)
        write = dot(Constant(self.F_macro),y)+self.v
        dx = Measure('dx')(subdomain_data=self.domain.subdomains)   
        dx = dx(metadata={'quadrature_degree': 1})
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv,v_)*dx
        b_proj = inner(write,v_)*dx
        u = Function(V,name='Displacement')
        solve(a_proj==b_proj,u,solver_parameters={"linear_solver": "mumps"} )
        self.fileResults.write(u,self.time)
        return u
        

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

    def __call__(self,F_macro,i):
        
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
        F_macro = Constant(self.F_macro)                

        u,c = split(self.w)

        ################################
        # Define materials for phases
        ################################

        #self.material = [SVenantKirchhoff(u,F_macro, mu=180.5, lmbda=679.67),NeoHookean(u,F_macro,mu=1.9e3,lmbda=2.73e3)]
        #self.material = [MooneyRivlin(u,F_macro, mu=180.5, lmbda=679.67),NeoHookean(u,F_macro,mu=1.9e3,lmbda=2.73e3)]
        #self.material = [SVenantKirchhoff(u,F_macro, mu=80.5, lmbda=300.67),NeoHookean(u,F_macro,mu=1.9e2,lmbda=2.73e2)]
        #self.material = [MooneyRivlin(u,F_macro, E=200, nu=0.4),MooneyRivlin(u,F_macro,E=200,nu=0.4)]
        #self.material = [SVenantKirchhoff(u,F_macro, E=200, nu=0.3),MooneyRivlin(u,F_macro,E=200,nu=0.3)]
        #self.material = [SVenantKirchhoff(u,F_macro, E=100, nu=0.1),NeoHookean(u,F_macro,E=250,nu=0.3)]
        #self.material = [MooneyRivlin(u,F_macro, E=100, nu=0.1),NeoHookean(u,F_macro,E=250,nu=0.3)]
        #self.material = [SVenantKirchhoff(u,F_macro, E=100, nu=0.1),NeoHookean(u,F_macro,E=500,nu=0.3)]
        self.material = [NeoHookean(u,F_macro, E=300, nu=0.1),SVenantKirchhoff(u,F_macro,E=500,nu=0.3)]        # model-1
        #self.material = [NeoHookean(u,F_macro, E=300, nu=0.1),NeoHookean(u,F_macro,E=300,nu=0.1)]              # model-2
        #self.material = [NeoHookean(u,F_macro, E=300, nu=0.),NeoHookean(u,F_macro,E=300,nu=0.)]                # model-3
        #self.material = [SVenantKirchhoff(u,F_macro, E=300, nu=0.1),NeoHookean(u,F_macro,E=500,nu=0.3)]        # model-4
        #self.material = [NeoHookean(u,F_macro, E=300, nu=0.1),SVenantKirchhoff(u,F_macro,E=300,nu=0.3)]        # model-5
        #self.material = [NeoHookean(u,F_macro, E=300, nu=0.0),SVenantKirchhoff(u,F_macro,E=500,nu=0.0)]
        #self.material = [NeoHookean(u,F_macro, E=300, nu=0.1),NeoHookean(u,F_macro,E=300,nu=0.4)]
        #self.material = [SVenantKirchhoff(u,F_macro, E=100, nu=0.0),SVenantKirchhoff(u,F_macro,E=500,nu=0.0)]
        #self.material = [MooneyRivlin(u,F_macro, E=50, nu=0.0),NeoHookean(u,F_macro,E=50,nu=0.0)]
        #self.material = [MooneyRivlin(u,F_macro, E=50, nu=0.4),MooneyRivlin(u,F_macro,E=50,nu=0.4)]
        #self.material = [SVenantKirchhoff(u,F_macro, E=50, nu=0.1),SVenantKirchhoff(u,F_macro,E=50,nu=0.1)]
        #self.material = [SVenantKirchhoff(u,F_macro, E=300, nu=0.1),NeoHookean(u,F_macro,E=500,nu=0.3)]
        #self.material = [NeoHookean(u,F_macro, E=100, nu=0.1),SVenantKirchhoff(u,F_macro,E=100,nu=0.1)]
        #self.material = [NeoHookean(u,F_macro, E=400, nu=0.4),NeoHookean(u,F_macro, E=400, nu=0.4)]

        #self.material = [MooneyRivlin(u,F_macro, E=20, nu=0.2),SVenantKirchhoff(u,F_macro,E=20,nu=0.3)]
        #self.material = [NeoHookean(u,F_macro, E=200, nu=0.3),NeoHookean(u,F_macro,E=200,nu=0.3)]
        #self.material = [SVenantKirchhoff(u,F_macro, mu=180.5, lmbda=679.67),NeoHookean(u,F_macro,mu=1.9e2,lmbda=2.73e2)]
        #self.material = [NeoHookean(u,F_macro, mu=180.5, lmbda=679.67),SVenantKirchhoff(u,F_macro,mu=1.9e3,lmbda=2.73e3)]
        #self.material = [NeoHookean(u,F_macro, mu=180.5, lmbda=679.67),NeoHookean(u,F_macro,mu=1.9e3,lmbda=2.73e3)]
        #self.material = [NeoHookean(u,F_macro, mu=180.5, lmbda=679.67),NeoHookean(u,F_macro,mu=180.5, lmbda=679.67)]
        #self.material = [NeoHookean(u,F_macro, mu=1.5, lmbda=1.9e3),NeoHookean(u,F_macro,mu=1.5, lmbda=1.9e3)]
        #self.material = [NeoHookean(u,F_macro, mu=1.5, lmbda=2199),NeoHookean(u,F_macro,mu=1.5, lmbda=2199)]
        #self.material = [ArrudaBoyce(u,F_macro, mu=180.5, lmbda=679.67),NeoHookean(u,F_macro,mu=1.9e3,lmbda=2.73e3)]
        #self.material = [ArrudaBoyce(u,F_macro, mu=180.5, lmbda=679.67),NeoHookean(u,F_macro,mu=1.9e3,lmbda=2.73e3)]
        #self.material = [Gent(u,F_macro, mu=180.5, lmbda=679.67),NeoHookean(u,F_macro,mu=1.9e3,lmbda=2.73e3)]
        #self.material = [Gent(u,F_macro, mu=180.5, lmbda=679.67),Gent(u,F_macro,mu=180.5,lmbda=679.67)]
        #self.material = [MooneyRivlin(u,F_macro, mu=180.5, lmbda=679.67),MooneyRivlin(u,F_macro,mu=180.5,lmbda=679.67)]
        #self.material = [MooneyRivlin(u,F_macro, mu=180.5, lmbda=679.67),NeoHookean(u,F_macro,mu=180.5,lmbda=679.67)]
        #self.material = [MooneyRivlin(u,F_macro, mu=180.5, lmbda=679.67),NeoHookean(u,F_macro,mu=1.9e3,lmbda=2.73e3)]
        #self.material = [NeoHookean(u,F_macro, mu=180.5, lmbda=679.67),MooneyRivlin(u,F_macro,mu=1.9e3,lmbda=2.73e3)]
        #self.material = [ArrudaBoyce(u,F_macro, mu=180.5, lmbda=679.67),ArrudaBoyce(u,F_macro, mu=180.5, lmbda=679.67)]
        #self.material = [NeoHookean(u,F_macro, mu=1.9e3, lmbda=2.73e3),NeoHookean(u,F_macro,mu=1.9e3, lmbda=2.73e3)]
        #self.material = [SVenantKirchhoff(u,F_macro, mu=180.5, lmbda=679.67),SVenantKirchhoff(u,F_macro,mu=1.9e3,lmbda=2.73e3)]
        #self.material = [SVenantKirchhoff(u,F_macro, mu=180.5, lmbda=679.67),SVenantKirchhoff(u,F_macro,mu=180.5,lmbda=679.5)]
        
        dx = Measure('dx')(subdomain_data=self.domain.subdomains)       # Redefine dx for subdomains
        #print(self.domain.subdomains.array())
        #dx = dx(metadata={'quadrature_degree': 1})                      # Specifiy quadrature rules
        
        #self.PI = sum([inner(material[i].P,grad(v_))*dx(phase) for i, phase in enumerate(self.domain.phases)])

        ################################
        # Variational problem definition -> Lagrangian Linear Momentum Equation
        ################################
        self.PI = inner(self.material[0].P,nabla_grad(v_))*dx(1) + inner(self.material[1].P,nabla_grad(v_))*dx(2)  
        #self.PI = inner(self.material[0].P,grad(v_))*dx 
        
        #self.PI = inner(material[1].P,grad(v_))*dx 
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
            filename = File('stress.pvd')
            filename << P
            F = self.__project_F()                          # Project Deformation Gradient

            Piola = P.split(True)
            DG = F.split(True)
            P_hom = np.zeros(self.domain.dim**2) 
            F_hom = np.zeros(self.domain.dim**2) 
            
            for i in range(self.domain.dim**2):
                for j in range(self.domain.ele_num):
                    #P_hom[i] = (Piola[i].vector().get_local().mean())/self.domain.vol
                    P_hom[i] = np.dot(Piola[i].vector().get_local(),self.domain.ele_vol)/self.domain.vol
                    #P_hom[i] = np.dot(Piola[i].vector().get_local(),self.domain.ele_vol)/self.domain.ele_num/self.domain.vol
                    #P_hom[i] = Piola[i].vector().get_local().mean()
                    F_hom[i] = (DG[i].vector().get_local().mean())

            P_hom = P_hom.reshape(-1,self.domain.dim)
            F_hom = F_hom.reshape(-1,self.domain.dim)
            #S = np.dot(P_hom,np.linalg.inv(F_hom.T))
            S = np.dot(np.linalg.inv(F_hom.T),P_hom)
            #S = np.linalg.inv(self.F_macro+np.eye(self.domain.dim)).dot(P_hom)
            #S = 0.5*(F_hom.T.dot(F_hom)-np.eye(2))
            #S = 1/np.linalg.det(self.F_macro+np.eye(self.domain.dim))*F_hom.dot(S.dot(F_hom.T))
            #S = (F_hom).dot(S)
            #S = 1/np.linalg.det(F_hom)*np.dot(P_hom,F_hom.T)
            #S = P_hom


        
        return S, F_hom

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

         
        
class RVE_other(model):
    """ 
        General RVE model implementation
    """
    def __init__(self, domain, Ve=None):

        """ Initialize """

        bc = PeriodicBoundary(domain,periodicity=[0,1],tolerance=1e-10)   # Initialize periodic boundary conditions
        model.__init__(self,domain,bc, Ve)              # Initialize base-class


    def __call__(self,F_macro):
        
        """ Implement call method for function like usage """

        self.F_macro = F_macro
        self.convergence = True

    def problem(self):

        Ve = VectorElement("CG", self.domain.mesh.ufl_cell(), 1)
        Re = VectorElement("R", self.domain.mesh.ufl_cell(), 0)
        W = FunctionSpace(self.domain.mesh, MixedElement([self.Ve, self.Re]), constrained_domain=PeriodicBoundary(self.domain,tolerance=1e-10))
        V = FunctionSpace(self.domain.mesh, Ve)

        v_,lamb_ = TestFunctions(W)
        dv, dlamb = TrialFunctions(W)
        w = Function(W)
        u,c = split(w)
        dx = Measure('dx')(subdomain_data=self.domain.subdomains)

        mu = [180.5,1.9e3];lmbda=[679.67,2.73e3]
        K = [800,4e3]

        F_macro = Constant(((1., 0.15), (0.15,-0.2)))
        d = len(u)
        I = Identity(d)             # Identity tensor
        F = I + grad(u) + F_macro              # Deformation gradient
        F = variable(F)
        C = F.T*F                    # Right Cauchy-Green tensor
        B = F*F.T                    # Right Cauchy-Green tensor
        J = det(F)
        E = 0.5 * (C-I)

        psi_K = [lmbda[i]/2*(tr(E))**2 + mu[i]*tr(E*E.T) for i in range(2)]
        psi_N = [mu[i]/2*(tr(C)-3)  for i in range(2)]
        psi_N = [mu[i]/2*(tr(B)-3-2*ln(J))+lmbda[i]/2*(J-1)**2  for i in range(2)]
        lm_ = 2.8
        psi_A = [mu[i]*(1/2*(tr(C)-3)+1/(20*lm_**2)*(tr(C)**2-3**2)+11/(1050*lm_**2)*(tr(C)**3-3**3)+19/(7000*lm_**2)*(tr(C)**4-3**4)+519/(673750*lm_**2)*(tr(C)**5-3**5))+K[i]/2*((J**2-1)/2-ln(J)) for i in range(2)]

        P = [ diff(psi_A[0],F),diff(psi_N[1],F) ]

        S = [ inv(F)*P[0], inv(F)*P[1]]

        PI = sum([inner(P[i],grad(v_))*dx(i+1) for i in range(2)])  
        PI += dot(lamb_,u)*dx + dot(c,v_)*dx


        solve(PI==0, w, [],solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})
        (v, lamb) = w.split(True)

        # plotting deformed unit cell with total displacement u = F_macro*y + v
        y = SpatialCoordinate(self.domain.mesh)
        plt.figure()
        #F = I + grad(v) + F_macro              # Deformation gradient
        p = plot(dot(F_macro,y)+v, mode="displacement")
        #p = plot(+v, mode="displacement")


        plt.colorbar(p)
        plt.savefig("2.pdf")
        filename = File("ali.pvd")
        y = SpatialCoordinate(self.domain.mesh)
        total = 0.5*(dot(F_macro, y)+v)
        filename << project(total,V)


        V = TensorFunctionSpace(self.domain.mesh, "DG",0)
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv,v_)*dx
        b_proj = inner(P[0],v_)*dx(1) +inner(P[1],v_)*dx(2)
        P = Function(V)
        solve(a_proj==b_proj,P)
        F = project(F, TensorFunctionSpace(self.domain.mesh, 'DG',0))
        s11, s12,  s21, s22 = P.split(True)
        f11, f12,  f21, f22 = F.split(True)
        P = np.array([[s11.vector().get_local().mean(),s12.vector().get_local().mean()],[s21.vector().get_local().mean(),s22.vector().get_local().mean()]])/self.domain.vol
        F = np.array([[f11.vector().get_local().mean(),f12.vector().get_local().mean()],[f21.vector().get_local().mean(),f22.vector().get_local().mean()]])
        PK2 =np.linalg.inv(F).dot(P)
        print('PK2:\n',PK2)
        V = TensorFunctionSpace(self.domain.mesh, "DG",0)
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv,v_)*dx
        b_proj = inner(S[0],v_)*dx(1) +inner(S[1],v_)*dx(2)
        S = Function(V)
        solve(a_proj==b_proj,S)

        s11, s12,  s21, s22 = S.split(True)
        S = np.array([[s11.vector().get_local().mean(),s12.vector().get_local().mean()],[s21.vector().get_local().mean(),s22.vector().get_local().mean()]])/self.domain.vol
        print('PK2:\n',S)
        V = FunctionSpace(self.domain.mesh, "DG",0)
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv,v_)*dx
        b_proj = inner(psi_A[0],v_)*dx(1) +inner(psi_N[1],v_)*dx(2)
        psi = Function(V)
        solve(a_proj==b_proj,psi)

        print('psi:\n',psi.vector().get_local().mean())
                        
 #self.material = [ArrudaBoyce(u,F_macro, mu=180.5, lmbda=679.67),NeoHookean(u,F_macro,mu=1.9e3,lmbda=2.73e3)]                                                    
 #self.material = [ArrudaBoyce(u,F_macro, mu=180.5, lmbda=679.5),NeoHookean(u,F_macro,mu=1.9e3,lmbda=2.73e3)]
 #self.material = [ArrudaBoyce(u,F_macro, mu=180.5, lmbda=679.5, lmbda_m=2.8),NeoHookean(u,F_macro,mu=190,lmbda=273)]
 #self.material = [NeoHookean(u,F_macro, mu=180.5, lmbda=679.5),NeoHookean(u,F_macro,mu=1.90e3,lmbda=2.73e3)]
 #self.material = [NeoHookean(u,F_macro, mu=180.5, lmbda=679.5),NeoHookean(u,F_macro,mu=200,lmbda=300)]
 #self.material = [NeoHookean(u,F_macro,mu=1.9e3,lmbda=2.73e3),ArrudaBoyce(u,F_macro, lmbda=180.5, mu=679.5, lmbda_m=2.8)]
 #self.material = [NeoHookean(u,F_macro,mu=1.9e3,lmbda=2.73e3),ArrudaBoyce(u,F_macro, lmbda=180.5, mu=679.5, lmbda_m=2.8)]
 #self.material = [NeoHookean(u,F_macro,mu=3.5,lmbda=2.199e3),NeoHookean(u,F_macro,mu=1.5,lmbda=2.199e3)]
 #self.material = [ArrudaBoyce(u,F_macro, mu=1., lmbda=2.180e3, lmbda_m=2.8),ArrudaBoyce(u,F_macro, mu=1., lmbda=2.180e3, lmbda_m=2.8)]
 #self.material = [ArrudaBoyce(u,F_macro, mu=1., lmbda=2.180e3, lmbda_m=2.8),NeoHookean(u,F_macro,mu=1.5,lmbda=2.199e3)]
 #self.material = [NeoHookean(u,F_macro, E=2,nu=0.),NeoHookean(u,F_macro,E=2,nu=0.)]
 #self.material = [ArrudaBoyce(u,F_macro, E=2,nu=0.),ArrudaBoyce(u,F_macro,E=2,nu=0.)]
 #self.material = [ArrudaBoyce(u,F_macro, mu=180., lmbda=679.5, lmbda_m=2.8),ArrudaBoyce(u,F_macro, mu=180., lmbda=679.5, lmbda_m=2.8)]
 #self.material = [NeoHookean(u,F_macro,mu=180.5,lmbda=679.5),ArrudaBoyce(u,F_macro,mu=180.5,lmbda=679.5)]
 #self.material = [ArrudaBoyce(u,F_macro,mu=10.5,lmbda=69.5),NeoHookean(u,F_macro,mu=180.5,lmbda=679.5)]
