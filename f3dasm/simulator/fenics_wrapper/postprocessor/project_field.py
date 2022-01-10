from dolfin import *
import os

def project_P(solver):

    """ 
        Method: Projecting first Piola-Kirchhoff stress tensor.
                Another linear variational problem has to be solved.
    """

    V = TensorFunctionSpace(solver.domain.mesh, "DG",0)           # Define Discontinuous Galerkin space

    ################################
    # Similar type of problem definition inside the model
    ################################
    dx = Measure('dx')(subdomain_data=solver.domain.subdomains)   
    dx = dx(metadata={'quadrature_degree': 1})
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv,v_)*dx
    b_proj = inner(solver.material[0].P,v_)*dx(1) +inner(solver.material[1].P,v_)*dx(2)
    P = Function(V,name='Piola')
    solve(a_proj==b_proj,P)
    solver.fileResults.write(P,solver.time)
    return P


def project_u(solver):

    """ 
        Method: Projecting displacement.
                Another linear variational problem has to be solved.
    """

    V = FunctionSpace(solver.domain.mesh, solver.Ve)           # Define Discontinuous Galerkin space

    ################################
    # Similar type of problem definition inside the model
    ################################

    y = SpatialCoordinate(solver.domain.mesh)
    write = dot(Constant(solver.F_macro),y)+solver.v
    dx = Measure('dx')(subdomain_data=solver.domain.subdomains)   
    dx = dx(metadata={'quadrature_degree': 1})
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv,v_)*dx
    b_proj = inner(write,v_)*dx
    u = Function(V,name='Displacement')
    solve(a_proj==b_proj,u,solver_parameters={"linear_solver": "mumps"} )
    solver.fileResults.write(u,solver.time)
    return u
    

def project_F(solver):

    """ 
        Method: Projecting deformation gradient.
                Another linear variational problem has to be solved.
    """

    ################################
    # Similar type of problem definition inside the model
    ################################
    V = TensorFunctionSpace(solver.domain.mesh, "DG",0)       # Define Discontinuous Galerkin space

    dx = Measure('dx')(subdomain_data=solver.domain.subdomains)
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv,v_)*dx
    b_proj = inner(solver.material[0].F,v_)*dx(1) +inner(solver.material[1].F,v_)*dx(2)
    F = Function(V)
    solve(a_proj==b_proj,F)
    return F


def deformed(solver):

    """ Method: output the deformed state to a file """

    V = FunctionSpace(solver.domain.mesh,solver.Ve)
    y = SpatialCoordinate(solver.domain.mesh)
    write = dot(Constant(solver.F_macro),y)+solver.v
    filename = File(os.path.join(solver.work_dir, "deformation.pvd"))
    filename << project(write,V)

    ################################
    # Easy ploting for the 2D deformation
    ################################
    #y = SpatialCoordinate(solver.domain.mesh)
    ##F = Identity(solver.domain.dim) + grad(solver.v) + Constant(solver.F_macro)              # Deformation gradient
    #p = plot(dot(Constant(solver.F_macro),y)+solver.v, mode="displacement")
    ##p = plot(solver.v, mode="displacement")
    ##p = plot(solver.stress[0, 0])
    #plt.colorbar(p)
    #plt.savefig("rve_deformed.pdf")



