from dolfin import tr, det

def Invariants(A):
    """
    Invarients of a Tensor
    """

    I1 = tr(A)
    I2 = 0.5 * (tr(A)**2 - tr(A*A))
    I3 = det(A)
    return I1, I2, I3

def Lame(E,nu):
    mu = E / (2*(1+nu))
    lmbda = E*nu / ((1+nu)*(1-2*nu))
    return mu, lmbda

def decompose2D(tensor):
    """
    [[t1,t2]
     [t3,t3]]
    """
    t1,t2,t3,t4 = tensor.split(True)
    return t1,t2,t3,t4
