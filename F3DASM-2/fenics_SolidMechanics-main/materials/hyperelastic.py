from ..src.continuum import * 


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

class SVenantKirchhoff(Material):
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

        psi = self.lmbda/2*(tr(self.E))**2 + self.mu*tr(self.E*self.E.T) 

        return psi

class ArrudaBoyce(Material):
    """
        Arruda-Boyce Material Model

    """
    def __init__(self, u, F_macro, E=None, nu=None, mu=None, lmbda=None, lmbda_m=2.8):

        """ Initialize """

        ################################
        # Initialize material properties
        ################################
        if E is not None and nu is not None:
            self.mu, self.lmbda = Lame(E,nu)
        else:
            self.mu, self.lmbda = mu, lmbda
 
        self.K = self.lmbda + 2./3. * self.mu
        #print(self.K)
        self.C1 = self.mu/2. 
        self.D1 = self.K /2.
        self.lmbda_m = lmbda_m
        self.a = [0.5 , 1./20., 11./1050., 19./7000., 519./673750.]
        self.a1 = [1 , 3./5., 99./175., 513./875.,42039./67375.]
        self.mu_m = self.mu / sum([1,3./(5*self.lmbda_m**2),99./(175*self.lmbda_m**4),513./(875*self.lmbda_m**6),42039./(67375*self.lmbda_m**8)])

        self.beta = 1./self.lmbda_m**2
        Material.__init__(self,u=u, F_macro=F_macro)
        
    def Energy(self):

        """ Method: Implement energy density function """

        #self.mu = self.mu* 1/2*(sum([i*self.a1[i-1]*self.beta**(i-1) for i in range(1,6)]))

        #psi_C = self.mu/2* sum([self.a[i-1]*self.beta**(i-1)*(((tr(self.C))**(i)-3**i)) for i in range(1,6)])
        psi_J = self.K/2. * ((self.J**2-1)/2.-ln(self.J))
        #psi_J = self.K/2. * (ln(self.J**(1/2)))**2
        lm_ = self.lmbda_m
        psi_C = self.mu*(1/2*((tr(self.C)*self.J**(-2/3))-3)+1./(20.*lm_**2)*((tr(self.C)*self.J**(-2/3))**2-3**2)+\
               11/(1050*lm_**2)*((tr(self.C)*self.J**(-2/3))**3-3**3)+19/(7000*lm_**2)*((tr(self.C)*self.J**(-2/3))**4-3**4)+\
               519/(673750*lm_**2)*((tr(self.C)*self.J**(-2/3))**5-3**5))
 
        #psi_C = self.mu*(1/2*(tr(self.C)-3) +1./(20.*lm_**2)*(tr(self.C)**2-3**2)+\
        #        11/(1050*lm_**2)*(tr(self.C)**3-3**3)+19/(7000*lm_**2)*(tr(self.C)**4-3**4)+\
        #        519/(673750*lm_**2)*(tr(self.C)**5-3**5))
        return  psi_C+psi_J

class MooneyRivlin(Material):
    """
        Mooney-Rivlin Material Model

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
 
        self.K = self.lmbda + 2./3. * self.mu
        self.C1 = self.mu/2. 
        self.D1 = self.K /2.

        self.c01 = -1.5
        self.c10 = 3.4
        Material.__init__(self,u=u, F_macro=F_macro)

        
    def Energy(self):

        """ Method: Implement energy density function """
        psi = self.c10*(self.I1_C-3) + self.c01*(self.I2_C-3) + self.K/2*(self.J-1)**2 - 2*(self.c10+self.c01)*ln(self.J)

        return psi


class Gent(Material):
    """
        Gent Material Model

    """
    def __init__(self, u, F_macro, E=None, nu=None, mu=None, lmbda=None, Jm=80):

        """ Initialize """

        ################################
        # Initialize material properties
        ################################
        if E is not None and nu is not None:
            self.mu, self.lmbda = Lame(E,nu)
        else:
            self.mu, self.lmbda = mu, lmbda
 
        self.K = self.lmbda + 2./3. * self.mu
        self.C1 = self.mu/2. 
        self.D1 = self.K /2.
        self.Jm = Jm

        Material.__init__(self,u=u, F_macro=F_macro)
        
    def Energy(self):

        """ Method: Implement energy density function """
        psi_C = -self.mu/2*self.Jm*ln(1-(tr(self.C)-3-2*ln(self.J))/self.Jm)  
        psi_J = self.K/2. * ((self.J**2-1)/2.-ln(self.J))

        return psi_C + psi_J



