from ..src.sim import FEM 
import numpy as np

class simulate_fenics_rve(FEM):
    """

        SIMULATION-Module wrap for FEniCS

    """
    def __init__(self, in_data, model,**kwargs):

        """ Initialize """

        self.__name__ = 'FEniCS-RVE'                    # Name module
        super().__init__(in_data=in_data, model=model,**kwargs)                    # Initialize base-class

        self.dim = model.domain.dim

    def post_init(self):

        """ Implement post_init rule !!!"""

        if self.dim ==2:
            self.out_var = ['E11','E12','E22','S11','S12','S22']    # Name output
        elif self.dim == 3:
            self.out_var = ['E11','E12','E13','E22','E23','E33','S11','S12','S13','S22','S23','S33']    # Name output
        else:
            NotImplementedError

        self.out = np.zeros((self.in_data.num,len(self.out_var)))   # Initialize numpy output

    def to_do(self,i):
        
        """ Implement to_do method !!!"""

        ################################
        # Create Macroscopic Deformation Gradient 
        ################################
        F11 = self.in_data.DATA.DataFrame.iloc[i]['F11']       
        F12 = self.in_data.DATA.DataFrame.iloc[i]['F12']
        F22 = self.in_data.DATA.DataFrame.iloc[i]['F22']

        F_macro = np.array([[F11,F12],[F12,F22]])

        if self.dim == 3:
            F13 = self.in_data.DATA.DataFrame.iloc[i]['F13']       
            F23 = self.in_data.DATA.DataFrame.iloc[i]['F23']
            F33 = self.in_data.DATA.DataFrame.iloc[i]['F33']

            F_macro = np.array([[F11,F12,F13],[F12,F22,F23],[F13,F23,F33]])
            
        E = 0.5*(np.dot((F_macro+np.eye(self.dim)).T,(F_macro+np.eye(self.dim)))-np.eye(self.dim))

        self.model(F_macro,i)                             # Call your model with F_macro
        self.model.solver()                             # Solve your model
        S,_ = self.model.postprocess()                  # Post-process your results
        if self.dim == 3:
            self.out[i,:] = [E[0,0],E[0,1],E[0,2],E[1,1],E[1,2],E[2,2],S[0,0],S[0,1],S[0,2],S[1,1],S[1,2],S[2,2]]
        elif self.dim == 2:
            self.out[i,:] = [E[0,0],E[0,1],E[1,1],S[0,0],S[0,1],S[1,1]]          # Store your output 


class simulate_fenics_rve_old(FEM):
    """

        SIMULATION-Module wrap for FEniCS

    """
    def __init__(self, doe, model):

        """ Initialize """

        self.__name__ = 'FEniCS-RVE'                    # Name module
        super().__init__(doe, model)                    # Initialize base-class

    def post_init(self):

        """ Implement post_init rule !!!"""

        self.out_var = ['E11','E12','E22','S11','S12','S22']    # Name output
        self.out = np.zeros((self.doe.num,len(self.out_var)))   # Initialize numpy output

    def to_do(self,i):
        
        """ Implement to_do method !!!"""

        ################################
        # Create Macroscopic Deformation Gradient 
        ################################
        F11 = self.doe.pandas_data.iloc[i]['F11']       
        F12 = self.doe.pandas_data.iloc[i]['F12']
        F22 = self.doe.pandas_data.iloc[i]['F22']
        F_macro = np.array([[F11,F12],[F12,F22]])
        E = 0.5*(np.dot((F_macro+np.eye(2)).T,(F_macro+np.eye(2)))-np.eye(2))

        self.model(F_macro)                             # Call your model with F_macro
        self.model.solver()                             # Solve your model
        S,_ = self.model.postprocess()                  # Post-process your results
        self.out[i,:] = [E[0,0],E[0,1],E[1,1],S[0,0],S[0,1],S[1,1]]          # Store your output 


