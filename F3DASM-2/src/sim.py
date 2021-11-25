from tqdm import tqdm
import time 
import sys
import pandas as pd
import numpy as np
from .data import *

from .doe import DOE

class FEM():
    """
        Set of rules to be followed for the FEM module of your choice

        * Inherit from this class to your module in order to use its 
            functionalities.
    """
 
    def __init__(self, in_data, model,**kwargs):

        """ Initializing Method """

        #assert issubclass(type(doe), DOE), " Follow the rules and  inherit from the DOE base class! "
        if kwargs is not None:
            self.kwargs = kwargs
        
        self.in_data = in_data                          # Doe initialization
        self.out_data = 0                               # Initialize output
        self.model = model                              # Model initialization
        self.iter = self.in_data.DATA.values.shape[0]   # Number of simulations

    def post_init(self):

        """ Method: Intialize output and name output """

        pass 

    def to_do(self,i):

        """ Method: Define set of things to be done """

        pass
        
    def save(self,filename):
        
        """ Method: Pickle the output """  

        self.out_data.to_pickle(filename)

            
    def run(self):
        
        """ 
            Method: Start the fancy for loop for the jobs defined in 
                    self.to_do() 
        """

        self.post_init()                                # Call for the user-defined self.post_init()

        clock= np.zeros((self.iter,1))                  # Simulation time slots
        if self.kwargs:
            disable = self.kwargs['disable_tqdm']
            print(disable)

        for i in tqdm(range(self.iter),disable=disable):                # Fancy loop
            tic = time.perf_counter()               
            self.to_do(i)                               # Set of things to be done each iteration
            toc = time.perf_counter()
            clock[i] = toc - tic

        self.success = np.count_nonzero(np.mean(self.out,axis=1))               # Check success 
        self.fail_index =  np.where(np.mean(self.out,axis=1) == 0)[0]           # Spit fail-indecies

        self.out_data= DATA(self.out,self.out_var)            # Convert output from numpy to pandas
        self.clock = DATA(clock,['time(s)'])                  # Put simulation times to pandas

        self.out = self.out.astype('float')
        for i in range(self.iter):
            if self.in_data.values[i,:].any() != 0. and self.out[i,:].all() == 0.:
                self.out[i,:] = np.nan                 # Put NaN for failed simulations

        frames = [self.clock.DataFrame, self.in_data.DATA.DataFrame, self.out_data.DataFrame]     # Combine pandas frames 
        self.collective = pd.concat(frames,axis=1)



    def __str__(self):

        """ Overwrite print function """
        
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.set_option('display.expand_frame_repr', False)
        
        if self.module == "FEM":

            print('********************** START ************************')
            print('\n')
            print('                    F3DASM SUMMARY                   ')
            print('\n')
            print('*****************************************************')
            print(self.in_data)
            print('-----------------------------------------------------')
            print('                   SIMULATION INFO                   ')
            print('-----------------------------------------------------')
            print('\n')
            print('Name                 :',self.__name__)
            print('Average time/sim(s)  :',np.mean(self.clock.values))
            print('Sucess rate          :',self.success,'/',self.iter)
            print('Failed sim. id       :',self.fail_index)
            print('\n')
            print('-----------------------------------------------------')
            print('                      DATA INFO                      ')
            print('-----------------------------------------------------')
            print('\n')
            print(self.collective)
            print('\n')

        elif self.module == "ML":
            print('********************** START ************************')
            print('\n')
            print('                    F3DASM SUMMARY                   ')
            print('\n')
            print('*****************************************************')
            print(self.in_data)
            print('-----------------------------------------------------')
            print('                    TRAINING INFO                    ')
            print('-----------------------------------------------------')
            print('\n')
            print('Name                 :',self.__name__)
            print('Average time/sim(s)  :',np.mean(self.clock.values))
            print('\n')


        return '********************* FINISH ***********************'

    def log(self):

        """ Method: Activate logging option for a cleaner terminal """

        log = open(self.__name__+".log", "w")
        sys.stdout = log
        print(self)


class ML():
    """
        Set of rules to be followed for the ML modules of your choice

        * Inherit from this class to your module in order to use its 
            functionalities.
    """
 
    def __init__(self, in_data, model,**kwargs):

        """ Initializing Method """

        
        self.in_data = in_data                          # Doe initialization
        self.out_data = 0                               # Initialize output
        model(in_data)
        self.model = model                              # Model initialization
        self.iter = model.epochs                        # Number of training iterations

        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = False
            
        if 'disable_tqdm' in kwargs:
            self.disable_tqdm = kwargs['disable_tqdm']
        else:
            self.disable_tqdm = False

    def post_init(self):

        """ Method: Intialize output and name output """

        pass 

    def to_do(self,i):

        """ Method: Define set of things to be done """

        pass
        
    def save(self,filename):
        
        """ Method: Pickle the output """  

        self.out_data.to_pickle(filename)

            
    def run(self):
        
        """ 
            Method: Start the fancy for loop for the jobs defined in 
                    self.to_do() 
        """

        self.post_init()                                # Call for the user-defined self.post_init()

        clock= np.zeros((self.iter,1))                  # Simulation time slots

        for i in tqdm(range(self.iter),disable=self.disable_tqdm):                # Fancy loop
            tic = time.perf_counter()               
            self.to_do(i)                               # Set of things to be done each iteration
            toc = time.perf_counter()
            clock[i] = toc - tic

        self.success = np.count_nonzero(np.mean(self.out,axis=1))               # Check success 
        self.fail_index =  np.where(np.mean(self.out,axis=1) == 0)[0]           # Spit fail-indecies

        self.out_data= DATA(self.out,self.out_var)            # Convert output from numpy to pandas
        self.clock = DATA(clock,['time(s)'])                  # Put simulation times to pandas


    def __str__(self):

        """ Overwrite print function """
        
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.set_option('display.expand_frame_repr', False)
        
        if self.module == "FEM":

            print('********************** START ************************')
            print('\n')
            print('                    F3DASM SUMMARY                   ')
            print('\n')
            print('*****************************************************')
            print(self.in_data)
            print('-----------------------------------------------------')
            print('                   SIMULATION INFO                   ')
            print('-----------------------------------------------------')
            print('\n')
            print('Name                 :',self.__name__)
            print('Average time/sim(s)  :',np.mean(self.clock.values))
            print('Sucess rate          :',self.success,'/',self.iter)
            print('Failed sim. id       :',self.fail_index)
            print('\n')
            print('-----------------------------------------------------')
            print('                      DATA INFO                      ')
            print('-----------------------------------------------------')
            print('\n')
            print(self.collective)
            print('\n')

        elif self.module == "ML":
            print('********************** START ************************')
            print('\n')
            print('                    F3DASM SUMMARY                   ')
            print('\n')
            print('*****************************************************')
            print(self.in_data)
            print('-----------------------------------------------------')
            print('                    TRAINING INFO                    ')
            print('-----------------------------------------------------')
            print('\n')
            print('Name                 :',self.__name__)
            print('Average time/sim(s)  :',np.mean(self.clock.values))
            print('\n')


        return '********************* FINISH ***********************'

    def log(self):

        """ Method: Activate logging option for a cleaner terminal """

        log = open(self.__name__+".log", "w")
        sys.stdout = log
        print(self)

##############################################################################################
# PURGATORY
##############################################################################################


class SIMULATION_old():
    """
        Set of rules to be followed for the SIMULATION module of your choice

        * Inherit from this class to your module in order to use its 
            functionalities.
    """
 

    def __init__(self, doe, model):

        """ Initializing Method """

        assert issubclass(type(doe), DOE), " Follow the rules and  inherit from the DOE base class! "

        self.doe = doe                                  # Doe initialization
        self.out = 0                                    # Initialize output
        self.model = model                              # Model initialization
        self.iter = doe.data.shape[0]                   # Number of simulations

    def post_init(self):

        """ Method: Intialize output and name output """

        pass 

    def to_do(self,i):

        """ Method: Define set of things to be done """

        pass
        
    def save(self,name):
        
        """ Method: Pickle the output """  

        self.pandas_out.to_pickle(name)

            
    def run(self):
        
        """ 
            Method: Start the fancy for loop for the jobs defined in 
                    self.to_do() 
        """

        self.post_init()                                # Call for the user-defined self.post_init()

        clock= np.zeros((self.iter,1))                  # Simulation time slots

        for i in tqdm(range(self.iter)):                # Fancy loop
            tic = time.perf_counter()               
            self.to_do(i)
            toc = time.perf_counter()
            clock[i] = toc - tic

        self.success = np.count_nonzero(np.mean(self.out,axis=1))               # Check success 
        self.fail_index =  np.where(np.mean(self.out,axis=1) == 0)[0]           # Spit fail-indecies

        self.out = self.out.astype('float')
        self.out[self.out==0.] = np.nan                 # Put NaN for failed simulations

        self.pandas_out= pd.DataFrame(self.out,columns=self.out_var)            # Convert output from numpy to pandas
        self.pandas_clock = pd.DataFrame(clock,columns=['time(s)'])             # Put simulation times to pandas

        
    def __str__(self):

        """ Overwrite print function """
        
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.set_option('display.expand_frame_repr', False)

        print('********************** START ************************')
        print('\n')
        print('                    F3DASM SUMMARY                   ')
        print('\n')
        print('*****************************************************')
        print(self.doe)
        print('-----------------------------------------------------')
        print('                   SIMULATION INFO                   ')
        print('-----------------------------------------------------')
        print('\n')
        print('Name                 :',self.__name__)
        print('Average time/sim(s)  :',np.mean(self.pandas_clock.values))
        print('Sucess rate          :',self.success,'/',self.iter)
        print('Failed sim. id       :',self.fail_index)
        print('\n')
        print('-----------------------------------------------------')
        print('                      DATA INFO                      ')
        print('-----------------------------------------------------')
        print('\n')
        print(self.collective)
        print('\n')
        return '********************* FINISH ***********************'

    def log(self):

        """ Method: Activate logging option for a cleaner terminal """

        log = open(self.__name__+".log", "w")
        sys.stdout = log
        print(self)




         
