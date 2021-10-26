import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data import *

class DOE():
    """
        Set of rules to be followed for the DOE module of your choice

        * Inherit from this class to your module in order to use its 
            functionalities.
    """
    def __init__(self,num,variable):

        """ Initialize """

        self.variable = variable                        # Variable definitions as python dictionary
        self.num = num                                  # Total desired number of dictionaries
        self.dim = len(self.variable)                   # Dimension of the feature space
        self.keys = list(self.variable.keys())          # Names of the feature
        self.values = np.zeros((num,self.dim))          # Initialize the data-frame with numpy to put your doe 
        self.DATA = DATA(self.values,self.keys)

    def __str__(self):

        """ Overwrite print function """

        print('-----------------------------------------------------')
        print('                       DOE INFO                      ')
        print('-----------------------------------------------------')
        print('\n')
        print('Module Name          :',self.__name__)
        print('Method               :',self.method)
        print('Feature dimension    :',self.dim)
        print('Feature object count :',self.num)
        return '\n'
 
    
    def plot(self,feature=(1,2), bins=(100,500,1000)):
        
        """ Method: Plot doe points for the selected feature dimensions """
        
        fig = plt.figure()                              # Create figure object 
        ax = fig.add_subplot(1, 1, 1)                   # Create axis for the figure
        colors = ("r", "b", "g")                        # Define colors for bins 

        assert(len(bins) == len(colors))                # Keep the bin dimension and color dimension same

        count = 0
        for i,n in enumerate(bins):                     # Loop for plotting with desired colors
            ax.scatter(self.data[count:n,feature[0]-1],self.data[count:n,feature[1]-1],color=colors[i])
            count =n 

        ################################
        # Label your Axes
        ################################
        plt.ylabel(self.names[feature[0]-1])           
        plt.xlabel(self.names[feature[1]-1])

    def save(self,filename):

        """ Method: Pickle the doe points """  

        self.DATA.to_pickle(filename)

##################################################################################################
# PURGATORY
##################################################################################################
class DOE_past():
    """
        Set of rules to be followed for the DOE module of your choice

        * Inherit from this class to your module in order to use its 
            functionalities.
    """
    def __init__(self,num,variable):

        """ Initialize """

        self.variable = variable                        # Variable definitions as python dictionary
        self.num = num                                  # Total desired number of dictionaries
        self.dim = len(self.variable)                   # Dimension of the feature space
        self.names = list(self.variable.keys())         # Names of the feature
        self.data = np.zeros((num,self.dim))            # Initialize the data-frame with numpy to put your doe 
        self.dataFrame()                                # Call for the internal method to convert numpy array to pandas DataFrame

    def __str__(self):

        """ Overwrite print function """

        print('-----------------------------------------------------')
        print('                       DOE INFO                      ')
        print('-----------------------------------------------------')
        print('\n')
        print('Module Name          :',self.__name__)
        print('Method               :',self.method)
        print('Feature dimension    :',self.dim)
        print('Feature object count :',self.num)
        return '\n'
 
    
    def dataFrame(self):

        """ Method: Convert numpy doe to pandas doe """ 

        self.pandas_data = pd.DataFrame(self.data,columns=self.names)
    
    def plot(self,feature=(1,2), bins=(100,500,1000)):
        
        """ Method: Plot doe points for the selected feature dimensions """
        
        fig = plt.figure()                              # Create figure object 
        ax = fig.add_subplot(1, 1, 1)                   # Create axis for the figure
        colors = ("r", "b", "g")                        # Define colors for bins 

        assert(len(bins) == len(colors))                # Keep the bin dimension and color dimension same

        count = 0
        for i,n in enumerate(bins):                     # Loop for plotting with desired colors
            ax.scatter(self.data[count:n,feature[0]-1],self.data[count:n,feature[1]-1],color=colors[i])
            count =n 

        ################################
        # Label your Axes
        ################################
        plt.ylabel(self.names[feature[0]-1])           
        plt.xlabel(self.names[feature[1]-1])
                
    def save(self,name):

        """ Method: Pickle the doe points """  

        self.pandas_data.to_pickle(name)


