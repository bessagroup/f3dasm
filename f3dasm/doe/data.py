#######################################################
# Data class for the manipulation and transformation  #
# of data within F3DASM                               #
#######################################################

import os
import pickle
import numpy as np
import pandas as pd
import torch
import warnings

class DATA():
    """
        Data structure for data conversion and transfer between
        modules inside f3dasm.
    """

    def __init__(self, data, keys=None):

        """ Initialize data structure
        
        Args:
            data : pandas DataFrame, numpy array or pytorch tensor
            keys (list): names of data columns. Required for converting from numpy array or pythorch tensor 

        Returns:
            F3DASM data structures:  DataFrame, numpy array or pytorch tensor, values and feature names (keys)
        """

        ################################
        # If you have pandas DataFrame
        ################################
        if isinstance(data,pd.DataFrame):
            self.DataFrame = data
            self.values = self.DataFrame.values
            self.keys = self.DataFrame.columns.values.tolist()
            self.tensor = self.torch_tensor()
        
        ################################
        # If you have numpy array -> must name your data though!
        ################################
        elif isinstance(data,np.ndarray):
            assert keys != None, "Name your data!"
            if data.shape[0] < data.shape[1]:
                warnings.warn("Make sure you store your data in columns!", RuntimeWarning)

            self.values = data
            self.keys = keys
            self.tensor = self.torch_tensor()
            self.DataFrame = self.pandas_frame()
            
        ################################
        # If you have pytorch tensor -> must name your data though!
        ################################
        elif isinstance(data,torch.Tensor):
            assert keys != None, "Name your data!"
            if data.shape[0] < data.shape[1]:
                warnings.warn("Make sure you store your data in columns!", RuntimeWarning)

            self.values = data.numpy()
            self.keys = keys
            self.tensor = data
            self.DataFrame = self.pandas_frame()

        else:
            self.read_pickle(data)

    def __call__(self, data, keys=None):

        """ Parses data object and creates different representations"""

        ################################
        # If you have pandas DataFrame
        ################################
        if isinstance(data,pd.DataFrame):
            self.DataFrame = data
            self.values = self.DataFrame.values
            self.keys = self.DataFrame.columns.values.tolist()
            self.tensor = self.torch_tensor()

        ################################
        # If you have numpy array -> must name your data though!
        ################################
        elif isinstance(data,np.ndarray):
            assert keys != None, "Name your data!"
            if data.shape[0] < data.shape[1]:
                warnings.warn("Make sure you store your data in columns!", RuntimeWarning)

            self.values = data
            self.keys = keys
            self.tensor = self.torch_tensor()
            self.DataFrame = self.pandas_frame()

        ################################
        # If you have pytorch tensor -> must name your data though!
        ################################
        elif isinstance(data,torch.Tensor):
            assert keys != None, "Name your data!"
            if data.shape[0] < data.shape[1]:
                warnings.warn("Make sure you store your data in columns!", RuntimeWarning)

            self.values = data.numpy()
            self.keys = keys
            self.tensor = data
            self.DataFrame = self.pandas_frame()

        else:
            self.read_pickle(data)
            
    def __str__(self):
        
        """ Overwrites print function """

        return str(self.DataFrame)

    def pandas_frame(self):

        """ Method: Creates pandas DataFrame """

        return pd.DataFrame(self.values,columns=self.keys)

    def torch_tensor(self):

        """ Method: Creates pytorch tensor """

        # return torch.tensor(self.values).float()
    
    def to_pickle(self, filename):

        """ Method: Saves DataFrame as Pickle file """

        return self.DataFrame.to_pickle(filename)

    def to_csv(self, filename):

        """ Method: Saves DataFrame as .csv file """

        return self.DataFrame.to_csv(filename)

    def read_pickle(self,filename):

        """ Method: Reads pickled file, only if it contains pandas DataFrame """

        name, ext = os.path.splitext(filename)
        assert ext == '.pkl' or ext == '.pickle', 'File extension should be .pkl or .pickle'
        loading = open(filename,"rb")
        self.DataFrame = pickle.load(loading)
        
        if isinstance(self.DataFrame,pd.DataFrame):
            self.values = self.DataFrame.values
            self.name  = self.DataFrame.columns.values.tolist()
            self.tensor = self.torch_tensor()
        else:
            NotImplementedError


           


