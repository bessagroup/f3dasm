import os
import pickle
import numpy as np
import pandas as pd
import torch
import warnings

class DATA():
    """
        
        Data structure like class for allowing data passage between different
        modules inside f3dasm.

    """

    def __init__(self, data, keys=None):

        """ Initialize """

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

        """ After initialization changing the object """

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
        
        """ Overwrite print function """

        return str(self.DataFrame)

    def pandas_frame(self):

        """ Method: Create pandas DataFrame """

        return pd.DataFrame(self.values,columns=self.keys)

    def torch_tensor(self):

        """ Method: Create pytorch tensor """

        return torch.tensor(self.values).float()
    
    def to_pickle(self, filename):

        """ Method: Pickle DataFrame """

        return self.DataFrame.to_pickle(filename)

    def to_csv(self, filename):

        """ Method: Save DataFrame as .csv file """

        return self.DataFrame.to_csv(filename)

    def read_pickle(self,filename):

        """ Method: Read pickled file, iff it is pandas DataFrame """

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


           

