import numpy as np
import copy
import os
import tensorflow as tf
from SALib.sample import saltelli
from SALib.analyze import sobol
from timeit import default_timer as timer
from scipy import interpolate
from scipy import stats
import pandas as pd
import seaborn as sns
from scipy.integrate import simps, trapz
import h5py
from matplotlib.colors import LogNorm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)


# Matplotlib

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
from matplotlib import rc
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib as mpl
from scipy import stats

# Preprocessing data:
from sklearn.preprocessing import StandardScaler
import gpflow 
import logging
logging.basicConfig(format='%(asctime)s %(message)s')

# Postprocessing metrics:
from sklearn.metrics import mean_squared_error, r2_score , explained_variance_score, classification_report
import GPy
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pprint
from matplotlib.ticker import FuncFormatter

# To save the model:
from sklearn.externals import joblib
import os
import errno
import scipy.io as sio
try:
    import cPickle as pickle  # Improve speed
except ImportError:
    import pickle



class Unit_cell():
    
    def __init__(self, data, cell_params, D1):
        self._Response = {}
        
        P_D = cell_params[5]
        Length = P_D*D1
        maxStrain = Length/Length
        
        
        temp_U = data['riks_RP_Zplus_U'][:-2,2] 
        temp_F = data['riks_RP_Zplus_RF'][:-2,2] 
        stress_values = 1000*(temp_F/(np.pi*0.25*(D1)**2))
        strain_values = temp_U/Length   
        vec_indices = np.where(stress_values <= 0)

        stress_values_trimmed = abs(stress_values[vec_indices])
        strain_values_trimmed = abs(strain_values[vec_indices])
        
        if strain_values_trimmed[-1]<1.0:       
            strain_values_trimmed = np.append(strain_values_trimmed,1.0)
            stress_values_trimmed = np.append(stress_values_trimmed,0.0)
        
        strain = np.linspace(np.min(strain_values_trimmed),np.max(maxStrain),10000)
        interp = interpolate.pchip(strain_values_trimmed,stress_values_trimmed)                                                             
        stress = interp(strain)
        
        self._Response['max_strain'] = np.max(strain[:-1])
        self._Response['max_stress'] = np.max(stress[:-1])
        self._Response['strain'] = strain[:-1]
        self._Response['stress'] = stress[:-1]
        self._Response['Eabs'] = simps(stress,x = strain)
        self._Response['Pcrit'] = data['P_p3_crit'][0]/(np.pi*0.25*(D1)**2)*1000 
        self._Response['coilable'] = data['coilable'][0] 
        
    def plot_response(self):
        fig = plt.figure()
        plt.plot(self._Response['strain'], self._Response['stress'])
        plt.xlabel('strain')
        plt.ylabel('stress')
        plt.grid(True)
        return fig


class Dataset():


    
    def __init__(self, dir_path,
                 analysis_name = 'DOE_Ix-PD-100', 
                 DoE_dir = '1_DoEs/', 
                 Input_dir = '2_Inputs/', 
                 postproc_dir = '4_Postprocessing/'):
        
        # Loading the Data
        file_DoE = dir_path+'/'+DoE_dir+analysis_name
        file_Input = dir_path+'/'+Input_dir+analysis_name
        file_postproc = dir_path+'/'+postproc_dir+analysis_name+'/'+'STRUCTURES_postprocessing_variables.p'
        
        self._DoE = self.load_struct(file_DoE, 'DoE')
        self._Inputs = self.load_struct(file_Input, 'Input')        
        self._STRUCTURES_data = self.load_postproc(file_postproc)
        

        self._Response = {}             #Dict for Postprocessed results        
        self._X = [[[]]]                  #list of DoE parameters corresponding to succesfull simulations 
        
        self._failed_samples = None       
        self._good_samples = None  #list of indices of good samples
    
    def load_struct(self, file_path, struct_name): 
        
        '''
        file_path  - path to the input file
        struct name - name of the field of matlab struct: 'DoE' or 'Input'
        '''
        
        Input_data = sio.loadmat(file_path)
        Input_dict = {n: Input_data[struct_name][n][0][0] for n in Input_data[struct_name].dtype.names}

        for var in Input_dict.keys():
            if Input_dict[var].dtype.name == 'object':  #this loop deals with converting strings
                vlis = []
                for v in Input_dict[var][0]:        
                    vlis.append(v[0])
                Input_dict[var] = vlis

        return Input_dict
    
    def load_postproc(self, file_postproc):    
        #postproc_dir = '4_Postprocessing/'
    
        with open(file_postproc, 'rb') as pickle_file:
            try:
                STRUCTURES_data = pickle.load(pickle_file, encoding='latin1') #for python 3
            except Exception as e:
                STRUCTURES_data = pickle.load(pickle_file)   
        
        return STRUCTURES_data
        
    def postproc(self, D1 = 100.0, filtering = True, messages = False, 
                       params_out = ['Pcrit', 'Eabs','max_strain', 'max_stress', 'coilable']):
        '''
        params - list of parameters to retrieve, 
                 possible: 
                 params = ['Pcrit', 'Eabs','strain', 'stress', 'max_strain', 'max_stress','coilable']
        '''
        
        
        self._D1 = D1 #Dimeter of the unit cell base [mm]
        input_dim  = int(self._Inputs['size'][0][0])
        imperfection_dim = len(self._STRUCTURES_data["Input1"])         #number of imperfection parameters
        doe_dim = int(self._DoE['size'][0][0])
            
        self._failed_samples = []
        self._good_samples = [[[]]]
        
        for k  in params_out:
            self._Response[k] = [[[]]]#np.zeros((input_dim, imperfection_dim, doe_dim))
        
        #RETREIVE COILABLE DESIGNS SATISFYING COMPRESSIBILITY >80%
        for iInput in range(input_dim):
            for kImperf in range(0,imperfection_dim):
                    for jDoE in range(0,doe_dim):
                        data = self._STRUCTURES_data['Input'+ str(iInput+1)]['Imperfection'+str(kImperf+1)]
                        
                                      #0     1    2   3   4     5    6
                        #cell_params: #A_D1, G_E,I_x, Iy, J_D1, P_D, ConeSlope
                        cell_params = self._DoE['points'][jDoE]
                        
                        try:
                            if 'DoE'+str(jDoE+1) in data:
                                data = self._STRUCTURES_data['Input'+str(iInput+1)]['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]        

                                mm_sample = Unit_cell(data, cell_params, self._D1)

                                for k  in params_out:
                                    self._Response[k][iInput][kImperf].append(mm_sample._Response[k])
                                self._good_samples[iInput][kImperf].append(jDoE)

                            else:
                                self._failed_samples.append([iInput,kImperf, jDoE ]) 

                        except:
                            self._failed_samples.append([iInput,kImperf, jDoE ])                          
                            if messages:
                                print('Failed Sample (ikj):', iInput, kImperf, jDoE)
                        
                        self._X[iInput][kImperf] = np.array(self._DoE['points'][np.array(self._good_samples[iInput][kImperf])])
                        
                    if kImperf - 1< imperfection_dim:
                        self._good_samples[iInput].append([])
                        self._X[iInput].append([])
                        self._Response[k][iInput].append([])
            
            if iInput - 1< input_dim:
                self._good_samples.append([])
                self._X.append([])
                self._Response[k].append([])



plt.style.use('seaborn')

def plot_GP(mu, cov, Xnew, training_set = None):
    
    fig = plt.figure(figsize = (12, 6))
    plt.plot(Xnew, mu[:, -1], c = 'darkcyan', label = 'mean')
    plt.fill_between(Xnew, (mu-2*cov)[:, -1], (mu+2*cov)[:, -1], alpha = 0.3, color = 'darkcyan',label = '2 std' )
    
    if training_set is not None:
        Xtr = training_set._scaler.inverse_transform(training_set._Xtrain)
        
        plt.plot(Xtr, training_set._Ytrain, 'k.', label = 'training data')
        
    plt.legend()
    return fig


class GPR_sklearn():
        
    def __init__(self,training_dataset,   
                 kernel = Matern(length_scale = 1.0, nu = 2.5, length_scale_bounds=(1e-3, 100.0))):
        
        self._training_dataset = training_dataset
        self._scaler = training_dataset._scaler        
        self._model = GaussianProcessRegressor(kernel=kernel, alpha=1e-10 , n_restarts_optimizer=10)
        self._model.fit(self._training_dataset._Xtrain, self._training_dataset._Ytrain)

    def train(self):        
        self._model.fit(self._training_dataset._Xtrain, self._training_dataset._Ytrain)
    
    def predict(self, X_predict):
        if X_predict.ndim ==1:
            X_predict = X_predict.reshape(-1, 1)
        
        Xnew = self._scaler.transform(X_predict)          #rescaling using the same scaler as training data
        return self._model.predict(Xnew, return_std=True)     
        
        
        
class GPR_GPy():
        
    def __init__(self,training_dataset, kernel = None):
        
        if kernel is None:
            kernel = GPy.kern.Matern52(input_dim =training_dataset._Xtrain.shape[-1], name = 'matern')
    
        self._scaler =  training_dataset._scaler
        self._model=GPy.models.GPRegression(training_dataset._Xtrain, training_dataset._Ytrain, kernel)
        self._model.optimize()
        self._model.optimize_restarts(3)
        
    def predict(self, X_predict):
        
        if X_predict.ndim ==1:
            X_predict = X_predict.reshape(-1, 1)

        Xnew = self._scaler.transform(X_predict)            #rescaling using the same scaler as training data
        return self._model.predict(Xnew, full_cov = False)     
        

    


class TrainingDataset():
    '''
    a class for preprocessing dataset for ML training
    
    _Xtrain - x values, np array
    _Ytrain - y values, np array
    _scaler - sklearn StandardScaler used to rescale X data for training
    
    '''
    
    def __init__(self, X_train, Y_train, test_size = 0.9, seed = 0):
        '''
        test_size = % of data used for training - 0.9 = 90% of the data
        '''
        
        
        if type(Y_train) ==list:
            Y_train = np.array(Y_train)
        
        if X_train.ndim ==1:
            X_train= X_train.reshape(-1, 1)
        
        if Y_train.ndim ==1:
            Y_train =Y_train.reshape(-1, 1)
        
        np.random.seed(seed)
        indices = np.random.permutation(np.shape(X_train)[0])
        
        X = X_train[indices[:int(round(len(indices)*test_size))]]
        Y = Y_train[indices[:int(round(len(indices)*test_size))]]
        
        scaler = StandardScaler().fit(X)
        self._scaler = scaler
        self._Xtrain = scaler.transform(X)
        self._Ytrain = Y

