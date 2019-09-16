# -*- coding: utf-8 -*-
"""
@author: M.A. Bessa (M.A.Bessa@tudelft.nl)

If you want to do sensitivity analysis:

https://github.com/SALib/SALib
"""

# from IPython import get_ipython
# get_ipython().magic('reset -sf')

# print(__doc__)

import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
#from SALib.plotting.

# KERAS:
from keras.models import Sequential
from keras import optimizers
from keras.models import model_from_json
from keras.layers.core import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from keras import backend as K
import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
#from tensorflow.python.client import device_lib
#import mimic_alpha as ma
from timeit import default_timer as timer

start = timer()
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
#from sklearn.model_selection import train_test_split
# Machine learning method:
#from sklearn.svm import SVR
import gpflow 
#from gpflow.actions import Loop, Action, Condition, ActionContext
#from gpflow.training import AdamOptimizer
#import gpflow.training.monitor as mon
#from gpflow import settings


import logging
logging.basicConfig(format='%(asctime)s %(message)s')
import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)
#
from scipy import interpolate
from scipy import stats
from scipy.integrate import simps, trapz
import h5py
# Postprocessing metrics:
from sklearn.metrics import mean_squared_error, r2_score , explained_variance_score
# Validation
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import KFold
#
# To save the model:
from sklearn.externals import joblib
#import os
import errno
import scipy.io as sio
try:
    import cPickle as pickle  # Improve speed
except ImportError:
    import pickle
#
# Take care of GPU memory management:
#cfg = K.tf.ConfigProto()
#cfg.gpu_options.allow_growth = True
#K.set_session(K.tf.Session(config=cfg))

# -----------------------------------------------------------------------------


# Close all figures:
#plt.close("all")
# Take care of some general properties of the plots:
rc('text', usetex=True)
rc('font', family='serif',size=14)
#------------------------------------------------------------------------------
# Define the test size (from 0 to 1.0):
test_size=0.1


# Do you want to load a previously obtained machine learning model?
load_model = 1
# Choose Machine Learning method:
ML_method =4 # 0 = Scikit GPR, 1 = GPFlow GPR 2=NN 3=GPy sparse hetero 4=SGPR 5=SVGP 6=SMCMC
#
# How many points do you want to use to create the contour plots?
grid_points = 10000

D1 = 100.0

#------------------------------------------------------------------------------
# READ DATASET

#analysis_folder = 'AstroMat_1st_attempt'
#analysis_folder = 'AstroMat_5Params_PLA_batch1'
#analysis_folder = 'Single_AstroMat_1St_7Params_batch2'
analysis_folder = 'DOE_Single_Astromat_1St_7ParamsFirst_Imperfect_Coilable'
#analysis_folder = 'Single_AstroMat_1St_4Params'


#training_DoE_points = 700 # Number of input points used for training

dir_path = os.path.dirname(os.path.realpath(__file__))
#
# Read DoE file
DoE_dir = '1_DoEs/'
#file_DoE_path = dir_path+'/'+DoE_dir+'DOE_PD1_St_D1_D2D1_db_update.mat'
#file_DoE_path = dir_path+'/'+DoE_dir+'DOE_Single_Astromat_1St_7Params_batch2'
file_DoE_path = dir_path+'/'+DoE_dir+'DOE_TO_Classify_NEW'
#file_DoE_path = dir_path+'/'+DoE_dir+'DOE_Single_Astromat_1St_4Params.mat'
DoE_data = sio.loadmat(file_DoE_path)
X_all = DoE_data['DoE']['points'][0][0] # All the input points
X_all = X_all[:,:-1]
#******************ALTERNATIVELY THE CLASSIFIED
#with h5py.File('X_all_new_noiseless.h5', 'r') as hf:
#    X_all = hf['X_all_new_noiseless'][:]
# Extract the names of the features (input variables):
feature_names = []
for iFeature in range (0,len(DoE_data['DoE']['vars'][0][0][0])):
    feature_names.append(str(DoE_data['DoE']['vars'][0][0][0][iFeature][0]))
#end for
feature_names = feature_names[:-1]
#
# Read Input file:
Input_dir = '2_Inputs/'
#file_Input_path = dir_path+'/'+Input_dir+'INPUT_all_but_PD1_St_D1_D2D1_db.mat'
#file_Input_path = dir_path+'/'+Input_dir+'INPUT_Single_Astromat_1St_7Params_batch2'
file_Input_path = dir_path+'/'+Input_dir+'DOE_Single_Astromat_1St_7ParamsFirst_Imperfect'
#file_Input_path = dir_path+'/'+Input_dir+'INPUT_Single_Astromat_1St_4Params.mat'
Input_data = sio.loadmat(file_Input_path)
Input_points = Input_data['Input']['points'][0][0] # All the input points
#
# Read Postprocessing file
postproc_dir = '4_Postprocessing/'
file_postproc_path = dir_path+'/'+postproc_dir+analysis_folder+'/'+'STRUCTURES_postprocessing_variables.p'
with open(file_postproc_path, 'rb') as pickle_file:
    try:
        STRUCTURES_data = pickle.load(pickle_file, encoding='latin1') # for python 3
    except Exception as e:
        STRUCTURES_data = pickle.load(pickle_file) # for python 2
#
# File where to save the machine learning model:
machinelearn_dir = '5_MachineLearning/'
machinelearn_dir_path = dir_path+'/'+machinelearn_dir+analysis_folder+'/'
# If the folder doesn't exist, then create it:
try:
    os.makedirs(machinelearn_dir_path)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise
#
# Load output data
#Y = np.zeros(np.shape(X_all)[0])
#Y = []
#X = []
# Quick hack because I didn't run 4 Input points, but only 1:
Input_points = [Input_points[0]]
        
Y = [[] for _ in range(np.shape(Input_points)[0])] # Create a list of lists (for Inputs and DoE points)
        
Cuv_lost = [[] for _ in range(np.shape(Input_points)[0])] # Create a list of lists (for Inputs and DoE points)
        
Vec_lost = [[] for _ in range(np.shape(Input_points)[0])] # Create a list of lists (for Inputs and DoE points)
#Z = [[] for _ in range(np.shape(Input_points)[0])] # Create a list of lists (for Inputs and DoE points)
X = [[] for _ in range(np.shape(Input_points)[0])]
X1 = [[] for _ in range(np.shape(Input_points)[0])]
Y1_lost = [[] for _ in range(np.shape(Input_points)[0])]
X1_lost = [[] for _ in range(np.shape(Input_points)[0])]
Z_lost = [[] for _ in range(np.shape(Input_points)[0])]
Y1 = [[] for _ in range(np.shape(Input_points)[0])]
Y2 = [[] for _ in range(np.shape(Input_points)[0])]
Z1 = [[] for _ in range(np.shape(Input_points)[0])]
Z = [[] for _ in range(np.shape(Input_points)[0])]
X_lost = [[] for _ in range(np.shape(Input_points)[0])]  # Create a list of lists (for Inputs and DoE points)
Corresponding_DoE_points = [[] for _ in range(np.shape(Input_points)[0])] # Create a list of lists (for Inputs and DoE points)
Corresponding_DoE_points2 = [[] for _ in range(np.shape(Input_points)[0])]
Corresponding_DoE_points3 = [[] for _ in range(np.shape(Input_points)[0])]
#chi_square = [[] for _ in range(np.shape(Input_points)[0])]
#p_values = [[] for _ in range(np.shape(Input_points)[0])]
#QoI2_mean = [[] for _ in range(np.shape(Input_points)[0])]
#QoI2_stdev = [[] for _ in range(np.shape(Input_points)[0])]
#X1 = [[] for _ in range(np.shape(Input_points)[0])] # Create a list of lists (for Inputs and DoE points)
#Corresponding_DoE_points = [[] for _ in range(np.shape(Input_points)[0])] # Create a list of lists (for Inputs and DoE points)
#Y1 = [[] for _ in range(np.shape(Input_points)[0])] # Create a list of lists (for Inputs and DoE points)
#Z1 = [[] for _ in range(np.shape(Input_points)[0])] # Create a list of lists (for Inputs and DoE points)
#
input_dimensions = len(feature_names) # number of features (input space dimension)
#input_dimensions = input_dimensions
#

##############################################################################################################
imperfection_dim = len(STRUCTURES_data["Input1"])
doe_dim = len(STRUCTURES_data['Input1']['Imperfection1'])
##############################################################################################################
#
#plt.figure()
#ddx=[]
for iInput in range(np.shape(Input_points)[0]):
    count = 0
    for kImperf in range(0,imperfection_dim):
   
#            for jDoE in range(0,doe_dim):
#            for c,jDoE in enumerate([69547]):
            for c,jDoE in enumerate([70354]):
#            for c,jDoE in enumerate([ 49471, 17466, 76894, 67696,79572, 70354, 83253]):
                try:
                    
                    if 'DoE'+str(jDoE+1) in STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]:
#                    
#                        X1[iInput].append(X_all[jDoE])
                        
                        if STRUCTURES_data['Input'+str(iInput+1)]['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['coilable'][0] == 1 and STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['P_p3_crit'][0]*1.2 < STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['P_p3_crit'][1]:
                            Length = X_all[jDoE][5]*D1
#                            X1[iInput].append(X_all[jDoE])
                            maxStrain = Length/Length
                            X[iInput].append(X_all[jDoE])
                            Y[iInput].append(STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['P_p3_crit'][0]/(np.pi*0.25*(D1)**2)*1000 )
                           
                            temp_U= STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['riks_RP_Zplus_U'][:-2,2] 
                            temp_F = STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['riks_RP_Zplus_RF'][:-2,2] 
                            stress_values = 1000*(temp_F/(np.pi*0.25*(D1)**2))
                            strain_values = temp_U/Length   
                            vec_indices = np.where(stress_values <= 0)
                            stress_values_trimmed = abs(stress_values[vec_indices])
                            strain_values_trimmed = abs(strain_values[vec_indices])
#                            stress_values_trimmed = (-1)*(stress_values)
#                            strain_values_trimmed = (-1)*(strain_values)
                            max_location_stress = np.argmax(stress_values_trimmed)
                            max_location_strain = np.argmax(strain_values_trimmed)
#                            Corresponding_DoE_points[iInput].append(jDoE+1)
                            if np.max(strain_values_trimmed[max_location_strain])> np.min([0.8,maxStrain]):
#                                 Compute area below the curve:
#                                Corresponding_DoE_points[iInput].append(jDoE+1)
                                if strain_values_trimmed[-1]<1.0:
                                    strain_values_trimmed = np.append(strain_values_trimmed,1.0)
                                    stress_values_trimmed = np.append(stress_values_trimmed,0.0)
                                vector = np.linspace(np.min(strain_values_trimmed),np.max(maxStrain),10000)
                                interp = interpolate.pchip(strain_values_trimmed,stress_values_trimmed)
#                                derivatives = interp.derivative().roots()
                                
                                curve = interp(vector)
                                vector = vector[:-1]
                                curve = curve[:-1]
#                                curve=curve*1.1
                                area = simps(curve,x=vector)
                                
#                                if area < 70.0: # and area >2.5 :
                                
                                Z[iInput].append( area )
                                X1[iInput].append(X_all[jDoE])
                                Z1[iInput].append(maxStrain)
#                                Cuv_lost[0].append( curve )
#                                ddx.append(derivatives)
#                                Vec_lost[0].append( vector )
                                Y1[iInput].append(STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['P_p3_crit'][0]/(np.pi*0.25*(D1)**2)*1000 )
                                Y2[iInput].append(STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['coilable'][0])
                                Corresponding_DoE_points[iInput].append(jDoE+1)
                                
#                                if area > 48.0:
#                                    Corresponding_DoE_points2[iInput].append(jDoE+1)
                                    
#                                if STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['P_p3_crit'][0]/(np.pi*0.25*(D1)**2)*1000 >86.0:
##           
#                                Corresponding_DoE_points2[iInput].append(jDoE+1)
#                                plt.figure(33)
#                                plt.plot(vector,curve,'-', label='Noisy')
##        #                        y_temp = 1000*(y_temp/(np.pi*0.25*(100.0)**2))
#                                if area < 40.0 and area>30:
##           
#                                    Corresponding_DoE_points3[iInput].append(jDoE+1)
#                                    plt.figure(44)
#                                plt.plot(vector,curve,'-', label='Energy absorbed')
                                plt.plot(vector,curve,'-', label='Critical buckling load')
#                                if STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['P_p3_crit'][0]/(np.pi*0.25*(D1)**2)*1000 >75.0:
#                                plt.plot(vector,curve,'-', label=str(jDoE+1))
                    else:
                        X_lost[iInput].append(X_all[jDoE])

                    count = count + 1
                except Exception as e:
                    count = count # don't count this point
        #
    #

#Y = np.array(Y)
#Z = np.array(Z)
#X = np.array(X[0])
#QoI2_mean = np.array(QoI2_mean)
#QoI2_stdev = np.array(QoI2_stdev)
#               
#
plt.xlabel('Strain')
plt.ylabel('Stress [kPa]')
plt.legend(loc='upper right')
#                
Y = np.array(Y)
Cuv_lost = np.array(Cuv_lost[0])
Vec_lost = np.array(Vec_lost[0])
Y11 = np.array(Y1)
Y1_lost =np.array(Y1_lost)
Z1 = Z
Z = np.array(Z)
Z_lost = np.array(Z_lost)
X = np.array(X[0])
#Y = Y[:,:70000]
#X = X[:70000]

X_lost = np.array(X_lost[0])

# JUST A QUICK HACK TO CHECK WHAT ARE THE MAXIMUM VALUES AND CORRESPONDING INPUT
Y1 = Y1[0]
X1 = X1[0]
Z1 = Z1[0]
X_coil_noPlas_SORTED = [X1[i] for i in np.argsort(Y1)]
#X_coil_noPlas_SORTED = X1[0][np.argsort(Y1)[0]]
Y_coil_noPlas_SORTED = [Y1[i] for i in np.argsort(Y1)]
#Y_coil_noPlas_SORTED = Y1[0][np.argsort(Y1)[0]]
#Z1_coil_noPlas_SORTED = [Z1[0][i] for i in np.argsort(Y1)]
#Z_coil_noPlas_SORTED = [Z[0][i] for i in np.argsort(Y1)]
#
Corresponding_DoE_points_SORTED = [Corresponding_DoE_points[0][i] for i in np.argsort(Y1)]






# Calculate the mean and standard deviation for the different loading conditions
#Y_mean = Y[0]
# Quick hack to compute solid length
Y_mean = Z[0]
#Y_mean = Y11[0]
X = np.array(X1)
#Y_mean = np.array(Y)

#Y_std = np.zeros(np.shape(Y_mean))
Y_std = Y_mean*0.5
#Y_std = np.array(Y_std)
# -----------------------------------------------------------------------------

#for jInput_mean in range(1):
    # Split the dataset into training and testing:
#    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_mean[jInput_mean],test_size=0.10, random_state=1)
np.random.seed(0)
indices = np.random.permutation(np.shape(X)[0])
# [ 0.9  ,  0.675,  0.45 ,  0.225,  0.   ]
X_train = X[indices[:-int(round(len(indices)*test_size))]]
Y_mean_train = Y_mean[indices[:-int(round(len(indices)*test_size))]]
Y_std_train = Y_std[indices[:-int(round(len(indices)*test_size))]]
X_test  = X[indices[-int(round(len(indices)*test_size)):]]
Y_mean_test  = Y_mean[indices[-int(round(len(indices)*test_size)):]]
Y_std_test  = Y_std[indices[-int(round(len(indices)*test_size)):]]
#X_test = X_train
#Y_test = Y_train
#X_train = np.reshape(X_train,(-1,1))

# Scale the data
file_scaler_path = machinelearn_dir_path+'/'+'Sklearn_SCALER.sav'
#file_scaler_path = machinelearn_dir_path+'/'+'Sklearn_SCALER_SA.sav'
try:
    # Load the scaler for the data:
    scaler = joblib.load(file_scaler_path)
except Exception as e:        
    # Scale the data for better fit:
    scaler = StandardScaler().fit(X_train)
    # save to disk
    joblib.dump(scaler, file_scaler_path) # save the scaler
#
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)  


## Save the scaler for the data for future use:
#if ML_method == 0: # GRP
#    file_scaler_path = machinelearn_dir_path+'Input'+str(1)+'/'+'data_scaler_GPR_model.sav'
#elif ML_method == 1: # NN
#    file_scaler_path = machinelearn_dir_path+'Input'+str(1)+'/'+'data_scaler_NN_model.sav'
#else:
#    raise ValueError('Choose a valid ML_method')
#    
#try:
#    # Load the scaler for the data:
#    scaler = joblib.load(file_scaler_path)
#except Exception as e:        
#    # Scale the data for better fit:
#    scaler = StandardScaler()
#    joblib.dump(scaler, file_scaler_path) # save the scaler
#
#X_train_scaled = scaler.fit_transform(X_train)
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Fit ML model
machinelearn_Input_dir_path = machinelearn_dir_path +'Input'+str(1)
# If the folder doesn't exist, then create it:
try:
    os.makedirs(machinelearn_Input_dir_path)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise
#

# If using GPR:
if ML_method == 0:
#        my_kernel = 1.0 * RBF(length_scale=1.0)# length_scale_bounds=(1e-1, 10.0))
#        Y_mean_train = Y_mean_train.reshape(-1,1)
#        Y_std_train = Y_std_train.reshape(-1,1)
#        X_neww = X_train_scaled[:,0].reshape(-1,1) 
#        my_kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    my_kernel = 1.0 * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-1, 10.0))
#        my_kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
    #
    # !!!!!!!! WITHOUT regularization (no noise!):
#        reg = GaussianProcessRegressor(kernel=my_kernel, random_state=0, n_restarts_optimizer=10)
#        reg = GaussianProcessRegressor(kernel=my_kernel, alpha=0.1**2)
    # !!!!!!!! WITH REGULARIZATION (noisy data!):
    # -> Tikhonov regularization:
#        reg = GaussianProcessRegressor(kernel=my_kernel, alpha=(Y_std_train/Y_mean_train) ** 2, n_restarts_optimizer=0)
    reg = GaussianProcessRegressor(kernel=my_kernel, alpha=0.1**2 , n_restarts_optimizer=0)
    # -> Just considering the STDV:
#        reg = GaussianProcessRegressor(kernel=my_kernel, alpha=(Y_std_train) ** 2, random_state=0, n_restarts_optimizer=10)
    #
    reg.fit(X_train_scaled , Y_mean_train)
#    Y_pred, Y_std = reg.predict(scaler.transform(X_test), return_std=True)
    #X_test_neww = X_test[:,0].reshape(-1,1)
    Y_mean_pred, Y_std_pred = reg.predict(scaler.transform(X_test), return_std=True)
#        reg.scaler = scaler # Save also the scaler
#        dlugosc = reg.kernel_.k2.get_params()['length_scale']
#        sigma_f = np.sqrt(reg.kernel_.k1.get_params()['constant_value'])
    #
    print(reg.kernel_)
    # Save the model to disk
#        file_machinelearn_path = machinelearn_Input_dir_path+'/'+'GaussianProcess_model.sav'
#        joblib.dump(reg, file_machinelearn_path)
    #
elif ML_method == 1:
    my_kernel = gpflow.kernels.Matern52(input_dim=input_dimensions, lengthscales=1.0, variance=1.0, ARD=True)
    Y_mean_train = Y_mean_train.reshape(-1,1)
    m = gpflow.models.GPR(X_train_scaled, Y_mean_train, kern=my_kernel)
#        m.kern.variance = 3.0
    #noise = (Y_std_train/Y_mean_train) ** 2
    m.likelihood.variance = 0.1**2
    m.likelihood.variance.trainable = False
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)
    
    Y_mean_pred, Y_var_pred = m.predict_y(scaler.transform(X_test))
    Y_std_pred = np.sqrt(Y_var_pred)
#        reg.scaler = scaler # Save also the scaler
    dlugosc = m.kern.lengthscales
    sigma_f = m.kern.variance
    print(m)
    #
    # Save the model to disk
#        file_machinelearn_path = machinelearn_Input_dir_path+'/'+'GaussianPFlow_model.sav'
#        joblib.dump(m, file_machinelearn_path)
    #
# If using neural networks:
elif ML_method == 2:
    if load_model == 0: # Create model and save it
        reg = Sequential()
        reg.add(Dense(400, input_dim=input_dimensions, activation='relu'))
        reg.add(Dense(400, activation='relu'))
        reg.add(Dense(400, activation='relu'))
#            reg.add(Dense(400, activation='relu'))
#            reg.add(Dense(400, activation='relu'))
#            reg.add(Dense(400, activation='relu'))
#            reg.add(Dense(400, activation='relu'))
#            reg.add(Dense(400, activation='relu'))
#            reg.add(Dense(400, activation='relu'))

        reg.add(Dense(1))
        optimizer = optimizers.Adam(lr=0.0001)
        reg.compile(loss='mse', optimizer=optimizer)
        #
        # Define early stopping
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=10.0, patience=10)
    #    model.fit(X_train_scaled, Y_train,
    #              nb_epoch=1000, batch_size=20,
    #              validation_data=(scaler.transform(X_test), Y_test),
    #              callbacks=[early_stopping])
        #
        history = reg.fit(X_train_scaled, Y_mean_train, epochs=1000, batch_size=1000, validation_data=(scaler.transform(X_test), Y_mean_test))
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
        #
        # Evaluate model:
#        scores = reg.evaluate(scaler.transform(X_test), Y_mean_test, verbose=0)
#        print("%s: %.2f" % (reg.metrics_names, scores))
        #
#            reg.scaler = scaler
        #
        # Make predictions for the test data
        Y_mean_pred = reg.predict(scaler.transform(X_test))
        Y_std_pred = np.zeros(np.shape(Y_mean_pred)) # NN don't include UQ
        #
        # Save the model to disk
#            file_machinelearn_path = machinelearn_Input_dir_path+'/'+'NeuralNetworks_model_250'
#            # serialize model to JSON
#            reg_json = reg.to_json()
#            with open(file_machinelearn_path+'.json', "w") as json_file:
#                json_file.write(reg_json)
#            # serialize weights to HDF5
#            reg.save_weights(file_machinelearn_path+'.h5')
#            print("Saved model to disk")
        #
    elif load_model == 1: # Just load the model and move on!
        # load json
        file_machinelearn_path = machinelearn_Input_dir_path+'/'+'NeuralNetworks_model'
#            file_machinelearn_path = machinelearn_dir_path+'Input'+str(1)+'/'+'NeuralNetworks_model_Pcr'
        json_file = open(file_machinelearn_path+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        reg = model_from_json(loaded_model_json)
        # load weights into new model
        reg.load_weights(file_machinelearn_path+'.h5')
        print("Loaded model from disk")
         
        # evaluate loaded model on test data
#            reg.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        reg.compile(loss='mse', optimizer='AdaDelta', metrics=['accuracy'])
#            score = reg.evaluate(scaler.transform(X_test), Y_mean_test, verbose=0)
#            print("%s: %.2f" % (reg.metrics_names, score))
        #
        Y_mean_pred = reg.predict(scaler.transform(X_test))
#            Y_mean_pred2 = reg.predict(scaler.transform(XF_vector))
        Y_std_pred = np.zeros(np.shape(Y_mean_pred)) # NN don't include UQ

elif ML_method == 3:
#        my_kernel = GPy.kern.RBF(input_dim=4,variance=1.0,lengthscale=1.0)
#        X = X_train_scaled
#        Y = Y_mean_train.reshape(-1,1)
#        Y_std_train = Y_std_train.reshape(-1,1)
    # X and Y are N by 1 datasets
    #Z = np.random.choice(X[:,0], 12)[:,None] # Where 7 is the number of inducing inputs
    #Z = X[np.rando m.randint(X.shape[0], size=12), :]
#        Z=  X[0:80,:].copy()
#        variances = np.abs(np.random.uniform(0.01, 0.2, (Y_mean_train.shape[0], 1))*np.random.normal(size=Y_mean_train.shape).reshape(-1,1))
    variances = ((Y_std_train/Y_mean_train) ** 2).reshape(-1,1)
#        variances = 0.01*np.ones(np.shape(X_train_scaled)[0]).reshape(-1,1)
    lik = GPy.likelihoods.HeteroscedasticGaussian({'output_index':np.array(0)}, variance=variances)
    lik.fix()
    # variances are the variances for each datapoint
    m = GPy.core.SparseGP(X_train_scaled, Y_mean_train.reshape(-1,1), X_train_scaled[0:3000,:].copy(), GPy.kern.RBF(input_dim=input_dimensions, lengthscale=1), likelihood=lik, 
       Y_metadata={'output_index':np.arange(X_train_scaled.shape[0])}, 
       inference_method=GPy.inference.latent_function_inference.VarDTC()
    )
    
    m.optimize(optimizer='tnc',messages=True, max_iters=100)
    print(m)
#        Y_mean_pred, Y_var_pred = m.predict_noiseless(scaler.transform(X_test))
    p = m.predict_noiseless(scaler.transform(X_test))
    Y_mean_pred = p[0]
    Y_var_pred = p[1]
    Y_std_pred = np.sqrt(Y_var_pred)
#        reg.scaler = scaler # Save also the scaler
    dlugosc = m.rbf.lengthscale.values[0]
    sigma_f = m.kern.variance
    print(m)
    
elif ML_method == 4:
    
    my_kernel = gpflow.kernels.Matern52(input_dim=input_dimensions, lengthscales=1.0, variance=1.0, ARD=True)
    Y_mean_train = Y_mean_train.reshape(-1,1)
   # noise = (Y_std_train/Y_mean_train) ** 2
#        lik = GPy.likelihoods.HeteroscedasticGaussian({'output_index':np.array(0)}, variance=noise)
    #lik = gpflow.likelihoods.Gamma()
#        lik.fix()
    
#        m = gpflow.models.GPR(X_train_scaled, Y_mean_train, kern=my_kernel)
    m = gpflow.models.SGPR(X_train_scaled, Y_mean_train, kern=my_kernel, Z=X_train_scaled[0:1200].copy())
#        m.feature.set_trainable(False)
    print(m)
#        m.kern.variance = 3.0
    
#    m.likelihood.variance = 0.1**2
    m.likelihood.variance = 0.1
#    m.likelihood.variance.trainable = False
    
#        #CREATE MONITOR FOR GPFLOW
#        session = m.enquire_session()
#        global_step = mon.create_global_step(session)
#        
#        print_task = mon.PrintTimingsTask().with_name('print')\
#            .with_condition(mon.PeriodicIterationCondition(10))\
#            .with_exit_condition(True)
#        
#        sleep_task = mon.SleepTask(0.01).with_name('sleep').with_name('sleep')
#        
#        saver_task = mon.CheckpointTask('./monitor-saves').with_name('saver')\
#            .with_condition(mon.PeriodicIterationCondition(10))\
#            .with_exit_condition(True)
#        
#        file_writer = mon.LogdirWriter('./model-tensorboard')
#        
#        model_tboard_task = mon.ModelToTensorBoardTask(file_writer, m).with_name('model_tboard')\
#            .with_condition(mon.PeriodicIterationCondition(10))\
#            .with_exit_condition(True)
#        
#        lml_tboard_task = mon.LmlToTensorBoardTask(file_writer, m).with_name('lml_tboard')\
#            .with_condition(mon.PeriodicIterationCondition(100))\
#            .with_exit_condition(True)
#            
#        class CustomTensorBoardTask(mon.BaseTensorBoardTask):
#            def __init__(self, file_writer, model, X_test, Y_mean_test):
#                super().__init__(file_writer, model)
#                self.X_test = X_test
#                self.Y_mean_test = Y_mean_test
#                self._full_test_err = tf.placeholder(gpflow.settings.tf_float, shape=())
#                self._full_test_nlpp = tf.placeholder(gpflow.settings.tf_float, shape=())
#                self._summary = tf.summary.merge([tf.summary.scalar("test_rmse", self._full_test_err),
#                                                 tf.summary.scalar("test_nlpp", self._full_test_nlpp)])
#        
#            def run(self, context: mon.MonitorContext, *args, **kwargs) -> None:
#                minibatch_size = 1000
#                preds = np.vstack([self.model.predict_y(X_test[mb * minibatch_size:(mb + 1) * minibatch_size, :])[0]
#                                    for mb in range(-(-len(X_test) // minibatch_size))])
#                test_err = np.mean((Y_mean_test - preds) ** 2.0)**0.5
#                self._eval_summary(context, {self._full_test_err: test_err, self._full_test_nlpp: 0.0})
#              
#        custom_tboard_task = CustomTensorBoardTask(file_writer, m, X_test, Y_mean_test).with_name('custom_tboard')\
#            .with_condition(mon.PeriodicIterationCondition(100))\
#            .with_exit_condition(True)
#            
#        monitor_tasks = [print_task, model_tboard_task, lml_tboard_task, custom_tboard_task, saver_task, sleep_task]
#        monitor = mon.Monitor(monitor_tasks, session, global_step)
#        
#        if os.path.isdir('./monitor-saves'):
#            mon.restore_session(session, './monitor-saves')
        
        
        
    if load_model == 0: # Create model and save it
        opt = gpflow.train.ScipyOptimizer()
    #        opt = AdamOptimizer(0.01)
    #        with mon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
    #            opt.minimize(m, step_callback=monitor, maxiter=2000, global_step=global_step)
    ##
    #        file_writer.close()
    #        
        opt.minimize(m, disp=True)
        
        Y_mean_pred, Y_var_pred = m.predict_y(scaler.transform(X_test))
        Y_std_pred = np.sqrt(Y_var_pred)
    #        reg.scaler = scaler # Save also the scaler
        dlugosc = m.kern.lengthscales
        sigma_f = m.kern.variance
        print(m)
#     
        saver= gpflow.saver.Saver()
#        file_machinelearn_path = machinelearn_Input_dir_path+'/'+'Energy_noisyV3'
        file_machinelearn_path = machinelearn_Input_dir_path+'/'+'Buckling_noisySA'
        saver.save(file_machinelearn_path, m)
#        saver.save('./Buckling', m)
#        m.read_trainables()['SGPR/kern/variance']
        
    if load_model == 1: # Load model
        file_machinelearn_path = machinelearn_Input_dir_path+'/'+'Energy_noisyV2'
#        file_machinelearn_path = machinelearn_Input_dir_path+'/'+'Buckling_noisy'
        m =  gpflow.saver.Saver().load(file_machinelearn_path)
#        m =  gpflow.saver.Saver().load('./Buckling')
        m.read_trainables()['SGPR/kern/lengthscales']
        print(m)
        Y_mean_pred, Y_var_pred = m.predict_y(scaler.transform(X_test))
        Y_std_pred = np.sqrt(Y_var_pred)
        
elif ML_method == 5:
    my_kernel = gpflow.kernels.Matern52(input_dim=input_dimensions, lengthscales=1.0, variance=1.0, ARD=True)
    Y_mean_train = Y_mean_train.reshape(-1,1)
#        noise = (Y_std_train/Y_mean_train) ** 2

#        lik = gpflow.likelihoods.Gaussian(variance=0.01)
    lik = gpflow.likelihoods.Gaussian()
#        lik = gpflow.likelihoods.StudentT(scale=2.0)
#        lik.trainable = False
    
#        m = gpflow.models.GPR(X_train_scaled, Y_mean_train, kern=my_kernel)
#        m = gpflow.models.SVGP(X_train_scaled, Y_mean_train, kern=my_kernel,likelihood=lik, whiten=False, Z=X_train_scaled[0:1200].copy())
    m = gpflow.models.GPRFITC(X_train_scaled, Y_mean_train, kern=my_kernel, Z=X_train_scaled[0:1200].copy())
#        m = gpflow.models.SVGP(X_train_scaled, Y_mean_train, kern=my_kernel, likelihood=lik, Z=X_train_scaled[0:1200].copy())
#        m.feature.set_trainable(False)
    print(m)
#        m.kern.variance = 3.0
    
    m.likelihood.variance = 0.1**2
    m.likelihood.variance.trainable = False
    
    print(m)
            
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)
    
    Y_mean_pred, Y_var_pred = m.predict_y(scaler.transform(X_test))
    Y_std_pred = np.sqrt(Y_var_pred)
#        reg.scaler = scaler # Save also the scaler
    dlugosc = m.kern.lengthscales
    sigma_f = m.kern.variance
    print(m)
elif ML_method == 6:
#        my_kernel = gpflow.kernels.RBF(input_dim=4, lengthscales=1.0, variance=1.0) + gpflow.kernels.White(1, variance=0.01),
    Y_mean_train = Y_mean_train.reshape(-1,1)
#        noise = (Y_std_train/Y_mean_train) ** 2
    with gpflow.defer_build():
        m = gpflow.models.SGPMC(X_train_scaled, Y_mean_train, kern=gpflow.kernels.RBF(input_dim=input_dimensions, lengthscales=1.0, variance=1.0) + gpflow.kernels.White(1, variance=0.01), 
                 likelihood=gpflow.likelihoods.Gaussian(),
                 Z=X_train_scaled[0:200].copy())
        m.kern.kernels[0].variance.prior = gpflow.priors.Gamma(1.,1.)
        m.kern.kernels[0].lengthscales.prior = gpflow.priors.Gamma(1.,1.)
        m.kern.kernels[1].variance.trainables = False

    m.compile()
#        with gpflow.defer_build():
#            lik = gpflow.likelihoods.Gaussian()
#    #        m = gpflow.models.SGPR(X_train_scaled, Y_mean_train, kern=my_kernel, Z=X_train_scaled[0:80].copy())
#            m = gpflow.models.SGPMC(X_train_scaled, Y_mean_train, kern=my_kernel, likelihood=lik, Z=X_train_scaled[0:75].copy())
#            print(m)
#            m.kern.lengthscales.prior = gpflow.priors.Gamma(1., 1.)
#            m.kern.variance.prior = gpflow.priors.Gamma(1.,1.)
#           
#    #        m.feature.set_trainable(False)
#            print(m)
#        m.compile()                    
#        o = gpflow.train.AdamOptimizer(0.01)
#        o.minimize(m, maxiter=100) # start near MAP
    
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)
    
    s = gpflow.train.HMC()
    samples = s.sample(m, 100, epsilon=0.12, lmax=20, logprobs=False)#, verbose=True)
        
    Y_mean_pred, Y_var_pred = m.predict_y(scaler.transform(X_test))
    
    Y_std_pred = np.sqrt(Y_var_pred)
#        reg.scaler = scaler # Save also the scaler
#        dlugosc = m.kern.lengthscales
#        sigma_f = m.kern.variance
    print(m)

    #end if load_model
    #
# end if ML_method
#
# Print error metrics
#
print("------------- jInput_mean %i -------------"
      % (1,) )
print("The mean squared error is %0.3e" 
      % mean_squared_error(Y_mean_test, Y_mean_pred) )
print("The R2 score is %0.3f" 
      % r2_score(Y_mean_test, Y_mean_pred) )
print("The explained variance score is %0.3f" 
      % explained_variance_score(Y_mean_test, Y_mean_pred) )
#    print(dlugosc)
#    print(sigma_f)
#    
# -----------------------------------------------------------------------------
#######SENSITIVITY ANALYSIS
#X99=scaler.transform(X) 
#
#problem = {
#    'num_vars': len(feature_names),
#    'names': feature_names,
#    'bounds': [[np.min(X99[:,0]), np.max(X99[:,0])],
#               [np.min(X99[:,1]), np.max(X99[:,1])],
#               [np.min(X99[:,2]), np.max(X99[:,2])],
#               [np.min(X99[:,3]), np.max(X99[:,3])],
#               [np.min(X99[:,4]), np.max(X99[:,4])],
#               [np.min(X99[:,5]), np.max(X99[:,5])],
#               [np.min(X99[:,6]), np.max(X99[:,6])]]
##               [np.min(X99[:,7]), np.max(X99[:,7])]]
#               
#}
#
#param_values = saltelli.sample(problem, 12000)
##Y99 = np.zeros([param_values.shape[0]])
#Y99, shit = m.predict_y(param_values)
#Y99 = Y99.reshape(-1,)
##for i, Xi in enumerate(param_values):
###    Y99[i] =  m.predict_y(scaler.transform(X_test))[0]
##    Y99[i] =  m.predict_y(Xi)[0]
#
#Si = sobol.analyze(problem, Y99, print_to_console=True)

######SENSITIVITY ANALYSIS 2


#problem = {
#    'num_vars': len(feature_names),
#    'names': feature_names,
#    'bounds': [[1.0e-5, 4.096e-3 ],
#               [0.335, 0.45],
#               [1.1275e-11, 1.3981e-6],
#               [1.1275e-11, 1.3981e-6],
#               [1.353165e-11 , 6.76681e-6],
#               [0.25, 1.5],
#               [0.0, 0.8],
#               [np.min(X_all[:,7]), np.max(X_all[:,7])]]
#               
#}
#
#param_values = saltelli.sample(problem, 1000)
##Y99 = np.zeros([param_values.shape[0]])
#Y99, shit = m.predict_y(scaler.transform(param_values))
#Y99 = Y99.reshape(-1,)
##for i, Xi in enumerate(param_values):
###    Y99[i] =  m.predict_y(scaler.transform(X_test))[0]
##    Y99[i] =  m.predict_y(Xi)[0]
#
#Si = sobol.analyze(problem, Y99, print_to_console=True)
#Si = sobol.analyze(problem, Y99, print_to_console=True)
### PLOT FIGURES:
##import seaborn as sns
#
#plt.figure()
##ax = sns.barplot(x=problem['names'], y=Si['ST'], ci=Si['ST_conf'], capsize=.2)
##ax = sns.barplot(x=problem['names'], y=Si['S1'], ci=Si['S1_conf'], capsize=.2)
#x_pos = [i for i, _ in enumerate(problem['names'])]
#plt.bar(problem['names'],Si['ST'], yerr=Si['ST_conf'], capsize=5)
#plt.bar(problem['names'],Si['S1'], yerr=Si['S1_conf'], capsize=5)
#plt.xticks(rotation=45)
##plt.xlabel("Energy Source")
#N = 5
##men_means = (20, 35, 30, 35, 27)
##women_means = (25, 32, 34, 20, 25)
#
#ind = np.arange(N) 
#width = 0.35       
#plt.bar(ind, men_means, width, label='Men')
#plt.bar(ind + width, women_means, width,
#    label='Women')
#
#plt.ylabel('Scores')
#plt.title('Scores by group and gender')
#
#plt.xticks(ind + width / 2, ('G1', 'G2', 'G3', 'G4', 'G5'))
#plt.legend(loc='best')
#plt.show()
##plt.ylabel("Energy Output (GJ)")
##plt.title("Energy output from various fuel sources")
#
##plt.xticks(x_pos, x)
#
#plt.show()

#####################################################################################
###    #CONVERT features and switch to stress from force
##
##R2 = []
##Ypred7mean = []
##Ypred7stdv = []
##with open('4paramsNNvalidation.pickle', 'rb') as f:
###    with open('Sparse4paramsValidation.pickle', 'rb') as f:
##    PredXF, PredYF = pickle.load(f)
##for p in range(4):
##    if ML_method == 0:
##        Y_mean_pred_gridpoints_vector, Y_std_pred_gridpoints_vector = reg.predict(scaler.transform(PredXF[p]), return_std=True)
##    elif ML_method == 4 or ML_method == 1 or ML_method==5 or ML_method==6:
##        Y_mean_pred_gridpoints_vector, Y_var_pred_gridpoints_vector = m.predict_y(scaler.transform(PredXF[p]))
##        Y_std_pred_gridpoints_vector = np.sqrt(Y_var_pred_gridpoints_vector)
##    elif ML_method == 3:
##        Y_mean_pred_gridpoints_vector, Y_var_pred_gridpoints_vector = m.predict_noiseless(scaler.transform(PredXF[p]))
##        Y_std_pred_gridpoints_vector = np.sqrt(Y_var_pred_gridpoints_vector)
##    elif ML_method == 2:
##        Y_mean_pred_gridpoints_vector = reg.predict(scaler.transform(PredXF[p]))
##        
##        Y_std_pred_gridpoints_vector = np.zeros(np.shape(Y_mean_pred_gridpoints_vector))
##    print("The 4to7 R2 score is %0.3f" 
##      % r2_score(PredYF[p], Y_mean_pred_gridpoints_vector) )
##    R2.append(r2_score(PredYF[p], Y_mean_pred_gridpoints_vector))
##    Ypred7mean.append(Y_mean_pred_gridpoints_vector)
##    Ypred7stdv.append(Y_std_pred_gridpoints_vector)
#
##with open('7params_predictions_Sparse1300_coilableonly.pickle', 'wb') as z:
##    pickle.dump([Ypred7mean, Ypred7stdv], z)
##    X[:,0] = X[:,0]*D1**2
###    X[:,1] = X[:,1]*E
##    X[:,2] = X[:,2]*D1**4
##    X[:,3] = X[:,3]*D1**4
##    X[:,4] = X[:,4]*D1**4
##    X[:,5] = X[:,5]*D1
##    X[:,6] = X[:,6]*D1
##    Y = Y/Mastarea*1000 ####### MAXBUCKLING IN KILOPASCALS [kPa]
##
#####################################################################################
#    # SCATTER PLOTS
##    fig = plt.figure()
##    ax = fig.add_subplot(111, projection='3d')
##    
##    #ax.scatter(X_train[:,0], X_train[:,1], Y_train, c='r', marker='o')
##    #ax.scatter(X_test[:,0], X_test[:,1], Y_test, c='r', marker='o')
##    ax.scatter(X[:,0], X[:,1], Y_mean[jInput_mean], c='r', marker='o')
##    
##    ax.set_xlabel('X Label')
##    ax.set_ylabel('Y Label')
##    ax.set_zlabel('Z Label')
##    
##    plt.show()
##
##
##    fig = plt.figure()
##    ax = fig.add_subplot(111, projection='3d')
##    
###    ax.scatter(X_train[:,0], X_train[:,1], Y_mean_pred, c='r', marker='o')
###    ax.scatter(X_test[:,0], X_test[:,1], Y_mean_pred, c='r', marker='o')
##    ax.scatter(X[:,0], X[:,1], Y_mean_pred, c='r', marker='o')
##    
##    ax.set_xlabel('X Label')
##    ax.set_ylabel('Y Label')
##    ax.set_zlabel('Z Label')
##    
##    plt.show()
#    #
#fig1 = plt.figure() # 3D surface
##    fig1.set_size_inches(3.75, 2.75, forward=True)
#fig2 = plt.figure() # countour plot of mean
##    fig2.set_size_inches(3.75, 2.75, forward=True)
#fig3 = plt.figure() # contour plot of STDV
##    fig3.set_size_inches(3.75, 2.75, forward=True)
##
## Define the 2 features to plot and the remaining fixed values:
##    features_to_plot = [[0, 1 , 24 , 0.5], [0, 2 , 0.1, 0.5], [0, 3 , 0.1 , 24],
##                        [1, 2 , 60.0 , 0.7], [1, 3 , 60.0 , 24], [2, 3 , 60.0 , 0.1]]
##    features_to_plot = [[0, 1 , 6 , 0.5], [0, 1 , 12 , 0.5], [0, 1 , 24 , 0.5],
##                        [0, 1 , 48 , 0.5]]
##    features_to_plot = [[1, 2 , 87.5 , 0.0], [1, 2 , 87.5 , 0.2], [1, 2 , 87.5 , 0.4],
##                        [1, 2 , 87.5 , 0.6]]
##    features_to_plot = [[1, 2 , 225.0 , 0.0], [1, 2 , 225.0 , 0.2], [1, 2 , 225.0 , 0.4],
##                        [1, 2 , 225.0 , 0.6]]
##    features_to_plot = [[0, 1 , 75.0 , 0.6, 0.25], [0, 1 , 75.0 , 0.6, 0.5], [0, 1 , 75.0 , 0.6, 0.75],
##                        [0, 1 , 75.0 , 0.6, 1.0]]
##    features_to_plot = [[0, 1 , 75.0 , 0.0, 0.25], [0, 1 , 75.0 , 0.2, 0.25], [0, 1 , 75.0 , 0.4, 0.25],
##                        [0, 1 , 75.0 , 0.6, 0.25]]
##    features_to_plot = [[0, 1 , 32.9 , 0.79, 0.25], [0, 1 , 32.9 , 0.79, 0.5], [0, 1 , 32.9 , 0.79, 0.75],
##                        [0, 1 , 32.9 , 0.79, 1.0]]
##    features_to_plot = [[2, 3, 0.1, 4, 1.0], [0, 1, 10.0, 0.4, 1.0]]
##    features_to_plot = [[0, 1 , 75.0 , 0.2, 0.25], [0, 1 , 75.0 , 0.4, 0.25], [0, 1 , 75.0 , 0.6, 0.25],
##                        [0, 1 , 75.0 , 0.8, 0.25]]
##
##   0    ,           1       ,    2  ,        3       ,           4
##    '$D_1$' , '$\frac{P}{D_1}$' , '$EI$', '$\frac{d}{P}$', '$\frac{D_1-D_2}{D_1}$'
##    features_to_plot = [[3, 4 , 75.0, 0.2, 580.0], [3, 4 , 75.0, 0.53333, 580.0], [3, 4 , 75.0, 0.8, 580.0],
##                    [3, 4 , 75.0, 1.0, 580.0]]
##    features_to_plot = [[0, 2 , 0.53333, 0.02, 0.0], [0, 2 , 0.53333, 0.02, 0.25], [0, 2 , 0.53333, 0.02, 0.5],
##                        [0, 2 , 0.53333, 0.02, 0.75]]
##    features_to_plot = [[1, 2 , 75.0, 0.0375, 0.0], [1, 2 , 75.0, 0.0375, 0.25], [1, 2 , 75.0, 0.0375, 0.5],
##                        [1, 2 , 75.0, 0.0375, 0.75]]
##    0.17162329466495763, 4.249078965619972, 21.611504447114207, 0.2848660820176946, 0.8319719271332593
#
##   0    ,           1       ,    2             ,           3
##'$D_1$' , '$\frac{P}{D_1}$' , '$\frac{d}{D_1}$', '$\frac{D_1-D_2}{D_1}$'
##    features_to_plot = [[1, 2 , 75.0, 0.0], [1, 2 , 75.0, 0.25], [1, 2 , 75.0, 0.5],
##                    [1, 2 , 75.0, 0.75]]
# #         0    ,           1       ,       2        ,           3       ,        4         ,         5        ,             6
##    '$\frac{A}{D_1}$' ,'$\frac{G}{E}$' ,'$\frac{Ix}{D1}$' ,$\frac{Ix}{D1}$'  ,$\frac{J}{D1}$'  ,'$\frac{P}{D_1}$' ,'$\frac{D_1-D_2}{D_1}$',
##    features_to_plot = [#[0, 5 , 0.3937, 4.908739E-10, 4.908739E-10, 9.817477E-10, 0.0],
##                        #[0, 5 , 0.3937, 2.485049E-9, 2.485049E-9, 4.970098E-9, 0.25],
##                        #[0, 5 , 0.3937, 7.853982E-9, 7.853982E-9, 1.570796E-8, 0.5],
##                        [2, 5 5E-5, 0.6, 3.976078E-8, 7.952156E-8, 0.75]
##                    ]
##features_to_plot = [[2, 5, 5.0E-5, 0.6, 3.976078E-8, 7.952156E-8, 0.25], [2, 5, 5.0E-5, 0.6, 3.976078E-8, 7.952156E-8, 0.25],[2, 5, 5.0E-5, 0.6, 3.976078E-8, 7.952156E-8, 0.5],
##                [2, 5, 5.0E-5, 0.6, 3.976078E-8, 7.952156E-8, 0.75]]
##features_to_plot = [[2, 5, 2.22E-4, 0.36, 3.9761E-9, 7.82E-9, 0.0],
##                    [2, 5, 2.22E-4, 0.36, 3.9761E-9, 7.82E-9, 0.25],
##                    [2, 5, 2.22E-4, 0.36, 3.9761E-9, 7.82E-9, 0.5],
##                    [2, 5, 2.22E-4, 0.36, 3.9761E-9, 7.82E-9, 0.75]]
##features_to_plot = [[3, 4, 1E-3, 0.3415, 5.2E-7, 0.47, 0.0,  0.07],
##                    [3, 4, 1E-3, 0.3415, 5.2E-7, 0.47, 0.25, 0.07],
##                    [3, 4, 1E-3, 0.3415, 5.2E-7, 0.47, 0.5,  0.07],
##                    [3, 4, 1E-3, 0.3415, 5.2E-7, 0.47, 0.75, 0.07]]
##features_to_plot = [[3, 4, 1E-3, 0.3415, 5.2E-7, 0.47, 0.0,  ],
##                    [3, 4, 1E-3, 0.3415, 5.2E-7, 0.47, 0.25, ],
##                    [3, 4, 1E-3, 0.3415, 5.2E-7, 0.47, 0.5,  ],
##                    [3, 4, 1E-3, 0.3415, 5.2E-7, 0.47, 0.75, ]]
##features_to_plot = [[3, 5, 1E-3, 0.3415, 5.2E-7, 2.8E-6, 0.0],
##                    [3, 5, 1E-3, 0.3415, 5.2E-7, 2.8E-6, 0.25],
##                    [3, 5, 1E-3, 0.3415, 5.2E-7, 2.8E-6, 0.5],
##                    [3, 5, 1E-3, 0.3415, 5.2E-7, 2.8E-6, 0.75]]
#
#features_to_plot = [[2,3, 1E-3, 0.36, 2.5E-6, 0.66, 0.0],
#                    [2,3, 1E-3, 0.36, 2.5E-6, 0.66, 0.25],
#                    [2,3, 1E-3, 0.36, 2.5E-6, 0.66, 0.5],
#                    [2,3, 1E-3, 0.36, 2.5E-6, 0.66, 0.75]]
#
##PredYF= []
##with open('Energy_Matrix.pickle', 'rb') as q:
##    Z_mean_pred_gridpoints_matrices = pickle.load(q)
##with open('Classification.pickle', 'rb') as p:
##    XX,YY,Clas = pickle.load(p)
#
##
#for pairidx, pair in enumerate(features_to_plot):
#    # Contour plots for a lot of points
#    a_min, a_max = X[:, pair[0]].min() , X[:, pair[0]].max() 
#    b_min, b_max = X[:, pair[1]].min() , X[:, pair[1]].max()
#    #
#    plt.figure(fig1.number)
#    nRows_subplot = np.ceil(np.sqrt(len(features_to_plot)))
#    nCols_subplot = np.ceil(np.double(len(features_to_plot))/nRows_subplot)
#    ax = fig1.add_subplot(nRows_subplot, nCols_subplot, pairidx + 1, projection='3d')
##        plt.tight_layout(h_pad=3.0, w_pad=2.0, pad=2.5)
#    #
#    # Compute R2 score within the window of interest:
#    #    print_r2_score = r2_score(Y_mean_test[(X_test[:,0]<a_max)&(X_test[:,0]>a_min)&(X_test[:,1]<b_max)&(X_test[:,1]>b_min)], Y_mean_pred[(X_test[:,0]<a_max)&(X_test[:,0]>a_min)&(X_test[:,1]<b_max)&(X_test[:,1]>b_min)])
#    #    print("The R2 score within the window of interest is %0.3f" 
#    #           % print_r2_score)
#    #
#    # Draw surface by defining a grid of points and respective predictions:
#    a_vector = np.linspace(a_min,a_max,np.int(np.ceil(np.sqrt(grid_points))))
#    b_vector = np.linspace(b_min,b_max,np.int(np.ceil(np.sqrt(grid_points))))
#    #
#    A_matrix, B_matrix = np.meshgrid(a_vector,b_vector) # Uniform grid of points
#    #
#    XF_vector = np.ones((len(A_matrix.ravel()),np.shape(X)[1]))
##        XF_vector = [A_matrix.ravel(), B_matrix.ravel()] # convert inputs to vector format
##        XF_vector = np.transpose(XF_vector)
#    subplot_title = ' '
#    counter = 0
#    for iFeature in range(0,np.shape(X)[1]):
#        if iFeature == pair[0]:
#            XF_vector[:,iFeature] = A_matrix.ravel()
#        elif iFeature == pair[1]:
#            XF_vector[:,iFeature] = B_matrix.ravel()
#        else:
##            X_plot[:,iFeature] = X_train[np.shape(X_train)[0]/2,iFeature]*X_plot[:,iFeature]
#            counter = counter + 1
#            XF_vector[:,iFeature] = pair[1+counter]*XF_vector[:,iFeature]
#            subplot_title = subplot_title + str(feature_names[iFeature]) +'='+str(pair[1+counter])
#            if iFeature != np.shape(X)[1]-1: # then write a comma
#                subplot_title = subplot_title +', '
#            #end if
#        #end if
#    #
#    if ML_method == 0:
#        Y_mean_pred_gridpoints_vector, Y_std_pred_gridpoints_vector = reg.predict(scaler.transform(XF_vector), return_std=True)
#    elif ML_method == 4 or ML_method == 1 or ML_method==5 or ML_method==6:
#        Y_mean_pred_gridpoints_vector, Y_var_pred_gridpoints_vector = m.predict_y(scaler.transform(XF_vector))
#        Y_std_pred_gridpoints_vector = np.sqrt(Y_var_pred_gridpoints_vector)
#    elif ML_method == 3:
#        Y_mean_pred_gridpoints_vector, Y_var_pred_gridpoints_vector = m.predict_noiseless(scaler.transform(XF_vector))
#        Y_std_pred_gridpoints_vector = np.sqrt(Y_var_pred_gridpoints_vector)
#    elif ML_method == 2:
#        Y_mean_pred_gridpoints_vector = reg.predict(scaler.transform(XF_vector))
#        # Compute the stress and multiply by maximum amount of longerons that can be manufactured for this design:
##            Y_mean_pred_gridpoints_vector = Y_mean_pred_gridpoints_vector/( (200.0/2.0)**2.0*np.pi ) \
##                                            * (200.0*np.pi/(XF_vector))
##            Y_mean_pred_gridpoints_vector = [Y_mean_pred_gridpoints_vector[i]/  # divide by cross-section
##                                              ((XF_vector[i][0]/2.0)**2.0*np.pi)* # multiply by maximum number of longerons possible
##                                              np.floor( XF_vector[i][0]*np.pi/(XF_vector[i][2]*XF_vector[i][0]*2.0) ) for i in range(np.shape(XF_vector)[0])]
#        Y_std_pred_gridpoints_vector = np.zeros(np.shape(Y_mean_pred_gridpoints_vector))
#    #
#    Y_mean_pred_gridpoints_matrix = np.reshape(Y_mean_pred_gridpoints_vector,np.shape(A_matrix))
#    #
#    Y_std_pred_gridpoints_matrix = np.reshape(Y_std_pred_gridpoints_vector,np.shape(A_matrix))
#    #
#    #
##    XX1 = XX[pairidx]
##    YY1 = YY[pairidx]
##    Clas1 = Clas[pairidx]
#
##    Z_mean_pred_gridpoints_matrix = Z_mean_pred_gridpoints_matrices[0][pairidx]
#    # A simple way to quantify the uncertainty of the noisy predictions:
#    print("The mean uncertainty is %0.3e"
#          % np.mean(Y_std_pred_gridpoints_vector*2) )
#    print("The STDV of the uncertainty is %0.3e"
#          % np.std(Y_std_pred_gridpoints_vector*2) )
#    #
##    PredYF.append(Y_mean_pred_gridpoints_matrix)
#    
#
#    # Plot surface
##        ax = fig1.gca(projection='3d')
#    #
#    surf = ax.plot_surface(A_matrix, B_matrix, Y_mean_pred_gridpoints_matrix,
#                           cmap=cm.coolwarm, alpha=0.8,
#                           linewidth=0, antialiased=False)
##        fig1.colorbar(surf, shrink=0.5, aspect=5)
#    surf2 = ax.plot_surface(A_matrix, B_matrix, Y_mean_pred_gridpoints_matrix-2*Y_std_pred_gridpoints_matrix,
#    #                            cmap=cm.coolwarm,
#                            color='k',alpha=0.1,
#    #                            color=ma.colorAlpha_to_rgb('k', 0.1)[0],
#                            linewidth=0, antialiased=False)
#    #
#    surf3 = ax.plot_surface(A_matrix, B_matrix, Y_mean_pred_gridpoints_matrix+2*Y_std_pred_gridpoints_matrix,
#    #                            cmap=cm.coolwarm,
#                            color='k',alpha=0.1,
#    #                            color=ma.colorAlpha_to_rgb('k', 0.1)[0],
#                            linewidth=0, antialiased=False)
#    #
#    # Include training points
#    #    ax = fig.add_subplot(111, projection='3d')
#    #
#    #    ax.scatter(X_train[:,0], X_train[:,1], Y_mean_pred, c='r', marker='o')
#    #    ax.scatter(X_test[:,0], X_test[:,1], Y_mean_pred, c='r', marker='o')
#    #    ax.scatter(X[:,0], X[:,1]*1000, Y_mean_pred, c='r', marker='o')
#    #
#    #    ax.scatter(X_train[:,0], X_train[:,1]*1000, Y_mean_train, c='r', marker='o')
#    #    ax.scatter(X_test[:,0], X_test[:,1]*1000, Y_mean_test, c='r', marker='o')
#    #    ax.scatter(X[:,0], X[:,1]*1000, Y_mean[jInput_mean], c='r', marker='o')
##        X_to_plot = X[(X[:,0]<a_max)&(X[:,0]>a_min)&(X[:,1]<b_max)&(X[:,1]>b_min)]
##        Y_mean_to_plot = Y_mean
##        Y_mean_to_plot = Y_mean_to_plot[(X[:,0]<a_max)&(X[:,0]>a_min)&(X[:,1]<b_max)&(X[:,1]>b_min)]
##        ax.scatter(X_to_plot[:,0], X_to_plot[:,1], Y_mean_to_plot, c='r', marker='o')
#    #
#    #
#    ax.set_xlabel(feature_names[pair[0]], fontsize = 10)
#    ax.set_ylabel(feature_names[pair[1]], fontsize = 10)
#    ax.set_title(subplot_title)
#    ax.set_zlabel('Critical load [kPa]', fontsize = 10)
#    ax.set_xlim([a_min,a_max])
#    ax.set_ylim([b_min,b_max])
#    # Make square axes:
#    x0_lim,x1_lim = ax.get_xlim()
#    y0_lim,y1_lim = ax.get_ylim()
#    ax.set_aspect(abs(x1_lim-x0_lim)/abs(y1_lim-y0_lim))
#    plt.show()
#    #
#    #######################################################################
#    # Make contour
##    plt.rc('text', usetex=True)
##    plt.rc('font', family='serif', size=20)
##    if jInput_mean == 0:b
##        load_label = 'Y'
##    else:
##        load_label = 'X'
#    #
#    load_label = 'Y'
#    load_label2 = 'Z'
#    plt.figure(fig2.number)
#    ax = fig2.add_subplot(nRows_subplot, nCols_subplot, pairidx + 1)
#    ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
#    origin = 'lower'
#    CS = plt.contourf(A_matrix, B_matrix, Y_mean_pred_gridpoints_matrix, 30,
#                  #[-1, -0.1, 0, 0.1],
#                  #alpha=0.5,
#                  cmap=plt.cm.hsv,
#                  origin=origin, zorder=1)
#    #
##    DS = plt.contour(A_matrix, B_matrix, Z_mean_pred_gridpoints_matrix, 20, colors='k',  zorder=5)
##    plt.clabel(DS,inline=1, fontsize=12)
#    cbar = plt.colorbar(CS)
#    
#    c_white = matplotlib.colors.colorConverter.to_rgba('white')    
#    c_white_trans = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0.0)
#    levels = [-1,0.0,1.0,2.0]
#    cmapc = mpl.colors.ListedColormap([c_white, c_white_trans])
#    bounds = [0.5]
#    
##    ES = plt.contourf(XX1,YY1,Clas1, levels, cmap = cmapc,  zorder=10)
#    
#    
##    cbar.ax.set_ylabel(r'Mean($P^{cr}_'+load_label+'$) [kPa]')
#    
#    labels = [r'Mean($E^{U}_'+load_label+'$) [kPa Strain]', r'Mean($P^{cr}_'+load_label+'$) [kPa]']
#    legend_elements = [Line2D([0], [0], color='k', lw=3, label=labels[0]),
#                   Patch(facecolor='mediumorchid', edgecolor='k',
#                         label=labels[1])]
##    DS.collections[1].set_label(labels[0])
##    CS.collections[2].set_label(labels[1])
#    leg = plt.legend(handles=legend_elements, loc='lower right')
#    leg.set_zorder(20)
#    
#    # Add the contour line levels to the colorbar
#    #    cbar.add_lines(CS)
#    #
#    #
#    #
##    plt.title(r'Collapse Buckling Moment '+load_label)
#    plt.xlabel(feature_names[pair[0]], fontsize = 12)
#    plt.ylabel(feature_names[pair[1]], fontsize = 12)
#    plt.title(subplot_title)
##        plt.tight_layout(h_pad=1.0, w_pad=0.5, pad=2.5)
##        ax.subplots_adjust(bottom = 0)
##        ax.subplots_adjust(top = 1)
##        ax.subplots_adjust(right = 1)
##        ax.subplots_adjust(left = 0)
#    # Make square axes:
#    x0_lim,x1_lim = ax.get_xlim()
#    y0_lim,y1_lim = ax.get_ylim()
#    ax.set_aspect(abs(x1_lim-x0_lim)/abs(y1_lim-y0_lim))
#    plt.show()
#    #
#    #######################################################################
#    # Make 2nd contour
#    plt.figure(fig3.number)
#    ax = fig3.add_subplot(nRows_subplot, nCols_subplot, pairidx + 1)
#    origin = 'lower'
#    CS2 = plt.contourf(A_matrix, B_matrix, Y_std_pred_gridpoints_matrix, 30,
#                  #[-1, -0.1, 0, 0.1],
#                  #alpha=0.5,
#                  cmap=plt.cm.hsv,
#                  origin=origin)
#    #
#    cbar2 = plt.colorbar(CS2)
#    cbar2.ax.set_ylabel(r'STDV($P^{cr}_'+load_label+'$) [kPa]')
#    # Add the contour line levels to the colorbar
#    #    cbar2.add_lines(CS2)
##    plt.title(r'STD for Collapse Buckling Moment '+load_label)
#    plt.xlabel(feature_names[pair[0]], fontsize = 12)
#    plt.ylabel(feature_names[pair[1]], fontsize = 12)
#    plt.title(subplot_title)
##        plt.tight_layout(h_pad=1.0, w_pad=0.5, pad=2.5)
##        ax.subplots_adjust(bottom = 0)
##        ax.subplots_adjust(top = 1)
##        ax.subplots_adjust(right = 1)
##        ax.subplots_adjust(left = 0)
#    # Make square axes:
#    x0_lim,x1_lim = ax.get_xlim()
#    y0_lim,y1_lim = ax.get_ylim()
#    ax.set_aspect(abs(x1_lim-x0_lim)/abs(y1_lim-y0_lim))
#    plt.show()
##end for pair
#
## end of for iInput_mean
##    with open('Energy_Matrix.pickle', 'wb') as f:
##        pickle.dump([PredYF], f)    
#
## -----------------------------------------------------------------------------
#
#
## SURFACE PLOT
##fig = plt.figure()
##ax = fig.gca(projection='3d')
##X = np.arange(-5, 5, 0.25)
##Y = np.arange(-5, 5, 0.25)
##X_train_axis1, X_train_axis2 = np.meshgrid(X_train[:,0], X_train[:,1])
##R = np.sqrt(X**2 + Y**2)
##Z = np.sin(R)
##surf = ax.plot_surface(X_train[:,0], X_train[:,1], Y_train, rstride=1, cstride=1, cmap=cm.coolwarm,
##                       linewidth=0, antialiased=False)
##ax.set_zlim(-1.01, 1.01)
##
##ax.zaxis.set_major_locator(LinearLocator(10))
##ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
##
##fig.colorbar(surf, shrink=0.5, aspect=5)
##
##ax.set_xlabel('X Label')
##ax.set_ylabel('Y Label')
##ax.set_zlabel('Z Label')
##
##plt.show()
##    
end = timer()
print(end - start) # Time in seconds, e.g. 5.38091952400282
