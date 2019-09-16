#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: M.A. Bessa (M.A.Bessa@tudelft.nl)
"""

# from IPython import get_ipython
# get_ipython().magic('reset -sf')

# print(__doc__)
import numpy as np
#import mimic_alpha as ma
# Matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rc
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
import matplotlib
# Preprocessing data:
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Machine learning method:
#from sklearn.svm import SVR
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
#                                              ExpSineSquared, DotProduct,
#                                              ConstantKernel)

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)
import gpflow
import GPy
import h5py
# Postprocessing metrics:
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
# Validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
#
# To save the model:
from sklearn.externals import joblib
import os
import errno
import scipy.io as sio
try:
    import cPickle as pickle  # Improve speed
except ImportError:
    import pickle
#
# Close all figures:
plt.close("all")
# Take care of some general properties of the plots:
rc('text', usetex=False)
rc('font',size=12)
#------------------------------------------------------------------------------
#Test size?
test_size=0.2

# Do you want to do a grid search to try to find good parameters?
grid_search = 0
# Do you want to load previously saved ML model
load_model=0
#Which ML method 0=SVC 1=SVGP 2=Scikit GP 3= GPy Full GP with Laplace or EP posterior approximation (select in code)
ML_method = 1
#Do you want to create new DoE matlab file with all points coilable
save_new_DoE = 0
# Parameters:
n_classes = 2
plot_colors = "rb"
n_plot_steps = 400 # resolution (number of steps in x and y)
target_names = ['Not coilable', 'Coilable']

# Metric(s) to evaluate the models
metrics = [] # initate empty variable to allow use of multiple metrics (or just one)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
metrics.append([accuracy_score,'Accuracy classification score']) # the second column is just to print the NAME of the metric
metrics.append([precision_score,'Precision classification score'])
metrics.append([recall_score,'Recall classification score'])
#metrics.append([f1_score,'F1 score']) # the second column is just to print the NAME of the metric

#------------------------------------------------------------------------------
# READ DATASET

#analysis_folder = 'AstroMat_1st_attempt'
#analysis_folder = 'AstroMat_5Params_PLA_batch1'
#analysis_folder = 'Single_AstroMat_1St_7Params_batch2'
analysis_folder = 'DOE_Single_Astromat_1St_7ParamsFirst_Imperfect_Coilable'
#analysis_folder = 'DOE_Single_Astromat_1St_7ParamsFirst_Imperfect'
#analysis_folder = 'Single_AstroMat_1St_4Params'

#training_DoE_points = 700 # Number of input points used for training

dir_path = os.path.dirname(os.path.realpath(__file__))
#
# Read DoE file
DoE_dir = '1_DoEs/'
#file_DoE_path = dir_path+'/'+DoE_dir+'DOE_PD1_St_D1_D2D1_db_update.mat'
#file_DoE_path = dir_path+'/'+DoE_dir+'DOE_Single_Astromat_1St_7Params_batch2'
#file_DoE_path = dir_path+'/'+DoE_dir+'DOE_Single_Astromat_1St_7ParamsFirst_Imperfect'
file_DoE_path = dir_path+'/'+DoE_dir+'DOE_TO_Classify_NEW'
DoE_data = sio.loadmat(file_DoE_path)
X_all = DoE_data['DoE']['points'][0][0] # All the input points
#DELETE THE IMPERFECTION COLUMN
X_all= np.delete(X_all, -1, axis=1)
# Extract the names of the features (input variables):
feature_names = []
for iFeature in range (0,len(DoE_data['DoE']['vars'][0][0][0])):
    feature_names.append(str(DoE_data['DoE']['vars'][0][0][0][iFeature][0]))
#end for
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
#Z = [[] for _ in range(np.shape(Input_points)[0])] # Create a list of lists (for Inputs and DoE points)
X = [[] for _ in range(np.shape(Input_points)[0])]
X_lost = [[] for _ in range(np.shape(Input_points)[0])]  # Create a list of lists (for Inputs and DoE points)
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


imperfection_dim = len(STRUCTURES_data["Input1"])
doe_dim = len(STRUCTURES_data['Input1']['Imperfection1'])
##############################################################################################################
#

for iInput in range(np.shape(Input_points)[0]):
    count = 0
    for kImperf in range(0,imperfection_dim):
   
            for jDoE in range(0,doe_dim):
                try:
               
                    if 'DoE'+str(jDoE+1) in STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]:
                        Y[0].append( STRUCTURES_data['Input'+str(iInput+1)]['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['coilable'][0] )
                        X[0].append(X_all[jDoE])
#                    if STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['P_p3_crit'][0] < 550.0:
#                        temp_U = max(abs(STRUCTURES_data['Input'+str(iInput+1)]['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['riks_RP_Zplus_U'][:,2]))
#                    if  max(abs(STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['riks_RP_Zplus_U'][:,2])) <= 150.0:
#                         print(jDoE+1)
#                         X_lost[iInput].append(X_all[jDoE])

#                         continue
#                    else:
#                        X[iInput].append(X_all[jDoE])
#                        Y[iInput].append(STRUCTURES_data['Input1']['Imperfection'+str(kImperf+1)]['DoE'+str(jDoE+1)]['P_p3_crit'][0] )
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
#                
Y = np.array(Y)
X = np.array(X[0])
#Y = Y[:,:20000]
#X = X[:20000]

X_lost = np.array(X_lost[0])

# Calculate the mean and standard deviation for the different loading conditions
Y_mean = Y[0]

#Y_mean = np.array(Y)

#Y_std = np.zeros(np.shape(Y_mean))
Y_std = Y_mean*0.1
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
X_all_scaled = scaler.transform(X_all)


#clf = DecisionTreeClassifier().fit(X_train_2features, Y_mean_train)    
#clf = DecisionTreeClassifier().fit(X_train_scaled, Y_mean_train)

#clf=AdaBoostClassifier().fit(X_train_scaled, Y_mean_train)
#clf=GaussianNB().fit(X_train_scaled, Y_mean_train)
#clf= GaussianProcessClassifier(1.0 * RBF(1.0)).fit(X_train_scaled, Y_mean_train)
#clf=SVC(C=100, gamma=1, cache_size=2000).fit(X_train_scaled, Y_mean_train)
if ML_method == 0:
    clf=SVC().fit(X_train_scaled, Y_mean_train)
    Y_train_predicted = clf.predict(X_train_scaled)
    Y_test_predicted = clf.predict(X_test_scaled)
    
elif ML_method == 1:
    my_kernel = gpflow.kernels.Matern52(input_dim=7, lengthscales=1.0, variance=1.0, ARD=True)
    Y_mean_train = Y_mean_train.reshape(-1,1)
#        noise = (Y_std_train/Y_mean_train) ** 2

#        lik = gpflow.likelihoods.Gaussian(variance=0.01)
    lik = gpflow.likelihoods.Bernoulli()
#        lik = gpflow.likelihoods.StudentT(scale=2.0)
#        lik.trainable = False
    
#        m = gpflow.models.GPR(X_train_scaled, Y_mean_train, kern=my_kernel)
    m = gpflow.models.SVGP(X_train_scaled, Y_mean_train, kern=my_kernel,likelihood=lik, whiten=False, Z=X_train_scaled[0:400].copy())
#    m = gpflow.models.SVGP(X_train_scaled, Y_mean_train, kern=my_kernel,likelihood=lik, whiten=False, Z=X_train_scaled[0:800].copy())
#    m = gpflow.models.GPRFITC(X_train_scaled, Y_mean_train, kern=my_kernel, Z=X_train_scaled[0:1200].copy())
#        m = gpflow.models.SVGP(X_train_scaled, Y_mean_train, kern=my_kernel, likelihood=lik, Z=X_train_scaled[0:1200].copy())
#        m.feature.set_trainable(False)
    print(m)
    if load_model == 0: # Create model and save it
    #   # Initially fix the hyperparameters.
        m.feature.set_trainable(False)
        gpflow.train.ScipyOptimizer().minimize(m, maxiter=20)
    
        # Unfix the hyperparameters.
        m.feature.set_trainable(True)
    #    gpflow.train.ScipyOptimizer(options=dict(maxiter=200)).minimize(m)
        gpflow.train.ScipyOptimizer().minimize(m, maxiter=1000)
        
        saver= gpflow.saver.Saver()
        saver.save('./Classification_noisyV2', m)
        m.read_trainables()['SVGP/kern/variance']
    if load_model == 1: # Create model and save it 
        m =  gpflow.saver.Saver().load('./Classification_noisyV2')
        m.read_trainables()['SVGP/kern/variance']
        print(m)
    
    Y_train_predicted = m.predict_y(X_train_scaled)[0]
    Y_test_predicted = m.predict_y(X_test_scaled)[0]
    Y_train_predicted = np.rint(Y_train_predicted)
    Y_test_predicted = np.rint(Y_test_predicted)
#    Coil_DoE_data = m.predict_y(X_all_scaled)[0]
#    Coil_DoE_data = np.rint(Coil_DoE_data)       

elif ML_method == 2:
#     onstantKernel(1.0) * RBF(length_scale=1.0)
        my_kernel = 1.0 * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-1, 10.0))
    #        my_kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
        #
        # !!!!!!!! WITHOUT regularization (no noise!):
#        reg = GaussianProcessRegressor(kernel=my_kernel, random_state=0, n_restarts_optimizer=10)
#        reg = GaussianProcessRegressor(kernel=my_kernel, alpha=0.1**2)
        # !!!!!!!! WITH REGULARIZATION (noisy data!):
        # -> Tikhonov regularization:
#        reg = GaussianProcessRegressor(kernel=my_kernel, alpha=(Y_std_train/Y_mean_train) ** 2, n_restarts_optimizer=0)
        reg = GaussianProcessClassifier(kernel=my_kernel, n_restarts_optimizer=1)
        # -> Just considering the STDV:
    #        reg = GaussianProcessRegressor(kernel=my_kernel, alpha=(Y_std_train) ** 2, random_state=0, n_restarts_optimizer=10)
        #
        
        reg.fit(X_train_scaled , Y_mean_train)
        print(reg.kernel_)
    #    Y_pred, Y_std = reg.predict(scaler.transform(X_test), return_std=True)
        #X_test_neww = X_test[:,0].reshape(-1,1)
        Y_train_predicted = reg.predict(scaler.transform(X_train_scaled))
        Y_test_predicted = reg.predict(scaler.transform(X_test_scaled))
#        reg.scaler = scaler # Save also the scaler
#        dlugosc = reg.kernel_.k2.get_params()['length_scale']
#        sigma_f = np.sqrt(reg.kernel_.k1.get_params()['constant_value'])
        #
elif ML_method == 3:
    my_kernel = GPy.kern.Matern52(input_dim=7, variance=1.0, lengthscale=1.0,  ARD=True)
    Y_mean_train = Y_mean_train.reshape(-1,1)
    inf = GPy.inference.latent_function_inference.Laplace()
    lik = GPy.likelihoods.Bernoulli()
    

#    m = GPy.models.GPClassification(X_train_scaled, Y_mean_train, kernel=my_kernel)
    m = GPy.core.GP(X_train_scaled, Y_mean_train, kernel=my_kernel, likelihood=lik,inference_method=inf)
#    m = gpflow.models.GPRFITC(X_train_scaled, Y_mean_train, kern=my_kernel, Z=X_train_scaled[0:1200].copy())
#        m = gpflow.models.SVGP(X_train_scaled, Y_mean_train, kern=my_kernel, likelihood=lik, Z=X_train_scaled[0:1200].copy())
#        m.feature.set_trainable(False)
    print(m)
#   # Initially fix the hyperparameters.
    m.optimize(messages=1)
    print(m)
#    m.feature.set_trainable(False)
#    gpflow.train.ScipyOptimizer().minimize(m, maxiter=20)
#
#    # Unfix the hyperparameters.
#    m.feature.set_trainable(True)
##    gpflow.train.ScipyOptimizer(options=dict(maxiter=200)).minimize(m)
#    gpflow.train.ScipyOptimizer().minimize(m, maxiter=1000)        
    
    Y_train_predicted = m.predict(X_train_scaled)[0]
    Y_test_predicted = m.predict(X_test_scaled)[0]
    Y_train_predicted = np.rint(Y_train_predicted)
    Y_test_predicted = np.rint(Y_test_predicted)
    

for Imetric in range(np.shape(metrics)[0]):
    metric = metrics[Imetric][0] # select metric used
    print("Train", metrics[Imetric][1],"for ML model:", metric(Y_mean_train, Y_train_predicted))
    print("Test", metrics[Imetric][1],"for ML model:", metric(Y_mean_test, Y_test_predicted))
#    print("Train %s for ML model: %0.5f"
#          % (metrics[Imetric][1], metric(Y_mean_train, Y_train_predicted)) )
#    print("Test %s for ML model: %0.5f"
#          % (metrics[Imetric][1], metric(Y_mean_test, Y_test_predicted)) )

# For multi-label classification some metrics need the "average" option to be changed from the default 'binary'
print("Train F1 score per class for ML model:", f1_score(Y_mean_train, Y_train_predicted,average=None))
print("Test F1 score per class for ML model:", f1_score(Y_mean_test, Y_test_predicted,average=None))
print("Train Confusion Matrix for ML model:", confusion_matrix(Y_mean_train, Y_train_predicted))
print("Test Confusion Matrix for ML model:", confusion_matrix(Y_mean_test, Y_test_predicted))
print("Train Classification report for ML model:", classification_report(Y_mean_train, Y_train_predicted))
print("Test Classification report for ML model:", classification_report(Y_mean_test, Y_test_predicted))
#print("Train Precision score with macro-averaging for ML model:", precision_score(Y_mean_train, Y_train_predicted,average='macro'))
#print("Test Precision score with macro-averaging for ML model:", precision_score(Y_mean_test, Y_test_predicted,average='macro'))

#file_DoE2_path = dir_path+'/'+DoE_dir+'DOE_TO_CLASSIFY'
##file_DoE_path = dir_path+'/'+DoE_dir+'DOE_Single_Astromat_1St_4Params.mat'
#DoE_data2 = sio.loadmat(file_DoE2_path)
#X_all2 = DoE_data2['DoE']['points'][0][0] # All the input point
#Imperfect = X_all2 [:-1]
#X_all2 = np.delete(X_all2, -1, axis=1)
#X_all2_scaled = scaler.transform(X_all2)
#
#Coil_DoE_data = m.predict_y(X_all2_scaled)[0]
#Coil_DoE_data = np.rint(Coil_DoE_data)  
#
#if save_new_DoE == 1:
#    Coil_DoE_data
#    X_all_new = np.concatenate((X_all,Coil_DoE_data), axis=1)
#    #np.where(~X_all_new.any(axis=1))[0]
#    ##data= np.delete(data,np.where(~data.any(axis=1))[0], axis=0)
#    X_all_new = X_all_new[np.all(X_all_new[:,7].reshape(-1,1) != 0, axis=1)]
#    X_all_new = np.delete(X_all_new, -1, axis=1)
#    
#    DoE_data['DoE']['points'][0][0] = X_all_new 
#    
#    file_DoE_path_new = dir_path+'/'+DoE_dir+'DOE_Single_Astromat_1St_7Params_batch4_new'
#    ##file_DoE_path = dir_path+'/'+DoE_dir+'DOE_Single_Astromat_1St_4Params.mat'
#    #
#    sio.savemat(file_DoE_path_new, DoE_data)


#with h5py.File('X_all_new_noiseless.h5', 'w') as hf:
#    hf.create_dataset("X_all_new_noiseless",  data=X_all_new)

#     #         0    ,           1       ,       2        ,           3       ,        4         ,         5        ,             6
##    '$\frac{A}{D_1}$' ,'$\frac{G}{E}$' ,'$\frac{Ix}{D1}$' ,$\frac{Iy}{D1}$'  ,$\frac{J}{D1}$'  ,'$\frac{P}{D_1}$' ,'$\frac{D_1-D_2}{D_1}$',

#features_to_plot = [[2, 4  , 5E-5, 0.3937, 5E-8, 1.49, 0.5],
#                    [2, 4  , 5E-5, 0.3937, 8E-8, 1.49, 0.5],
#                    [2, 4  , 5E-5, 0.3937, 4E-7, 1.49, 0.5],
#                    [2, 4  , 5E-5, 0.3937, 9E-7, 1.49, 0.5]]
#features_to_plot = [[3, 4  , 5E-5, 0.3937, 2.1E-10, 1.49, 0.5],
#                    [3, 4  , 5E-5, 0.3937, 1E-6, 1.49, 0.5],
#                    [3, 4  , 5E-5, 0.3937, 1.2E-6, 1.49, 0.5],
#                    [3, 4  , 5E-5, 0.3937, 1.397E-6, 1.49, 0.5]]
#features_to_plot = [[2, 3  , 5E-5, 0.3937, 2.6E-10, 1.49, 0.5],
#                    [2, 3  , 5E-5, 0.3937, 1.4E-8, 1.49, 0.5],
#                    [2, 3  , 5E-5, 0.3937, 1.4E-7, 1.49, 0.5],
#                    [2, 3  , 5E-5, 0.3937, 6.76E-6, 1.49, 0.5]]
#features_to_plot = [[2, 3  , 5E-5, 0.3937, 1.3E-10, 1.49, 0.5],
#                    [2, 3  , 5E-5, 0.3937, 5E-7, 1.49, 0.5],
#                    [2, 3  , 5E-5, 0.3937, 1.0E-6, 1.49, 0.5],
#                    [2, 3  , 5E-5, 0.3937, 6.0E-6, 1.49, 0.5]]

features_to_plot = [[3,4, 3.8E-3, 0.43, 1.2E-6, 0.7, 0.0],
                    [3,4, 3.8E-3, 0.43, 1.2E-6, 0.7, 0.25],
                    [3,4, 3.8E-3, 0.43, 1.2E-6, 0.7, 0.5],
                    [3,4, 3.8E-3, 0.43, 1.2E-6, 0.7, 0.78]]
#     #         0    ,           1       ,       2        ,           3       ,        4         ,         5        ,             6
##    '$\frac{A}{D_1}$' ,'$\frac{G}{E}$' ,'$\frac{Ix}{D1}$' ,$\frac{Ix}{D1}$'  ,$\frac{J}{D1}$'  ,'$\frac{P}{D_1}$' ,'$\frac{D_1-D_2}{D_1}$',
#features_to_plot = [[0, 5 , 0.3937, 4.908739E-10, 4.908739E-10, 9.817477E-10, 0.0],
#                    [0, 5 , 0.3937, 2.485049E-9, 2.485049E-9, 4.970098E-9, 0.25],
#                    [0, 5 , 0.3937, 7.853982E-9, 7.853982E-9, 1.570796E-8, 0.5],
#                    [0, 5 , 0.3937, 3.976078E-8, 3.976078E-8, 7.952156E-8, 0.75]]
#features_to_plot = [[1, 2 , 200.0, 0.0, 1.0], [1, 2 , 200.0, 0.0, 5.0], [1, 2 , 200.0, 0.0, 10.0],

XX = []
YY = []
PredYF= []

#                    [1, 2 , 200.0, 0.0, 15.0]]

for pairidx, pair in enumerate(features_to_plot):
    #
#    X_train_2features = X_train[:,pair]
    
    # Plot the decision boundary
    nRows_subplot = np.ceil(np.sqrt(len(features_to_plot)))
    nCols_subplot = np.ceil(np.double(len(features_to_plot))/nRows_subplot)
    plt.subplot(nRows_subplot, nCols_subplot, pairidx + 1)
    
#    x_min, x_max = X_train_2features[:, 0].min() - 1, X_train_2features[:, 0].max() + 1
#    y_min, y_max = X_train_2features[:, 1].min() - 1, X_train_2features[:, 1].max() + 1
    
    x_min, x_max = X_train[:, pair[0]].min() , X_train[:, pair[0]].max() 
    y_min, y_max = X_train[:, pair[1]].min() , X_train[:, pair[1]].max()
    plot_step_x = (x_max-x_min)/n_plot_steps
    plot_step_y = (y_max-y_min)/n_plot_steps
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step_x),
                         np.arange(y_min, y_max, plot_step_y))
#    plt.tight_layout(h_pad=3.0, w_pad=0.5, pad=2.5)
    #
    X_plot = np.ones((len(xx.ravel()),np.shape(X)[1]))
    #
    subplot_title = ' '
    counter = 0
    for iFeature in range(0,np.shape(X)[1]):
        if iFeature == pair[0]:
            X_plot[:,iFeature] = xx.ravel()
        elif iFeature == pair[1]:
            X_plot[:,iFeature] = yy.ravel()
        else:
#            X_plot[:,iFeature] = X_train[np.shape(X_train)[0]/2,iFeature]*X_plot[:,iFeature]
            counter = counter + 1
            X_plot[:,iFeature] = pair[1+counter]*X_plot[:,iFeature]
#            X_plot[:,iFeature] = pair[iFeature]*X_plot[:,iFeature]
#            counter = counter + 1
            subplot_title = subplot_title + str(feature_names[iFeature]) +'='+str(pair[1+counter])
            if counter < np.shape(X)[1]-2: # then write a comma
                subplot_title = subplot_title +', '
            #end if
            
        #end if
#    Z = clf.predict(np.c_[xx.ravel(), yy.ravel() , ])
    if ML_method==0:    
        Z = clf.predict(scaler.transform(X_plot))
    if ML_method==1:    
        Z = m.predict_y(scaler.transform(X_plot))[0]
        Z = np.rint(Z)
    Z = Z.reshape(xx.shape)
    
    XX.append(xx)
    YY.append(yy)

    PredYF.append(Z)
#    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    levels = [-1,0.0,1.0,2.0]
    c_green = matplotlib.colors.colorConverter.to_rgba('green')
    c_green_trans = matplotlib.colors.colorConverter.to_rgba('green',alpha = 0.5)
    cmapc = mpl.colors.ListedColormap(['white', c_green_trans])
    bounds = [0.5]
#    cmapc.set_over('0.25')
#    cmapc.set_under('0.75')
    cs = plt.contourf(xx, yy, Z,levels , cmap=cmapc)#plt.cm.RdYlBu)
#    cs = plt.contourf(xx, yy, Z, [0.0,1.0,2.0], cmap=plt.cm.RdYlBu)
    #
    plt.xlabel(feature_names[pair[0]])
    plt.ylabel(feature_names[pair[1]])
    plt.title(subplot_title)
    artists, labels = cs.legend_elements()
#    plt.legend(artists[0], labels[0], handleheight=2)
    plt.legend([artists[0],artists[1]],['Not Coilable','Coilable'])
    #
    # Save the model to disk
    input_machinelearn_dir_path = machinelearn_dir_path+'Input'+str(iInput+1)+'/'
    # If the folder doesn't exist, then create it:
    try:
        os.makedirs(input_machinelearn_dir_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    #
##    file_machinelearn_path = input_machinelearn_dir_path+'DecisionTreeClassifier_model.sav'
#    file_machinelearn_path = input_machinelearn_dir_path+'SVC_model.sav'
#    joblib.dump(clf, file_machinelearn_path)
    
    # Plot the training points
#    Y_mean_test = clf.predict(X_test)
#    for i, color in zip(range(n_classes), plot_colors):
#        idx = np.where(Y_mean_test == i)
#        plt.scatter(X_test[idx, pair[0]], X_test[idx, pair[1]], c=color, label=target_names[i],
#                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
    #
#end if

plt.suptitle("Super-compressibility classification")

#with open('Classification.pickle', 'wb') as f:
#        pickle.dump([XX,YY,PredYF], f)    

#plt.legend(loc='lower right', borderpad=0, handletextpad=0)
#plt.axis("tight")
plt.show()

# If you want to plot particular points:
#plt.subplot(nRows_subplot, nCols_subplot, 2)
#plt.plot([1.5/40],[0.44],'kx' , [2.0/40],[0.44],'ko')

### If you want to know the prediction for the points that coil but don't fail:
#unique, counts = np.unique(clf.predict(scaler.transform(X_temp)), return_counts=True)
#print(dict(zip(unique, counts)))


