'''
Created on 2020-05-05 22:36:35
Last modified on 2020-05-07 10:09:52
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------

'''


#%% imports

# standard library
import os
import pickle

# third-party
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.gaussian_process.kernels import Matern


#%% initialization

var_name = 'sigma_crit'
example_name = 'example_supercompressible_3d'
pkl_filename = 'DoE_results.pkl'
train_size = .5
test_size = 1 - train_size


#%% get data

# open file and get panda dataframe
with open(os.path.join(example_name, pkl_filename), 'rb') as file:
    data = pickle.load(file)
points = data['points']

# get number of inputs
n_inputs = len(points.columns) - 3

# missing indices
indices = pd.notnull(points.loc[:, var_name]).values

# get y data
y = points.loc[indices, var_name].values

# get X data
X = points.iloc[indices, range(n_inputs)].values


#%% machine learning

# split data
indices = range(len(y))
X_train = X[indices[:-int(round(len(indices) * test_size))]]
X_test = X[indices[-int(round(len(indices) * test_size)):]]
y_mean_train = y[indices[:-int(round(len(indices) * test_size))]]
y_mean_test = y[indices[-int(round(len(indices) * test_size)):]]
n_train = len(y_mean_train)

# scale data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

# train model
my_kernel = 1.0 * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-1, 10.0))
reg = GaussianProcessRegressor(kernel=my_kernel, alpha=0.1**2, n_restarts_optimizer=0)
reg.fit(X_train_scaled, y_mean_train)

# predict test
y_mean_pred, y_std_pred = reg.predict(scaler.transform(X_test), return_std=True)

# error metrics
mse = mean_squared_error(y_mean_test, y_mean_pred)
r2 = r2_score(y_mean_test, y_mean_pred)
expl_var = explained_variance_score(y_mean_test, y_mean_pred)
print("The mean squared error is %0.3e" % mse)
print("The R2 score is %0.3f" % r2)
print("The explained variance score is %0.3f" % expl_var)

# plot
plt.figure()
plt.plot(y_mean_test, y_mean_pred, 'o')
plt.plot([np.min(y_mean_test), np.max(y_mean_test)], [np.min(y_mean_test), np.max(y_mean_test)], 'r-')
plt.title('$n_{train} = %i$, $R^{2} = %0.3f$, $mse = %0.3e$' % (n_train, r2, mse))
plt.ylabel("Predicted")
plt.xlabel("Observed")
plt.show()
