#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:28:47 2022

@author: surya
"""

import tensorflow as tf
import pickle
import scipy.optimize
import autograd
import autograd.core
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import xarray

def res_to_dataset(losses, frames):
    ds = xarray.Dataset({
        'loss': (('step',), losses),
        'output': (('step', 'x'), frames),
    }, coords={'step': np.arange(len(losses))})
    
    return ds

def weights_to_file(model,directory,filename):
    """ 
    Pickles the trainable weights into a file 
    For use of visualization of loss landscapes
    """
    new_param = dict()
    lis_tv = model.trainable_variables #returns a list of the trainable
    #variables of the CNN model
    for i,var in enumerate(lis_tv):
        key = model.trainable_variables[i].name
        new_param[key] = var.numpy()#can convert to numpy if needed    
    file_path = directory +'/'+filename+'.p'
    pickle.dump(new_param,open(file_path,'wb'))
    #return filename+'.p'

def convert_autograd_to_tensorflow(func):#S:func is completely written in numpy autograd
    @tf.custom_gradient
    def wrapper(x):
        vjp, ans = autograd.core.make_vjp(func, x.numpy())
        def first_grad(dy):            
            @tf.custom_gradient
            def jacobian(a):
                vjp2, ans2 =  autograd.core.make_vjp(egrad(func), a.numpy())
                return ans2,vjp2 # hessian                    

            return dy* jacobian(x)  
        return ans, first_grad
    
    return wrapper

def _set_variables(variables, x):
  shapes = [v.shape.as_list() for v in variables]
  values = tf.split(x, [np.prod(s) for s in shapes])
  for var, value in zip(variables, values):
    var.assign(tf.reshape(tf.cast(value, var.dtype), var.shape))


def _get_variables(variables):
  return np.concatenate([
      v.numpy().ravel() if not isinstance(v, np.ndarray) else v.ravel()
      for v in variables])

def train_lbfgs(model, func_obj, max_iterations, path ="", 
                n_saves =1, conv_criteria = False, limit = 0.01, **kwargs):
    """

            
    """
    func = convert_autograd_to_tensorflow(func_obj.ask) #So that gradients can flow to the models 
    model(None) # Build the model
    fval = []   # To store the function values at each optimziation step   
    outs = []   #Storing teh outputs of the model (normalized!!) 
    tvars = model.trainable_variables
    
    flag =  False
    indices =[]
    if path != '': # Routine to store the model's variables to Hard disk
        filename = 'lbfgs_weights_'
        print("Filename used: ",filename)
        i=0
        while i < max_iterations:
            if i == 0:
                indices.append(i)
            i+=n_saves
            if i < max_iterations:
                indices.append(i)
        indices.append(max_iterations)
        weights_to_file(model,path,filename+str(indices[0]))
        flag =  True
    
    def value_and_grad(z): # Is called by optimzier to know 
                #the gradient of teh function w.r.t model variaobles
        _set_variables(tvars, z) # Copy the values of z onto the model's variables      
        with tf.GradientTape() as tape:
            tape.watch(tvars)
            logits = 0.0 + tf.cast(model(None), tf.float64)           
            loss = func(tf.reshape(logits, (func_obj.dim)))
        grad = tape.gradient(loss,tvars)
        # Read temp file created by lbfgsb.py
        file_s = open("./n_iterations_lbfgs.txt", 'r')
        iter_str = file_s.read()
        file_s.close()
        code_lbfgs = iter_str.split(".")[-1]
        #print(code_lbfgs)
        if len(code_lbfgs) != 1:
            pass
        else:
            fval.append(loss.numpy().copy())
            outs.append(logits.numpy()[0].copy())
            
            i = len(fval)-1
            nonlocal flag
            nonlocal indices
    
            
            if flag and i-1 in indices[1:]:
                if conv_criteria and i-1 > 10:
                    last_losses = np.array(fval[-10:])
                    per_change = -1*np.diff(last_losses)/ last_losses[:-1]*100
                    if np.all(per_change <= limit):
                        flag = False
                        indices = indices[:i]
                    weights_to_file(model,path,filename+str(i-1))
                    #truncate indices                                        
                else:
                    weights_to_file(model,path,filename+str(i-1))
                    
        return float(loss.numpy()), _get_variables(grad).astype(np.float64)
        
    x0 = _get_variables(tvars).astype(np.float64)
    
    # rely upon the step limit instead of error tolerance for finishing.
    _, _, info = scipy.optimize.fmin_l_bfgs_b(
        value_and_grad, x0, maxfun=max_iterations, factr=1, pgtol=1e-14,**kwargs)
    
    # Convert outs to xarray dataset            
    return res_to_dataset(fval, outs), indices


def train_tf_optimizer(model, func_obj, optimizer, max_iterations, path ="",
                n_saves =1, conv_criteria = False, limit = 0.01, **kwargs):
    """
    
    """
    func = convert_autograd_to_tensorflow(func_obj.ask) #So that gradients can flow to the models 
    model(None) # Build the model
    fval = []   # To store the function values at each optimziation step   
    outs = []   #Storing teh outputs of the model (normalized!!) 
    tvars = model.trainable_variables
    
    flag =  False
    indices =[]
    if path != '': # Routine to store the model's variables to Hard disk
        filename = 'lbfgs_weights_'
        print("Filename used: ",filename)
        i=0
        while i < max_iterations:
            if i == 0:
                indices.append(i)
            i+=n_saves
            if i < max_iterations:
                indices.append(i)
        indices.append(max_iterations)
        weights_to_file(model,path,filename+str(indices[0]))
        flag =  True

    for i in range(max_iterations + 1):
        with tf.GradientTape() as t:
            t.watch(tvars)
            logits = 0.0 + tf.cast(model(None), tf.float64)           
            loss = func(tf.reshape(logits, (func_obj.dim)))
    
        fval.append(loss.numpy().copy())
        outs.append(logits.numpy()[0].copy())
      #Saving weight files to disk as pickled file: Applies convergence criterion as well
        if i == 0:#already saved initialization weight file
            pass
        else:
            if flag and i in indices[1:]:
                if conv_criteria and i > 10:
                    last_losses = np.array(fval[-10:])
                    per_change = -1*np.diff(last_losses)/ last_losses[:-1]*100
                    if np.all(per_change <= limit):
                        flag = False
                        indices = indices[:i+1]
                        weights_to_file(model,path,filename+str(i))                                     
                else:
                    weights_to_file(model,path,filename+str(i)) 

        if i < max_iterations:
            grads = t.gradient(loss, tvars)
            optimizer.apply_gradients(zip(grads, tvars))
      
    return res_to_dataset(fval, outs), indices

