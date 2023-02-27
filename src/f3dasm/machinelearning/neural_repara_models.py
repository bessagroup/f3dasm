#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:40:18 2023

@author: surya
"""
#%%
from f3dasm.machinelearning.model import Model
from typing import List
import tensorflow as tf
import autograd.numpy as np

def set_random_seed(seed):
  if seed is not None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def tensor_list_to_vector(tensor_list: List[tf.Tensor]) -> tf.Tensor:
  """Convert a list of tensors into a single (vertical) tensor.

  Useful to convert a model's `trainable_parameters` (list of tf.Variables) into
  a single vector (more convenient for Lanczos tridiagonalization).

  Args:
    tensor_list: List of tensorflow tensors to convert to a single vector.

  Returns:
    A vector obtained by concatenation and reshaping of the original tensors.
      The shape of the vector will be [w_dim, 1] where w_dim is the total number
      of scalars in the list of tensors.
  """
  return tf.concat([tf.reshape(p, [-1, 1]) for p in tensor_list], axis=0)


def vector_to_tensor_list(vector: tf.Tensor,
                          structure: List[tf.Tensor]) -> List[tf.Tensor]:
  """Inverse of `tensor_list_to_vector`.

  Convert a (vertical) vector into a list of tensors, following the shapes of
  the tensors in `structure`. For instance:

  ```
  model_weights = model.trainable_variables
  weights_as_vector = tensor_list_to_vector(model_weights)
  reshaped_weights = vector_to_tensor_list(
    weights_as_vector, structure=model_weights)
  assertShapes(model_weights, reshaped_weights)
  ```

  Args:
    vector: A tensor of shape [w_dim, 1], where w_dim is the total number
      of scalars in `structure`.
    structure: List of tensors defining the shapes of the tensors in the
      returned list. The actual values of the tensors in `structure` don't
      matter.

  Returns:
    A list of tensors of the same shape as the tensors in `structure`

  Raises:
    InvalidArgumentError: If the number of scalars in `vector` doesn't match the
      number of scalars in `structure`.
  """
  current_index = 0
  reshaped_tensors = []
  for example_tensor in structure:
    required_size = np.prod(example_tensor.shape)
    sliced = vector[current_index: current_index + required_size]
    reshaped_tensors.append(0.0 + tf.cast(tf.reshape(sliced, example_tensor.shape),tf.float32))
    #S:casting into float32 to be compatible with NN models in CNN
    current_index += required_size
  return reshaped_tensors    
    
class TfModel(tf.keras.Model):
    """Base class for Keras"""
    def __init__(self, seed=None):
      super().__init__()
      set_random_seed(seed)
      self.seed = seed

    
    
class GenericModel(TfModel, Model):
    """Base class for all machine learning models"""
    
    def __init__(self, seed=None, shape = None):
        """ Model with trainable weights in a given shape
        shape : Tuple
        """
        super().__init__(seed)
        z_init = tf.random.uniform(shape, minval = 0, maxval = 1.0)
        self.z = tf.Variable(z_init, trainable=True, dtype = tf.float32) 
 
    def call(self, inputs=None): 
        return self.forward(self.z)

    def forward(self, X):
        """Forward pass of the model: calculate an output by giving it an input

        Parameters
        ----------
        X
            Input of the model
        """
        return self.z

    def get_model_weights(self):
        """Retrieve the model weights as a 1D array"""
        return tensor_list_to_vector(self.get_weights()).numpy()

    def set_model_weights(self, weights: np.ndarray):
        reshaped_weights =  vector_to_tensor_list(weights,
                                  self.get_weights())
        self.set_weights(reshaped_weights)


#%%

# ######## for later ##################
# class FCNN_simple(model, TfModel):
# # Sigmoid for last layer, non trainable latent input with size =1!!
#   def __init__(
#       self,
#       seed=0,
#       args=None,      
      
#       depth = 2, # Number of layers
#       width = 10, # Width of each layer
#       kernel_init = tf.keras.initializers.GlorotNormal,
#       activation= tf.nn.leaky_relu,
#       bias_val = 3,
      
#       latent_scale=1.0, # Random normal with std_dev =  scale
#       latent_size=2,
#       latent_trainable = True,
#   ):
#     super().__init__(seed, args)

#     n_output = args['dim']  
#     if type(width) is not tuple:
#         width = [width for x in range(depth)]
#     net = inputs = layers.Input((latent_size,), batch_size=1)  
#     bias_init = tf.keras.initializers.RandomUniform(minval= -bias_val, maxval= bias_val)
#                                     # -1 to 1 is concentrated in middle    
#     for i in range(depth):
#         if activation is tf.nn.leaky_relu:
#             net = layers.Dense(width[i], kernel_initializer=kernel_init, activation =None)(net)
#             net = tf.nn.leaky_relu(net, alpha =0.01)
#         else:
#             net = layers.Dense(width[i], kernel_initializer=kernel_init, activation =activation)(net)

#     net = layers.Dense(n_output, kernel_initializer=kernel_init ,activation =
#                        tf.keras.activations.sigmoid, bias_initializer = bias_init)(net)
    
#     outputs = layers.Reshape([n_output])(net)
#     self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)

#     latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale)
#     self.z = self.add_weight(
#                     shape=inputs.shape, initializer=latent_initializer,
#                     name='z', trainable= latent_trainable)

#   def call(self, inputs=None):
#     return self.core_model(self.z)


