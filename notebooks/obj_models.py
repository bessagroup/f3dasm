#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:39:34 2022

@author: surya
"""
import autograd.numpy as np
import tensorflow as tf

layers = tf.keras.layers

def set_random_seed(seed):
  if seed is not None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
######################### Main Model Class
class Model(tf.keras.Model):

  def __init__(self, seed=None, args=None):
    super().__init__()
    set_random_seed(seed)
    self.seed = seed
    self.env = args

######################### Pixel Model Class
class PixelModel(Model):
  """
    The class for performing optimization in the input space of the functions.
    The initial parameters are chosen uniformly from [0,1] so as to ensure
        similarity across all functions
    TODO: May need to add the functionality to output denormalized bounds
  """
  def __init__(self, seed=None, args = None):
    super().__init__(seed)
    z_init = tf.random.uniform((1,args['dim']), minval = 0, maxval = 1.0)
    self.z = tf.Variable(z_init, trainable=True, dtype = tf.float32)#S:ADDED 

  def call(self, inputs=None):
    return self.z


def global_normalization(inputs, epsilon=1e-6):
  mean, variance = tf.nn.moments(inputs, axes=list(range(len(inputs.shape))))
  net = inputs
  net -= mean
  net *= tf.math.rsqrt(variance + epsilon)
  return net

######################### Simple FC architecture
class FCNN_simple(Model):
# Sigmoid for last layer, non trainable latent input with size =1!!
  def __init__(
      self,
      seed=0,
      args=None,      
      
      depth = 2, # Number of layers
      width = 10, # Width of each layer
      kernel_init = tf.keras.initializers.GlorotNormal,
      activation= tf.nn.leaky_relu,
      bias_val = 3,
      
      latent_scale=1.0, # Random normal with std_dev =  scale
      latent_size=2,
      latent_trainable = True,
  ):
    super().__init__(seed, args)

    n_output = args['dim']  
    if type(width) is not tuple:
        width = [width for x in range(depth)]
    net = inputs = layers.Input((latent_size,), batch_size=1)  
    bias_init = tf.keras.initializers.RandomUniform(minval= -bias_val, maxval= bias_val)
                                    # -1 to 1 is concentrated in middle    
    for i in range(depth):
        if activation is tf.nn.leaky_relu:
            net = layers.Dense(width[i], kernel_initializer=kernel_init, activation =None)(net)
            net = tf.nn.leaky_relu(net, alpha =0.01)
        else:
            net = layers.Dense(width[i], kernel_initializer=kernel_init, activation =activation)(net)

    net = layers.Dense(n_output, kernel_initializer=kernel_init ,activation =
                       tf.keras.activations.sigmoid, bias_initializer = bias_init)(net)
    
    outputs = layers.Reshape([n_output])(net)
    self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale)
    self.z = self.add_weight(
                    shape=inputs.shape, initializer=latent_initializer,
                    name='z', trainable= latent_trainable)

  def call(self, inputs=None):
    return self.core_model(self.z)

####### F-tounn model class for optimziing benchmark functions
class FtounnModel_bf(Model):
  
    def __init__(self, seed=0, args=None, depth=1, 
                 width=20, n_ffeatures =150, 
                 kernel_init = tf.keras.initializers.GlorotNormal,
                 rmin=0.1, rmax=50, bias_val = 2, activation =  tf.nn.leaky_relu):
        
      super().__init__(seed, args)
      self.seed = seed
      self.args = args
      outputDim = args['dim']
      # h = args['gridsize']
      if type(width) is not tuple:
          width = [width for x in range(depth)]
      inputDim = 2*n_ffeatures # (cos + sine) x n_fourier_features         
      # hw = h*h
      net = inputs = layers.Input((inputDim,), batch_size=1)
      bias_init = tf.keras.initializers.RandomUniform(minval= -bias_val, maxval= bias_val)
      
      for i in range(depth):
          if activation is tf.nn.leaky_relu:
              l = layers.Dense(units=width[i],
                               kernel_initializer=kernel_init, 
                               bias_initializer='zeros', activation =None)     
              net = l(net)
              #net = layers.BatchNormalization(momentum=0.01)(net, training=True)
              net = tf.nn.leaky_relu(net, alpha =0.01) #to be consistent with PyTorch
          else:
              net =layers.Dense(units=width[i],
                               kernel_initializer=kernel_init, 
                               bias_initializer='zeros', activation =activation)(net) 
      l = layers.Dense(units=outputDim, kernel_initializer=kernel_init, 
                           bias_initializer = bias_init,
                           activation =tf.keras.activations.sigmoid)       
      net = l(net)
      output = layers.Reshape([outputDim])(net)#tf.transpose(tf.reshape(net, [1,h,h]), perm=[0,2,1])
      self.core_model = tf.keras.Model(inputs=inputs, outputs=output)
      #self.xy = self.generatePoints(h)

      coordnMap = np.zeros((2, n_ffeatures)) # random fourier features
      for i in range(coordnMap.shape[0]):
        for j in range(coordnMap.shape[1]):
            coordnMap[i,j] = np.random.choice([-1.,1.])*\
                        np.random.uniform(1./(2*rmax), 1./(2*rmin))
      self.coordnMap = tf.constant(coordnMap)
      mid_domain = tf.constant([[0.5,0.5]], dtype =tf.float64)
      self.z = self.applyFourierMapping(mid_domain)#self.xy) # inputs
          
    def call(self, inputs=None):
      return self.core_model(self.z)

    def generatePoints(self, nx):
        # generate points in centre of grid- domain is from 0 to 1 [bottom left to bottom right to top]
      ctr = 0
      xy = np.zeros((nx*nx,2))
      step = 1/nx
      for j in range(nx):
          for i in range(nx):
              xy[ctr,0] = (step/2 + i*step)
              xy[ctr,1] = (step/2 + j*step)
              ctr +=1
      return xy  

    def applyFourierMapping(self, x):
      c = tf.cos(2*np.pi*tf.matmul(x,self.coordnMap)) # (hw, n_features)
      s = tf.sin(2*np.pi*tf.matmul(x,self.coordnMap))
      xv = tf.concat((c,s), axis = 1) # (hw,2*n_features)
      return xv
  
############ FNO model ############ 
class SpectralConv2d(layers.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2, scale=1):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        from https://github.com/zongyi-li/fourier_neural_operator   
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels)) * scale
        
        shape = in_channels, out_channels, self.modes1, self.modes2
        a1 = np.random.uniform(size = shape)
        b1 = np.random.uniform(size = shape)
        self.r1 = tf.Variable(initial_value =a1, trainable = True,
                                     dtype = tf.float32)
        self.im1 = tf.Variable(initial_value =b1, trainable = True,
                                     dtype = tf.float32)    
        rand= np.random.uniform()
        self.r2 = tf.Variable(initial_value =rand*a1, trainable = True,
                                     dtype = tf.float32)
        self.im2 = tf.Variable(initial_value =rand*b1, trainable = True,
                                     dtype = tf.float32)    

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return tf.einsum("bixy,ioxy->boxy", input, weights)

    def call(self, x):        
        batchsize = x.shape[0]
        x_ft = tf.signal.rfft2d(x)
        # ITEM ASSIGNMENT LOGIC 
        original = tf.zeros(shape = (batchsize, self.out_channels,  x.shape[-2], 
                             x.shape[-1]//2 + 1), dtype=tf.complex64)
        
        mask1_np = np.ones(shape = original.get_shape().as_list())#original.numpy().shape) 
        mask1_np[:,:, :self.modes1, :self.modes2] = 0.0
        
        mask2_np = np.ones(shape = original.get_shape().as_list()) 
        mask2_np[:,:, -self.modes1:, :self.modes2] = 0.0
        
        mask1 = tf.convert_to_tensor(mask1_np, dtype =tf.complex64)
        mask2 = tf.convert_to_tensor(mask2_np, dtype =tf.complex64)
        # Form the complex weights to be multiplied
        weights1 = self.scale*tf.dtypes.complex(self.r1, self.im1)
        other1 = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], 
                                  weights1)
        weights2 = self.scale*tf.dtypes.complex(self.r2, self.im2)
        other2 = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], 
                                  weights2)
        padding1 = tf.constant([[0,0],
                               [0,0], 
                               [0,mask1_np.shape[-2] - other1.shape[-2]],
                               [0,mask1_np.shape[-1] - other1.shape[-1]]])
        padding2 = tf.constant([[0,0],
                               [0,0], 
                               [mask2_np.shape[-2] - other2.shape[-2],0],
                               [0,mask2_np.shape[-1] - other2.shape[-1]]])
        
        itm_assign1 = original * mask1 + tf.pad(other1, padding1) * (1 - mask1)
        out_ft = itm_assign1 * mask2 + tf.pad(other2, padding2) * (1-mask2)
        #Return to physical space
        x = tf.signal.irfft2d(out_ft, fft_length =x.shape[2:])
        return x
############################## FNOModel ###################    
def Conv2D(filters, kernel_size, **kwargs):
  return layers.Conv2D(filters, kernel_size, padding='same', **kwargs)
    
class FNOModel(Model):
    def __init__(self,
        seed=0,
        args=None,
        
        modes1=12,#12 k_max
        modes2=12, 
        width=32,#32 d_v
        num_fns=4, 
        activation="relu",
        dense_k_init = tf.keras.initializers.VarianceScaling(
                    distribution ='uniform', scale = 1/3),
        bias_val = 3.5,
        
        padding=25,
        linear=True,# True
        scale=1,
        latent_scale = 1.0) :
        
        super().__init__(seed, args)
        l_activation = layers.Activation(activation)
        dense_b_init =dense_k_init
        
        #Make the model
        #Make the input as dummy tensor of dims (1,w,h)
        outputDim = args['dim']
        h = 1#args['nely'] 
        w = 2#args['nelx']
        #Make a grid of (x,y) values in the domain w x h
        shape =(1,w,h)
        grid = self.get_grid(shape)        
        #Layer 0 - input layer
        inputs = net = layers.Input(shape = grid.shape[1:], batch_size=1)
        #Layer 1 - FC layer - Lifting to higher dimensions         
        net = layers.Dense(width, kernel_initializer=dense_k_init,
                           bias_initializer = dense_b_init,
                           activation ='linear')(net)
        net = layers.Permute((3, 1, 2))(net) # Channel is index 0
        net = layers.ZeroPadding2D( ((0,padding),(0,padding)), 
                           data_format = 'channels_first')(net)  
        bias_init = tf.keras.initializers.RandomUniform(minval= -bias_val, maxval= bias_val)
        # For the spectral conv operations
        for i in range(num_fns):
            #Apply spectral conv layer
            net1 = SpectralConv2d(width, width, 
                                 modes1, modes2, scale=scale)(net)
            #Apply linear transformation
            #Same init scheme as menioned in Pytorch (except for the factor)
            if linear: 
                # For CPU ops -- need restrucring
                net2 = layers.Permute((2,3,1))(net)
                net2 = Conv2D(width, 1,
                            kernel_initializer=dense_k_init,
                            bias_initializer = dense_b_init,
                            activation = 'linear', 
                            data_format = 'channels_last')(net2)   
                net2 = layers.Permute((3,1,2))(net2)
                net = net1 + net2 
            else:
                net = net1
                
            if i != num_fns-1:
                net = l_activation(net)
        #Extract logits by removing padding
        net = net[:,:,:-padding,:-padding]
        net = layers.Permute((2, 3, 1))(net)      
        # Extra adjustment - for a uniform init
        net = layers.Reshape([h*w*width])(net)
        # Final FC layers
        net = layers.Dense(128, kernel_initializer=dense_k_init,
                           bias_initializer = dense_b_init,
                           activation =activation)(net)
        kernel_init = tf.keras.initializers.GlorotNormal # To have a uniform init
        net = layers.Dense(2, kernel_initializer=kernel_init,
                           bias_initializer = bias_init,
                           activation = 'linear')(net)
        outputs = layers.Activation('sigmoid')(net)
        
        #net = tf.squeeze(net, axis=[-1]) 
        #outputs = layers.Reshape([outputDim])(net)
        self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale, seed = seed)
        self.z = self.add_weight(
            shape=inputs.shape, initializer=latent_initializer, trainable= False, name='z')
        
    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = tf.constant(np.linspace(0, 1, size_x), dtype=tf.float64)
        gridx = tf.tile(tf.reshape(gridx,(1, size_x, 1, 1)), [batchsize, 1, size_y, 1])
        gridy = tf.constant(np.linspace(0, 1, size_y), dtype=tf.float64)
        gridy = tf.tile(tf.reshape(gridy,(1, 1, size_y, 1)), [batchsize, size_x, 1, 1])
        return tf.concat((gridx, gridy), axis=-1)        
        
    def call(self, inputs=None):
      return self.core_model(self.z)

