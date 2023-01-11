#                                                                       Modules
# =============================================================================

# Standard
from functools import partial
from typing import List, Protocol, Tuple

# Third-party
import numpy as np
import tensorflow as tf

from f3dasm.base.data import Data

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling', 'https://d2l.ai/']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================

def get_reshaped_array_from_list_of_arrays(flat_array: np.ndarray, list_of_arrays: List[np.ndarray]) -> List[np.ndarray]:
    total_array = []
    index = 0
    for mimic_array in list_of_arrays:
        number_of_values = np.product(mimic_array.shape)
        current_array = np.array(flat_array[index:index+number_of_values])

        if number_of_values > 1:
            current_array = current_array.reshape(-1, 1)  # Make 2D array

        total_array.append(current_array)
        index += number_of_values

    return total_array

def get_flat_array_from_list_of_arrays(list_of_arrays: List[np.ndarray]) -> List[np.ndarray]: # technically not a np array input!
    return np.concatenate([np.atleast_2d(array) for array in list_of_arrays])

class Model(Protocol):
    def forward(self, X):
        ...
    
    def get_model_weights(self):
        ...

    def set_model_weights(self):
        ...

class TensorflowModel(tf.keras.Model, Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.Sequential()

    def forward(self, X):
        assert hasattr(self, 'model'), 'model is defined'
        return self.model(X)

    def call(self, X, *args, **kwargs):  # Shape: (samples, dim)
        return self.forward(X, *args)

    def get_model_weights(self) -> List[np.ndarray]:
        return get_flat_array_from_list_of_arrays(self.model.get_weights())
        # return self.model.get_weights()

    def set_model_weights(self, weights: np.ndarray):
        reshaped_weights = get_reshaped_array_from_list_of_arrays(flat_array=weights.ravel(), list_of_arrays=self.model.get_weights())
        self.model.set_weights(reshaped_weights)


# -------------------------------------------------------------

class PassthroughLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape, units=1):
        super().__init__(input_shape=input_shape)
        self.units = units

    def build(self, input_shape):  # Create the state of the layer (weights)
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)

    def call(self, inputs):  # Defines the computation from inputs to outputs
        return self.w

class SimpleModel(TensorflowModel):
    def __init__(self, loss_function, dimensionality: int):  # introduce loss_function parameter because no data to compare to!
        super().__init__()
        self.model.add(PassthroughLayer(input_shape=(dimensionality,)))

        # Loss function is a benchmark function
        self._loss_function = loss_function

        # We don't have labels for benchmark function loss
        self.loss = partial(self.loss, Y_true=None)

# -------------------------------------------------------------

class LinearRegression(TensorflowModel):
    def __init__(self, dimensionality: int):  # introduce a dimensionality parameter because trainable weights!
        super().__init__()
        self.model.add(tf.keras.layers.Dense(1, input_shape=(dimensionality,)))


def MeanSquaredError(Y_pred, Y_true):
    fn = tf.keras.losses.MeanSquaredError()
    return fn(Y_true, Y_pred)

# -------------------------------------------------------------

class DataModule:
    """Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, root='../data'):
        self.root = root

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors)
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)


class Trainer():  # Dit moet eigenlijk een soort Function worden, maar dan met een ML architectuur en Data ...
    def __init__(self, model: Model = None, data: Data = None, loss_function=None):
        self.model = model
        self.data = data
        self.loss_function = loss_function

        # self.dimensionality = data.design.get_number_of_input_parameters()
        
    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # Two 2D arrays: loss (1,1), grad (dim, 1)

        self.model.set_model_weights(x)

        if self.data is None:
            X_data = x
            y_data = None
        else:
            X_data = self.data.get_input_data().to_numpy()
            y_data = self.data.get_output_data().to_numpy()

        with tf.GradientTape() as tape:
            loss = self.loss_function(Y_pred=self.model(X_data), Y_true=y_data)
            # loss = self.model.loss(Y_pred=self.model(X_data), Y_true=y_data)
        grads = tape.gradient(loss, self.model.trainable_variables)
        return np.atleast_2d(loss.numpy()), get_flat_array_from_list_of_arrays(grads)

    def f(self, x: np.ndarray):
        loss, _ = self.evaluate(x)
        return loss

    def __call__(self, x: np.ndarray):
        return self.f(x)

    def dfdx(self, x: np.ndarray):
        _, grads = self.evaluate(x)
        return grads

