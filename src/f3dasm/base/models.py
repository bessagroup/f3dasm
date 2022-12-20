from functools import partial
from typing import List

import numpy as np
import tensorflow as tf


def get_reshaped_array_from_list_of_arrays(flat_array: np.ndarray, list_of_arrays: List[np.ndarray]) ->  List[np.ndarray]:
    total_array = []
    index = 0
    for mimic_array in list_of_arrays:
        number_of_values = np.product(mimic_array.shape)
        current_array = np.array(flat_array[index:index+number_of_values])

        if number_of_values > 1:
            current_array = current_array.reshape(-1,1) # Make 2D array

        total_array.append(current_array)
        index += number_of_values

    return total_array

def get_flat_array_from_list_of_arrays(list_of_arrays: List[np.ndarray]) -> List[np.ndarray]: # technically not a np array input!
    return np.concatenate([np.atleast_2d(array) for array in list_of_arrays])

class MLArchitecture(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.Sequential()
        # self.model.add(tf.keras.layers.InputLayer(input_shape=(dimensionality,)))

    def _forward(self, X):
        assert hasattr(self, 'model'), 'model is defined'
        return self.model(X)

    def call(self, X, *args, **kwargs): # Shape: (samples, dim)
        return self._forward(X, *args)

    def loss(self, Y_pred, Y_true): #Y_hat = model output, Y = labels
        return self._loss_function(Y_pred, Y_true)

    def _loss_function(Y_pred, Y_true):
        raise NotImplementedError

    def get_model_weights(self) -> List[np.ndarray]:
        return get_flat_array_from_list_of_arrays(self.model.get_weights())
        # return self.model.get_weights()

    def set_model_weights(self, weights: np.ndarray):
        reshaped_weights = get_reshaped_array_from_list_of_arrays(flat_array=weights.ravel(), list_of_arrays=self.model.get_weights())
        self.model.set_weights(reshaped_weights)

    # RECONSIDER THESE METHODS

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])


# -------------------------------------------------------------

class SimpleModel(MLArchitecture):
    def __init__(self, loss_function): # introduce loss_function parameter because no data to compare to!
        super().__init__()

        # Loss function is a benchmark function
        self._loss_function = loss_function

        # We don't have labels for benchmark function loss
        self.loss = partial(self.loss, Y_true=None)

    def get_weights():
        return NotImplementedError("There are no trainable weights to for this type of models!")

    def set_weights():
        return NotImplementedError("There are no trainable weights to for this type of models!")

# -------------------------------------------------------------

class LinearRegression(MLArchitecture):
    def __init__(self, dimensionality: int): # introduce a dimensionality parameter because trainable weights!
        super().__init__()
        self.model.add(tf.keras.layers.Dense(1, input_shape=(dimensionality,)))

    def _loss_function(self, y_hat, y):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)


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


class Trainer:
    """Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, max_epochs, gradient_clip_val=0):
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val

        # self.optim = tf.keras.optimizers.SGD(0.03)

    def prepare_data(self, data: DataModule):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model: MLArchitecture):
        self.model = model

    def prepare_optimizer(self, optimizer):
        self.optimizer = optimizer

    def fit(self, model: MLArchitecture, data: DataModule, optimizer):
        self.prepare_data(data)
        self.prepare_model(model)
        self.prepare_optimizer(optimizer)
        # self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def extract_model(self):
        return self.model

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_linear_scratch`"""
        return batch

    def evaluate(self, X, Y_true):
        with tf.GradientTape() as tape:
            loss = self.model.loss(self.model(X), Y_true)

        grads = tape.gradient(loss, self.model.trainable_variables)
        return loss, grads



    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        # self.model.training = True
        for batch in self.train_dataloader:
            loss, grads = self.evaluate(*batch[:-1],batch[-1])

            # Optimization update
            w = self.model.get_model_weights()
            update = w - (0.03 * get_flat_array_from_list_of_arrays(grads))
            self.model.set_model_weights(update)

            self.train_batch_idx += 1


        if self.val_dataloader is None:
            return        
        # self.model.training = False
        for batch in self.val_dataloader:
            self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1
