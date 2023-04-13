from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..functions import Augmentor, Function

def warp(input_row: np.ndarray) -> np.ndarray:
    input_row = input_row ** 2
    return input_row

def reverse_warp(input_row: np.ndarray) -> np.ndarray:
    input_row = input_row ** .5
    return input_row

def scale(input_row: np.ndarray, output_row: np.ndarray) -> np.ndarray:
    output_row *= 1 + np.sin(2 * np.pi * input_row) / 2
    return output_row

def reverse_scale(input_row: np.ndarray, output_row: np.ndarray) -> np.ndarray:
    output_row /= 1 + np.sin(2 * np.pi * input_row) / 2
    return output_row


class Warp(Augmentor):

    def __init__(self, warp, reverse_warp) -> None:
        super().__init__()
        self.warp = warp
        self.reverse_warp = reverse_warp

    def augment(self, input_row: np.ndarray) -> np.ndarray:
        return self.warp(input_row)
    
    def reverse_augment(self, output_row: np.ndarray) -> np.ndarray:
        return self.reverse_warp(output_row)
    

class Scale(Augmentor):

    def __init__(self, scale_factor) -> None:
        super().__init__()
        self.scale_factor = scale_factor
    
    def scale(self, input_row: np.ndarray, output_row: np.ndarray) -> np.ndarray:
        output_row_scaled = output_row * self.scale_factor(input_row=input_row)
        return output_row_scaled
    
    def reverse_scale(self, input_row: np.ndarray, output_row: np.ndarray) -> np.ndarray:
        output_row_reverse_scaled = output_row / self.scale_factor(input_row=input_row)
        return output_row_reverse_scaled

    def augment(self, input_row: np.ndarray, output_row: np.ndarray) -> np.ndarray:
        return self.scale(input_row, output_row)
    
    def reverse_augment(self, input_row: np.ndarray, output_row: np.ndarray) -> np.ndarray:
        return self.reverse_scale(input_row, output_row)
    

class NoiseInterpolator(Augmentor):

    def __init__(self, noise_var, aug_coeff) -> None:
        super().__init__()
        self.noise_var = noise_var
        self.aug_coeff = aug_coeff

    def augment(self, input_row: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        scaler.fit(input_row)
        input_row_scaled = scaler.transform(input_row)
        noise = self.noise_var * np.random.randn(*input_row_scaled.shape)
        input_row_scaled_augmented = self.aug_coeff * input_row_scaled + (1 - self.aug_coeff) * noise
        input_row_augmented = scaler.inverse_transform(input_row_scaled_augmented)
        return input_row_augmented
    
    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        return super().reverse_augment(output)
    

class NoisyBaseFun(Function):
    def __init__(self, seed=None, base_fun=None, dimensionality=None, noise_augmentor=None,):
        super().__init__(seed)
        self.base_fun: Function = base_fun
        self.dimensionality = dimensionality
        self.noise_augmentor: NoiseInterpolator = noise_augmentor
        self.scale_bounds = base_fun.scale_bounds
        self.name = base_fun.name
    
    def __call__(self, input_x: np.ndarray):
        res = self.noise_augmentor.augment(input_row=self.base_fun(input_x))
        return res
    

class BiasedBaseFun(Function):
    def __init__(self, seed=None, base_fun=None, dimensionality=None, scale_augmentor=None,):
        super().__init__(seed)
        self.base_fun: Function = base_fun
        self.dimensionality = dimensionality
        self.scale_augmentor: Scale = scale_augmentor
        self.scale_bounds = base_fun.scale_bounds
        self.name = base_fun.name
    
    def __call__(self, input_x: np.ndarray):
        res = self.scale_augmentor.augment(input_row=input_x, output_row=self.base_fun(input_x))
        return res
