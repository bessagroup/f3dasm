from typing import List
import numpy as np
from f3dasm.functions import Augmentor
from sklearn.preprocessing import StandardScaler


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

    def __init__(self, scale, reverse_scale) -> None:
        super().__init__()
        self.scale = scale
        self.reverse_scale = reverse_scale

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
        input_row_scaled *= self.aug_coeff
        input_row_scaled += (1 - self.aug_coeff) * noise
        input_row = scaler.inverse_transform(input_row_scaled)
        return input_row
    
    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        return super().reverse_augment(output)