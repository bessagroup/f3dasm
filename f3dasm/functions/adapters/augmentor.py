from abc import ABC
from typing import List

import autograd.numpy as np

from f3dasm.base.utils import _descale_vector, _scale_vector


class Augmentor(ABC):
    """
    Base class for operations that augment an loss-funciton
    """

    def augment(self, input: np.ndarray) -> np.ndarray:
        """Stub function to augment the input of a function

        Args:
            input (np.ndarray): input vector that needs to be augmented

        Returns:
            np.ndarray: augmented input vector
        """
        ...

    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        """Stub function to reverse the augmented input

        Args:
            output (np.ndarray): augmented input vector that needs to be undone

        Returns:
            np.ndarray: original input vector
        """
        ...


class Noise(Augmentor):
    def __init__(self, noise: float):
        """Augmentor class to add noise to a function output

        Args:
            noise (float): Standard deviation of for Gaussian noise
        """
        self.noise = noise

    def augment(self, input: np.ndarray) -> np.ndarray:
        # TODO: change noise calculation to work with autograd.numpy
        noise: np.ndarray = np.random.normal(loc=0.0, scale=abs(self.noise * input), size=input.shape)
        y_noise = input + noise
        return y_noise

    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        # TODO: change noise calculation to work with autograd.numpy
        noise: np.ndarray = np.random.normal(loc=0.0, scale=abs(self.noise * output), size=output.shape)
        y_noise = output - noise
        return y_noise


class Offset(Augmentor):
    """Augmentor class to offset the input vector of a function

    Args:
        offset (np.ndarray): Constant vector that offsets the function input
    """

    def __init__(self, offset: np.ndarray):
        self.offset = offset

    def augment(self, input: np.ndarray) -> np.ndarray:
        return input - self.offset  # -

    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        return output + self.offset


class Scale(Augmentor):
    """Augmentor class to scale the input vector of a function to some bounds

    Args:
        scale_bounds (np.ndarray): continuous bounds of the data
        input_domain (np.ndarray): input domain of the objective function
    """

    def __init__(self, scale_bounds: np.ndarray, input_domain: np.ndarray):
        self.scale_bounds = scale_bounds
        self.input_domain = input_domain

    def augment(self, input: np.ndarray) -> np.ndarray:
        return _scale_vector(x=_descale_vector(input, scale=self.scale_bounds), scale=self.input_domain)

    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        return _scale_vector(x=_descale_vector(output, scale=self.input_domain), scale=self.scale_bounds)


class FunctionAugmentor:
    """Combination of Augmentors that can change the input and output of an objective function

    Args:
        input_augmentors (List[Augmentor]): list of input augmentors
        output_augmentors (List[Augmentor]): list of output augmentors
    """

    def __init__(self, input_augmentors: List[Augmentor] = None, output_augmentors: List[Augmentor] = None):
        self.input_augmentors = [] if input_augmentors is None else input_augmentors
        self.output_augmentors = [] if output_augmentors is None else output_augmentors

    def add_input_augmentor(self, augmentor: Augmentor) -> None:
        """Add an input augmentor

        Args:
            augmentor (Augmentor): augmentor to be added
        """
        self.input_augmentors.append(augmentor)

    def add_output_augmentor(self, augmentor: Augmentor) -> None:
        """Add an output augmentor

        Args:
            augmentor (Augmentor): augmentor to be added
        """
        self.output_augmentors.append(augmentor)

    def augment_input(self, x: np.ndarray) -> np.ndarray:
        """Alter the input vector with the augmentors

        Args:
            x (np.ndarray): input vector

        Returns:
            np.ndarray: augmented input vector
        """
        for augmentor in self.input_augmentors:
            x = augmentor.augment(x)

        return x

    def augment_reverse_input(self, x: np.ndarray) -> np.ndarray:
        """Retrieve the original input from the augmented input

        Args:
            x (np.ndarray): augmented input

        Returns:
            np.ndarray: original input vector
        """
        for augmentor in reversed(self.input_augmentors):
            x = augmentor.reverse_augment(x)

        return x

    def augment_output(self, y: np.ndarray) -> np.ndarray:
        """Alter the output vector with the augmentors

        Args:
            x (np.ndarray): output vector

        Returns:
            np.ndarray: augmented output vector
        """
        for augmentor in self.output_augmentors:
            y = augmentor.augment(y)

        return y

    def augment_reverse_output(self, y: np.ndarray) -> np.ndarray:
        """Retrieve the original output from the augmented input

        Args:
            x (np.ndarray): augmented input

        Returns:
            np.ndarray: original output vector
        """
        for augmentor in reversed(self.output_augmentors):
            y = augmentor.reverse_augment(y)

        return y
