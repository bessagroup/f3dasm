#                                                                       Modules
# =============================================================================

# Standard
from abc import ABC
from typing import List

# Third-party
import autograd.numpy as np

# Locals
from f3dasm.base.utils import _descale_vector, _scale_vector

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


class Augmentor(ABC):
    """
    Base class for operations that augment an loss-funciton
    """

    def augment(self, input: np.ndarray) -> np.ndarray:
        """Stub function to augment the input of a function

        :param input: vector that needs to be augmented
        :return: augmented vector
        """
        ...

    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        """Stub function to reverse the augmented input

        :param output: augmented vector that needs to be undone
        :return: original vector
        """

        ...


class Noise(Augmentor):
    def __init__(self, noise: float):
        """Augmentor class to add noise to a function output

        :param noise: standard deviation for Gaussian noise
        """

        self.noise = noise

    def augment(self, input: np.ndarray) -> np.ndarray:
        """Stub function to augment the input of a function

        :param input: vector that needs to be augmented
        :return: augmented vector
        """
        # TODO: change noise calculation to work with autograd.numpy
        noise: np.ndarray = np.random.normal(
            loc=0.0, scale=abs(self.noise * input), size=input.shape)
        y_noise = input + noise
        return y_noise

    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        """Stub function to reverse the augmented input

        :param output: augmented input vector that needs to be undone
        :return: original input vector
        """
        # TODO: change noise calculation to work with autograd.numpy
        noise: np.ndarray = np.random.normal(
            loc=0.0, scale=abs(self.noise * output), size=output.shape)
        y_noise = output - noise
        return y_noise


class Offset(Augmentor):
    def __init__(self, offset: np.ndarray):
        """Augmentor class to offset the input vector of a function

        :param offset: constant vector that offset the function input
        """
        self.offset = offset

    def augment(self, input: np.ndarray) -> np.ndarray:
        """Stub function to augment the input of a function

        :param input: vector that needs to be augmented
        :return: augmented vector
        """
        return input - self.offset  # -

    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        """Stub function to reverse the augmented input

        :param output: augmented input vector that needs to be undone
        :return: original input vector
        """
        return output + self.offset


class Scale(Augmentor):
    def __init__(self, scale_bounds: np.ndarray, input_domain: np.ndarray):
        """Augmentor class to scale the input vector of a function to some bounds

        :param scale_bounds: continuous bounds of the data
        :param input_domain: input domain of the objective function
        """
        self.scale_bounds = scale_bounds
        self.input_domain = input_domain

    def augment(self, input: np.ndarray) -> np.ndarray:
        """Stub function to augment the input of a function

        :param input: vector that needs to be augmented
        :return: augmented vector
        """
        return _scale_vector(x=_descale_vector(input, scale=self.scale_bounds), scale=self.input_domain)

    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        """Stub function to reverse the augmented input

        :param output: augmented input vector that needs to be undone
        :return: original input vector
        """
        return _scale_vector(x=_descale_vector(output, scale=self.input_domain), scale=self.scale_bounds)


class FunctionAugmentor:
    """Combination of Augmentors that can change the input and output of an objective function

    Args:
        input_augmentors (List[Augmentor]): list of input augmentors
        output_augmentors (List[Augmentor]): list of output augmentors
    """

    def __init__(self, input_augmentors: List[Augmentor] = None, output_augmentors: List[Augmentor] = None):
        """Combination of augmentors that can change the input and output of an objective function

        :param input_augmentors: list of input augmentors
        :param output_augmentors: list of output augmentors
        """
        self.input_augmentors = [] if input_augmentors is None else input_augmentors
        self.output_augmentors = [] if output_augmentors is None else output_augmentors

    def add_input_augmentor(self, augmentor: Augmentor) -> None:
        """Add an input augmentor

        :param augmentor: augmentor to be added
        """
        self.input_augmentors.append(augmentor)

    def add_output_augmentor(self, augmentor: Augmentor) -> None:
        """Add an output augmentor

        :param augmentor: augmentor to be added
        """
        self.output_augmentors.append(augmentor)

    def augment_input(self, x: np.ndarray) -> np.ndarray:
        """Alter the input vector with the augmentors

        :param x: input vector
        :return: augmented input vecotr
        """
        for augmentor in self.input_augmentors:
            x = augmentor.augment(x)

        return x

    def augment_reverse_input(self, x: np.ndarray) -> np.ndarray:
        """Retrieve the original input from the augmented input

        :param x: augmented input
        :return: original input vector
        """
        for augmentor in reversed(self.input_augmentors):
            x = augmentor.reverse_augment(x)

        return x

    def augment_output(self, y: np.ndarray) -> np.ndarray:
        """Alter the output vector with the augmentors

        :param y: output vector
        :return: augmented output vector
        """
        for augmentor in self.output_augmentors:
            y = augmentor.augment(y)

        return y

    def augment_reverse_output(self, y: np.ndarray) -> np.ndarray:
        """Retrieve the original output from the augmented input

        :param y: augmented input
        :return: original output vector
        """
        for augmentor in reversed(self.output_augmentors):
            y = augmentor.reverse_augment(y)

        return y
