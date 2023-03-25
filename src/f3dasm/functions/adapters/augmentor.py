#                                                                       Modules
# =============================================================================

# Standard
from abc import ABC
from copy import copy
from typing import List

# Third-party
import autograd.numpy as np

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

        Parameters
        ----------
        input
            vector that needs to be augmented

        Returns
        -------
            augmented vector
        """
        ...

    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        """Stub function to reverse the augmented input

        Parameters
        ----------
        output
            augmented vector that needs to be undone

        Returns
        -------
            original vector
        """
        ...


class Noise(Augmentor):
    def __init__(self, noise: float):
        """Augmentor class to add noise to a function output

        Parameters
        ----------
        noise
            standard deviation of Gaussian noise (mean is zero)
        """
        self.noise = noise

    def augment(self, input: np.ndarray) -> np.ndarray:
        if hasattr(input, "_value"):
            yy = copy(input._value)
            if hasattr(yy, "_value"):
                yy = copy(yy._value)
        else:
            yy = copy(input)

        scale = abs(self.noise * yy)

        noise: np.ndarray = np.random.normal(
            loc=0.0, scale=scale, size=input.shape)
        y_noise = input + float(noise)
        return y_noise

    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        return self.augment


class Offset(Augmentor):
    def __init__(self, offset: np.ndarray):
        """Augmentor class to offset the input vector of a function

        Parameters
        ----------
        offset
            constant vector that offsets the function input
        """
        self.offset = offset

    def augment(self, input: np.ndarray) -> np.ndarray:
        return input - self.offset

    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        return output + self.offset


class Scale(Augmentor):
    def __init__(self, scale_bounds: np.ndarray, input_domain: np.ndarray):
        """Augmentor class to scale the input vector of a function to some bounds

        Parameters
        ----------
        scale_bounds
            continuous bounds (lower and upper for every dimension) to be scaled towards
        input_domain
            input domain of the objective function
        """
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
        """Combination of augmentors that can change the input and output of an objective function

        Parameters
        ----------
        input_augmentors, optional
            list of input augmentors, by default None
        output_augmentors, optional
            list of output augmentors, by default None
        """
        self.input_augmentors = [] if input_augmentors is None else input_augmentors
        self.output_augmentors = [] if output_augmentors is None else output_augmentors

    def add_input_augmentor(self, augmentor: Augmentor) -> None:
        """Add an input augmentor

        Parameters
        ----------
        augmentor
            augmentor to be added
        """
        self.input_augmentors.append(augmentor)

    def insert_input_augmentor(self, augmentor: Augmentor, position: int) -> None:
        """Insert an input augmentor at any place in the input_augmentors list

        Parameters
        ----------
        augmentor
            augmentor
        position
            position to put this augmentor in the input_augmentors list
        """
        self.input_augmentors.insert(position, augmentor)

    def add_output_augmentor(self, augmentor: Augmentor) -> None:
        """Add an output augmentor

        Parameters
        ----------
        augmentor
            augmentor to be added
        """
        self.output_augmentors.append(augmentor)

    def augment_input(self, x: np.ndarray) -> np.ndarray:
        """Alter the input vector with the augmentor

        Parameters
        ----------
        x
            input vector

        Returns
        -------
            augmented input vector
        """
        for augmentor in self.input_augmentors:
            x = augmentor.augment(x)

        return x

    def augment_reverse_input(self, x: np.ndarray) -> np.ndarray:
        """Retrieve the original input from the augmented input

        Parameters
        ----------
        x
            augmented input

        Returns
        -------
            original input vector
        """
        for augmentor in reversed(self.input_augmentors):
            x = augmentor.reverse_augment(x)

        return x

    def augment_output(self, y: np.ndarray) -> np.ndarray:
        """Alter the output vector with the augmentors

        Parameters
        ----------
        y
            output vector

        Returns
        -------
            augmented output vector
        """
        for augmentor in self.output_augmentors:
            y = augmentor.augment(y)

        return y

    def augment_reverse_output(self, y: np.ndarray) -> np.ndarray:
        """Retrieve the original output from the augmented input

        Parameters
        ----------
        y
            augmented input

        Returns
        -------
            original output vector
        """
        for augmentor in reversed(self.output_augmentors):
            y = augmentor.reverse_augment(y)

        return y


def _scale_vector(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Scale a vector x to a given scale"""
    return (scale[:, 1] - scale[:, 0]) * x + scale[:, 0]


def _descale_vector(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse of the _scale_vector() function"""
    return (x - scale[:, 0]) / (scale[:, 1] - scale[:, 0])
