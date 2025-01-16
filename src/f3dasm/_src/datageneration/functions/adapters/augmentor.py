#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from copy import copy
from typing import List, Optional

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


class _Augmentor(ABC):
    """
    Base class for operations that augment an loss-funciton
    """
    @abstractmethod
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

    @abstractmethod
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


class EmptyAugmentor(_Augmentor):
    def augment(self, input: np.ndarray) -> np.ndarray:
        return input

    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        return output


class Noise(_Augmentor):
    def __init__(self, noise: float, rng: np.random.Generator):
        """Augmentor class to add noise to a function output

        Parameters
        ----------
        noise
            standard deviation of Gaussian noise (mean is zero)
        """
        self.noise = noise
        self.rng = rng

    def augment(self, input: np.ndarray) -> np.ndarray:
        if hasattr(input, "_value"):
            yy = copy(input._value)
            if hasattr(yy, "_value"):
                yy = copy(yy._value)
        else:
            yy = copy(input)

        scale = abs(self.noise * yy)

        if isinstance(input, float):
            # convert to numpy float
            input = np.float64(input)

        noise: np.ndarray = self.rng.normal(
            loc=0.0, scale=scale, size=input.shape)
        y_noise = input + float(noise)
        return y_noise

    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        return self.augment


class Offset(_Augmentor):
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


class Scale(_Augmentor):
    def __init__(self, scale_bounds: np.ndarray | List[List[float]],
                 input_domain: np.ndarray):
        """Augmentor class to scale the input vector of a
         function to some bounds

        Parameters
        ----------
        scale_bounds
            continuous bounds (lower and upper for every dimension)
             to be scaled towards
        input_domain
            input domain of the objective function
        """
        self.scale_bounds = np.array(scale_bounds)
        self.input_domain = input_domain

    def augment(self, input: np.ndarray) -> np.ndarray:
        return _scale_vector(
            x=_descale_vector(input, scale=self.scale_bounds),
            scale=self.input_domain)

    def reverse_augment(self, output: np.ndarray) -> np.ndarray:
        return _scale_vector(
            x=_descale_vector(output, scale=self.input_domain),
            scale=self.scale_bounds)


class FunctionAugmentor:
    """Combination of Augmentors that can change the input and
     output of an objective function

    Parameters
    ----------
    input_augmentors : List[Augmentor]
        list of input augmentors
    outpu_augmentors : List[Augmentor]
        list of output augmentors
    """

    def __init__(
            self, input_augmentors: Optional[List[_Augmentor]] = None,
            output_augmentors: Optional[List[_Augmentor]] = None):
        """Combination of augmentors that can change the input and output of
         an objective function

        Parameters
        ----------
        input_augmentors : Optional[List[_Augmentor]]
            list of input augmentors, by default None
        output_augmentors: Optional[List[_Augmentor]]
            list of output augmentors, by default None
        """
        self.input_augmentors = [] if \
            input_augmentors is None else input_augmentors
        self.output_augmentors = [] if \
            output_augmentors is None else output_augmentors

    def add_input_augmentor(self, augmentor: _Augmentor) -> None:
        """Add an input augmentor

        Parameters
        ----------
        augmentor
            augmentor to be added
        """
        self.input_augmentors.append(augmentor)

    def insert_input_augmentor(
            self, augmentor: _Augmentor, position: int) -> None:
        """Insert an input augmentor at any place in the input_augmentors list

        Parameters
        ----------
        augmentor
            augmentor
        position
            position to put this augmentor in the input_augmentors list
        """
        self.input_augmentors.insert(position, augmentor)

    def add_output_augmentor(self, augmentor: _Augmentor) -> None:
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
    """Scale a vector x to a given scale

    Parameters
    ----------
    x : np.ndarray
        vector to be scaled
    scale : np.ndarray
        scale to be scaled towards

    Returns
    -------
    np.ndarray
        scaled vector
    """
    return (scale[:, 1] - scale[:, 0]) * x + scale[:, 0]


def _descale_vector(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse of the _scale_vector() function

    Parameters
    ----------
    x : np.ndarray
        scaled vector
    scale : np.ndarray
        scale to be scaled towards

    Returns
    -------
    np.ndarray
        descaled vector
    """
    return (x - scale[:, 0]) / (scale[:, 1] - scale[:, 0])
