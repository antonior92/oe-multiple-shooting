"""
Some activation functions for neural network models.
"""
import numpy as np
from abc import (ABCMeta, abstractmethod)
from six import add_metaclass
from scipy.special import expit

__all__ = [
    'ActivationFunction',
    'Identity',
    'Logistic',
    'HyperbolicTangent'
]


@add_metaclass(ABCMeta)
class ActivationFunction(object):
    """
    Define how to compute an activation function
    and its derivative.
    """

    @abstractmethod
    def __call__(self, x):
        """
        Given an input compute the node output
        for the activation function.

        Parameters
        ----------
        x : float or array_like
            Node input.

        Returns
        -------
        z : float or array_like
            Node output. If the input is an
            array the output should be an array of
            same size with the activation function
            applied element wise.
        """
        return

    @abstractmethod
    def derivatives(self, x):
        """
        Activation function first derivative
        evaluated for a given input

        Parameters
        ----------
        x : float or array_like
            Node input.

        Returns
        -------
        dz : float or array_like
            Activation function first derivative.
            If the input is an array the output should be an
            array of same size with the activation function
            derivative applied element wise.
        """
        return


class Identity(ActivationFunction):
    """
    The identity activation function
    is defined as z = x
    """

    def __call__(self, x):

        return x

    def derivatives(self, x):
        """Unidimensional derivative"""

        return np.ones_like(x)


class Logistic(ActivationFunction):
    """
    The logistic activation function
    is defined as z = 1/(1+exp(-x))
    """

    def __call__(self, x):

        return expit(x)

    def derivatives(self, x):

        z = expit(x)
        dz = np.multiply(z, 1-z)
        return dz


class HyperbolicTangent(ActivationFunction):
    """
    The hyperbolic tangent activation function
    is defined as z = tanh(x)
    """

    def __call__(self, x):

        return np.tanh(x)

    def derivatives(self, x):

        z = np.tanh(x)
        dz = 1-np.power(z, 2)
        return dz
