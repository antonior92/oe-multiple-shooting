"""
Feedforward Network NARX model.
"""

from __future__ import division, print_function, absolute_import
import numpy as np
from .narx import NarxModel
from ..neuralnetwork import (feedforward_network, activation_function,
                             LeCunInitializer, scale_input_layer,
                             scale_output_layer)


__all__ = [
    'FeedforwardNetwork'
]


class FeedforwardNetwork(NarxModel):
    """
    Create a FeedforwardNetwork NARX model

    Init Parameters
    ---------------
    N : int
        Number of past output taken in account by the ARX model.
    M : int
        Number of past input taken in account by the ARX model.
    nhidden : list of ints
        A list containing the number of nodes
        in each hidden layer, ordered from the layer
        closest to the input layer to the layer closest
        to the output layer. The dimension
        of nhidden is the number of hidden
        layers.
    Ny : int, optional
        Number of outputs. By default ``Ny = 1``.
    Nu : int, optional
        Number of inputs. By default ``Nu = 1``.
    delay : int, optional
        System output delay. By default ``delay = 1``.
    activ_func : ActivationFunction or list of ActivationFunction, optional
        Activation function used on the hidden layers.
        A list of ActivationFunction should have dimension
        ``nhidden+1``, each ActivationFunction from the list
        will be used on the correspondent hidden layer and,
        the last element from the list, in the output layer.
        For a single ActivationFunction, the provided activation
        function will be used on all the hidden layers and the
        output layer will have an identity activation function.
        By default use HyperbolicTangent as activation function.

    Call Parameters
    ---------------
    y : array_like
        Array containing previous system outputs.
        It should have dimension (N, Ny).
    u : array_like
        Array containing system inputs.
        it should have dimension (M-delay+1, Nu).
    params : array_like
        Parameter array. It should
        have dimension (Nparams,).

    Returns
    -------
    ynext : array_like
        Array containing next system output accordingly
        to the NARX model. Dimension (Ny,).
    """

    def __init__(self, N, M, nhidden, Ny=1, Nu=1, delay=1,
                 activ_func=activation_function.HyperbolicTangent()):

        ninput = N*Ny+(M-delay+1)*Nu
        noutput = Ny

        self.net = feedforward_network.FeedforwardNetwork(ninput, noutput,
                                                          nhidden, activ_func)

        Nparams = self.net.nparams

        NarxModel.__init__(self, Nparams, N, M, Ny, Nu, delay)

    def __call__(self, y, u, params):

        # Check inputs
        y, u, params = self._arg_check(y, u, params)

        # Stack input and output and use as input
        # to the Neural Network
        x = self.marshalling_input(y, u)
        ynext = self.net(x, params)

        return ynext

    def derivatives(self, y, u, params, deriv_y=True,
                    deriv_u=True, deriv_params=True):

        # Check inputs
        y, u, params = self._arg_check(y, u, params)

        # Compute neural network derivatives
        x = self.marshalling_input(y, u)
        dx, dparams = self.net.derivatives(x, params)

        returns = []
        # Assemble dy
        if deriv_y:
            dy = np.reshape(dx[:, :self.N*self.Ny],
                            (self.Ny, self.N, self.Ny))
            returns += [dy]

        # Assemble du
        if deriv_u:
            du = np.reshape(dx[:,  self.N*self.Ny:],
                            (self.Ny, self.Mu, self.Nu))
            returns += [du]

        # Assemble dparams
        if deriv_params:
            returns += [dparams]

        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)

    def params_random_guess(self, ybounds, ubounds, init=LeCunInitializer()):
        """
        Return a random initial guess for the neural network
        parameters. This parameter can be used as initial parameter
        to an optimization process.

        Parameters
        ----------
        ybounds : array_like
            Array containing sequence of [lower_bound, upper_bound],
            one for each input. It should have shape (Ny, 2).
        ubounds : array_like
            Array containing sequence of [lower_bound, upper_bound].
            one for each input. It should have shape (Nu, 2).
        init : Initializer, optional
            Initializer object with the especifications
            for the initialization. By default use LeCunInitializer()
            with default configurations.

        Returns
        -------
        params : array_like
            An initial guess for the parameter vector.
        """
        # Get atributes
        N = self.N
        Mu = self.Mu
        Nu = self.Nu
        Ny = self.Ny

        ybounds = np.atleast_2d(ybounds)
        ubounds = np.atleast_2d(ubounds)

        # Check input shapes
        if ybounds.shape[0] != Ny or ybounds.shape[1] != 2:
            raise ValueError("ybounds.shape != (Ny, 2)")
        if ubounds.shape[0] != Nu or ubounds.shape[1] != 2:
            raise ValueError("ubounds.shape != (Nu, 2)")

        # Check input values
        if (ybounds[:, 0] >= ybounds[:, 1]).any():
            raise ValueError("lower_bound >= upper_bound for y")
        if (ubounds[:, 0] >= ubounds[:, 1]).any():
            raise ValueError("lower_bound >= upper_bound for u")

        # First guess for weights and bias
        weights, bias = init(self.net)

        # Get upper and lower bounds for input
        ylower = np.tile(ybounds[:, 0], (N, 1))
        ulower = np.tile(ubounds[:, 0], (Mu, 1))
        in_lower_bound = self.marshalling_input(ylower, ulower)

        yupper = np.tile(ybounds[:, 1], (N, 1))
        uupper = np.tile(ubounds[:, 1], (Mu, 1))
        in_upper_bound = self.marshalling_input(yupper, uupper)

        # Scale input
        weights, bias = scale_input_layer(weights, bias,
                                          in_upper_bound,
                                          in_lower_bound)

        # Get upper and lower bounds for input
        out_lower_bound = ybounds[:, 0]
        out_upper_bound = ybounds[:, 1]

        # Scale output
        weights, bias = scale_output_layer(weights, bias,
                                           out_upper_bound,
                                           out_lower_bound)

        # Assemble Parameters
        params = self.net.assemble_params(weights, bias)

        return params
