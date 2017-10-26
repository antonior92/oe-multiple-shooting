"""
Implement feedforward neural network function.
"""

import numpy as np
from .activation_function import ActivationFunction, Identity


__all__ = [
    'FeedforwardNetwork'
]


class FeedforwardNetwork(object):
    """
    Feedforward network model.

    Init Parameters
    ---------------
    ninput : int
        Number of network inputs.
    noutput : int
        Number of network outputs.
    nhidden : list of ints
        A list containing the number of nodes
        in each hidden layer, ordered from the layer
        closest to the input layer to the layer closest
        to the output layer. The dimension
        of nhidden is the number of hidden
        layers.
    activ_func : ActivationFunction or list of ActivationFunction
        Activation function used on the hidden layers.
        A list of ActivationFunction should have dimension
        ``nhidden+1``, each ActivationFunction from the list
        will be used on the correspondent hidden layer and,
        the last element from the list, in the output layer.
        For a single ActivationFunction, the provided activation
        function will be used on all the hidden layers and the
        output layer will have an identity activation function.

    Call Parameters
    ---------------
    x : array_like
        Array containing input.
    params : array_like
        Parameter vector.

    Returns
    -------
    z : array_like
        Array containing the output of the network
        for the given input.
    """

    def __init__(self, ninput, noutput, nhidden, activ_func):

        # Save number of inputs/outputs/hidden layers
        self.ninput = ninput
        self.noutput = noutput
        self.nhidden = nhidden

        # Compute number of parameters
        weights_shape = []
        bias_shape = []
        nparams = 0
        layer = [ninput]+nhidden+[noutput]
        for k in range(len(layer)-1):
            weights_shape += [(layer[k+1], layer[k])]
            bias_shape += [(layer[k+1],1)]
            nparams += layer[k+1]*(layer[k]+1)

        self.weights_shape = weights_shape
        self.bias_shape = bias_shape
        self.nparams = nparams
        self.nlayers = len(layer)-1

        # Get activation functions
        if isinstance(activ_func, ActivationFunction):
            self.activ_func = [activ_func]*len(nhidden)+[Identity()]
        elif all(isinstance(f, ActivationFunction) for f in activ_func):
            if len(activ_func) == len(nhidden)+1:
                self.activ_func = activ_func
            else:
                raise ValueError(r"If activ_func is a list it should have \
                                  dimension nhidden+1")
        else:
            raise ValueError(r"activ_func should be an ActivationFunction \
                               or list of ActivationFunction")

    def __call__(self, x, params):

        # Check Arguments
        x, params = self._arg_check(x, params)

        # Get weights and bias
        w, b = self.disassemble_params(params)

        # Go through the layers propagating
        # the values
        layer_output = x.reshape(self.ninput, 1)
        for i in range(self.nlayers):

            # Get activation function for the
            # current layer
            activ_func = self.activ_func[i]

            # Compute the layer input
            activ_func_input = np.dot(w[i], layer_output)+b[i]

            # Apply the activation function
            layer_output = activ_func(activ_func_input)

        # Return
        z = layer_output.flatten()
        return z

    def derivatives(self, x, params, deriv_x=True, deriv_params=True):
        """
        For a neural network function z = G(x, params) compute the derivatives
        in relation to x and in relation to params.

        Parameters
        ----------
        x : array_like
            Array containing input.
        params : array_like
            Parameter vector.
        deriv_x : boolean, optional
            Specify if the derivatives in relation to x
            should be returned by this function. By default,
            it is True.
        deriv_params : boolean, optional
            Specify if the derivatives in relation to the
            parameters should be returned by this function.
            By default, it is True.

        Returns
        -------
        dx : array_like, optional
            Array containing model derivatives in relation
            to x. Dimension (noutputs, ninputs)
        dparams : array_like, optional
            Array containing model derivatives in relation
            to params. Dimension (noutputs, nparams)
        """

        # Check Arguments
        x, params = self._arg_check(x, params)

        # Get weights and bias
        w, b = self.disassemble_params(params)

        # --------- Foward propagation -----------
        layer_output = x.reshape(self.ninput, 1)
        list_dfunc = []
        list_layer_output = [layer_output]
        for i in range(self.nlayers):

            # Get activation function for the
            # current layer
            activ_func = self.activ_func[i]

            # Compute the layer input
            activ_func_input = np.dot(w[i], layer_output)+b[i]

            # Apply the activation function
            layer_output = activ_func(activ_func_input)

            # Store it
            list_layer_output += [layer_output]

            # Get the derivative for the activation
            # function of the i-th layer
            dfunc = activ_func.derivatives(activ_func_input).flatten()

            # Store it
            list_dfunc += [dfunc]

        # ---------  Backwards propagation -----------

        # Get activation function derivative
        dfunc = list_dfunc.pop()

        # Get output
        z = list_layer_output.pop()

        # Initialize derivative
        deriv = np.diag(dfunc)

        # initialize parameters
        dparams = np.empty((self.noutput, self.nparams))

        # Start parameter index
        pindex = self.nparams

        for k in range(self.nlayers-1, 0, -1):

            # Get activation function derivative
            dfunc = list_dfunc.pop()

            # Get last layer output
            layer_output = list_layer_output.pop()

            # Compute dparams
            if deriv_params:
                for i in range(self.bias_shape[k][0])[::-1]:

                    # Compute the derivative
                    pindex -= 1
                    dparams[:, pindex] = deriv[:, i]

                for i in range(self.weights_shape[k][0])[::-1]:
                    for j in range(self.weights_shape[k][1])[::-1]:

                        # Compute the derivative
                        pindex -= 1
                        dparams[:, pindex] = deriv[:, i]*layer_output[j]

            # Propagate derivative backwards
            deriv = np.dot(deriv, w[k]) * dfunc

        # Get last layer output
        layer_output = list_layer_output.pop()

        # Compute dparams
        if deriv_params:
            k = 0
            for i in range(self.bias_shape[k][0])[::-1]:

                # Compute the derivative
                pindex -= 1
                dparams[:, pindex] = deriv[:, i]

            for i in range(self.weights_shape[k][0])[::-1]:
                for j in range(self.weights_shape[k][1])[::-1]:

                    # Compute the derivative
                    pindex -= 1
                    dparams[:, pindex] = deriv[:, i]*layer_output[j]

        # Compute dx
        if deriv_x:
            dx = np.dot(deriv, w[0])

        # Return
        if deriv_x and deriv_params:
            return dx, dparams
        elif deriv_params:
            return dparams
        else:
            return dx

    def _arg_check(self, x, params):
        """
        Check input arguments.
        """

        x = np.atleast_1d(x).flatten()
        params = np.atleast_1d(params).flatten()

        if x.shape[0] != self.ninput:
            raise ValueError("Wrong x dimension")
        if params.shape[0] != self.nparams:
            raise ValueError("Wrong params dimension")

        return x, params

    def disassemble_params(self, params):
        """
        Get weight and bias from a parameter vector

        Parameters
        ----------
        params : array_like
            Parameter vector.

        Returns
        -------
        weights : list of array_like
            List containing one weight matrix per layer
        bias : list of array_like
            List containing one bias vector per layer
        """

        # Check input
        if len(params) != self.nparams:
            raise ValueError("Wrong params dimension")

        index = 0
        weights = []
        bias = []
        for i in range(self.nlayers):

            # Get i-th weight matrix
            length = self.weights_shape[i][0] * \
                     self.weights_shape[i][1]
            w = np.atleast_2d(params[index:index+length])
            w = w.reshape(self.weights_shape[i])
            weights += [w]
            index += length

            # Get i-th bias vector
            length = self.bias_shape[i][0]
            b = np.atleast_2d(params[index:index+length])
            b = b.reshape(self.bias_shape[i])
            bias += [b]
            index += length

        return weights, bias

    def assemble_params(self, weights, bias):
        """
        Assemble parameter vector from weight matrix
        and bias vectors

        Parameters
        ----------
        weights : list of array_like
            List containing one weight matrix per layer
        bias : list of array_like
            List containing one bias vector per layer

        Returns
        -------
        params : array_like
            Parameter vector.
        """

        # Initialize empty matrix
        params = np.empty(self.nparams)

        # Check lenghts
        if len(weights) != self.nlayers:
            raise ValueError("weights should be a list containing \
                              one weights matrix per layer")
        if len(bias) != self.nlayers:
            raise ValueError("bias should be a list containing \
                              one bias vector per layer")

        index = 0
        for i in range(self.nlayers):

            # Set i-th weights matrix on parameter vector
            length = self.weights_shape[i][0] * \
                     self.weights_shape[i][1]
            w = np.atleast_2d(weights[i]).flatten()

            if len(w) != length:
                raise ValueError("Wrong weights matrix shape")

            params[index:index+length] = w
            index += length

            # Set i-th bias vector on parameter vector
            length = self.bias_shape[i][0]
            b = np.atleast_2d(bias[i]).flatten()

            if len(b) != length:
                raise ValueError("Wrong bias vector shape")

            params[index:index+length] = b
            index += length

        return params

