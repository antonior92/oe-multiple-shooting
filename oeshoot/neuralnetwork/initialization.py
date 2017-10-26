"""
Implement class and functions to be used on neural network
initialization
"""
import numpy as np

__all__ = [
    'LeCunInitializer',
    'scale_input_layer',
    'scale_output_layer'
]


class LeCunInitializer(object):
    """
    Weight initialization _[1]. Weights are initialized
    with standard deviation of ``gain/np.sqrt(fan_in)``
    and bias are initialized with zeros.

    Init Parameters
    ---------------
    gain : float, optional
        Scaling factor for the weights. By default it is 1.
    distribution : str, optional
        Weights are sampled from the especified distribution.
        The implemented distributions are 'normal'
        and 'uniform'. By default it is 'normal'.

    Call Parameters
    ---------------
    net : FeedforwardNetwork
        Network structure to be considered.

    Returns
    -------
    weights : list of array_like
        List containing one weight matrix per layer
    bias : list of array_like
        List containing one bias vector per layer

    References
    ----------
    .. [1] LeCun, Y., Bottou, L., Orr, G. B., and Muller, K. (1998a).
    Efficient backprop. In Neural Networks, Tricks of the Trade.

    """
    def __init__(self, gain=1,  distribution='normal'):

        # Test input
        if gain <= 0:
            raise ValueError("gain <= 0")
        if distribution != 'normal' and distribution != 'uniform':
            raise ValueError("Unknown distribution.")

        # Save parameters
        self.gain = gain
        self.distribution = distribution

    def __call__(self, net):

        gain = self.gain
        distribution = self.distribution

        # Initialize weight and bias lists
        weights = []
        bias = []

        # Initialize each weight matrix and each bias vector
        for i in range(net.nlayers):
            # Get shape
            wshape = net.weights_shape[i]
            bshape = net.bias_shape[i]

            sqrt_fanin = np.sqrt(wshape[1])

            # Compute weight matrix and bias vector
            if distribution == 'normal':
                w = np.random.normal(scale=gain/sqrt_fanin, size=wshape)
            elif distribution == 'uniform':
                # The np.sqrt(3) is to guarantee the
                # desired standard deviation
                w = np.random.uniform(low=-np.sqrt(3)*gain/sqrt_fanin,
                                      high=np.sqrt(3)*gain/sqrt_fanin,
                                      size=wshape)
            b = np.zeros(bshape)

            weights += [w]
            bias += [b]

        return weights, bias


def scale_input_layer(weights, bias, upper_bound, lower_bound):
    """
    Considering that the network input is bounded by upper and
    lower bounds, scale weights matrix and bias vector.

    Parameters
    ----------
    weights : list of array_like
        List containing one weight matrix per layer.
    bias : list of array_like
        List containing one bias vector per layer.
    upper_bound, lower_bound : array_like
        Lists containing upper and lower bounds for each input.

    Returns
    -------
    new_weights : list of array_like
        New list containing weights.
    new_bias : list of array_like
        New list containing bias.

    Notes
    -----
    Consider ``W = new_weights[0]/weights[0]``
    and ``b = new_bias[0]-bias[0]``,
    the following formula does hold:

        ``ninput*ones=W.dot(upper_bound)+b``
        ``ninput*ones=W.dot(lower_bound)+b``
    """
    # Get first layer weights and bias
    weight0 = np.atleast_2d(weights[0])
    bias0 = np.atleast_1d(bias[0])

    # Reshape input
    upper_bound = np.atleast_1d(upper_bound)
    lower_bound = np.atleast_1d(lower_bound)

    # Computing scaling

    # First step
    # Lets find the w and k, such that
    # w[j]*upper_bound[j]+k[j] = 1
    # w[j]*lower_bound[j]+k[j] = -1
    w = 2/(upper_bound-lower_bound)
    k = -(upper_bound+lower_bound)/(upper_bound-lower_bound)

    # The relative bias ``b`` is the sum of the contribution
    # of each input displacement ``k`` to the bias
    b0 = np.sum(k)

    # It will be the same for every node
    nfirstlayer = weight0.shape[0]
    b = np.repeat(b0, nfirstlayer)

    # At last, lets guarantee it has the right shape
    b = b.reshape(bias0.shape)

    # Apply scaling on each column of weight matrix
    new_weights = weights
    new_weights[0] = weight0*w

    # Add factor to bias vector
    new_bias = bias
    new_bias[0] = bias0 + b

    return new_weights, new_bias


def scale_output_layer(weights, bias, upper_bound, lower_bound):
    """
    Considering that the network output is bounded by upper and
    lower bounds, scale weights matrix and bias vector.

    Parameters
    ----------
    weights : list of array_like
        List containing one weight matrix per layer.
    bias : list of array_like
        List containing one bias vector per layer.
    upper_bound, lower_bound : array_like
        List containing upper and lower bounds for each input

    Returns
    -------
    new_weights : list of array_like
        New list containing weights.
    new_bias : list of array_like
        New list containing bias.

    Notes
    -----
    Consider ``W = new_weights[-1]/weights[-1]`` and
    ``b = new_bias[-1]-bias[-1]``, the following formula
    does hold:

        ``W[i, j]*1+b[i] = upper_bound[i]``
        ``W[i, j]*(-1)+b[i] = lower_bound[i]``
    """
    # Get last layer weights and bias
    weight_end = np.atleast_2d(weights[-1])
    bias_end = np.atleast_1d(bias[-1])

    # Reshape output
    upper_bound = np.atleast_1d(upper_bound)
    lower_bound = np.atleast_1d(lower_bound)

    # Computing scaling

    # First step
    # Lets find the w and b, such that
    # w[i]*1+b[i] = upper_bound[i]
    # w[i]*(-1)+b[i] = lower_bound[i]
    w = (upper_bound-lower_bound)/2
    b = (upper_bound+lower_bound)/2

    # Lets guarantee the bias has the right shape
    b = b.reshape(bias_end.shape)

    # Apply scaling on each line of weight matrix
    new_weights = weights
    new_weights[-1] = (weight_end.T*w).T

    # Add factor to bias vector
    new_bias = bias
    new_bias[-1] = bias_end + b

    return new_weights, new_bias
