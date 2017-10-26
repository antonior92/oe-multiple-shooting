"""
One-step-ahead prediction.
"""
from __future__ import division, print_function, absolute_import
import numpy as np

__all__ = [
    'predict',
    'predict_derivatives'
]


def predict(mdl, y, u, params):
    """
    Compute one-step-ahead sequence for a given
    set of measured input and outputs.

    Parameters
    ----------
    mdl : NarxModel
        NARX model object.
    y : array_like
        Measured output vector. It should
        have dimension (Nd, Ny).
    u : array_like
        Measured input vector. It should
        have dimension (Nd, Nu).
    params : array_like
        Parameter array. It should
        have dimension (Nparams,).

    Returns
    -------
    y1 : array_like
        Array containing one-step-ahead predictions.
        Dimension (Nd - max(M, N) + 1, Ny).
    """

    # Check input
    mdl, y, u, params = _predict_arg_check(mdl, y, u, params)

    # Define dimensions
    maxMN = max(mdl.M, mdl.N)
    Nd = y.shape[0]

    # Prediction
    y1 = np.zeros([Nd - maxMN + 1, mdl.Ny])
    for i in range(Nd - maxMN + 1):
        n = i + maxMN
        yvec = y[n-mdl.N:n, :][::-1, :]
        uvec = u[n-mdl.M:n-mdl.delay+1, :][::-1, :]
        y1[i, :] = mdl.__call__(yvec, uvec, params)

    return y1


def predict_derivatives(mdl, y, u, params):
    """
    Compute the derivatives of one-step-ahead sequence
    in relation to the parameter vector ``params``.

    Parameters
    ----------
    mdl : NarxModel
        NARX model object.
    y : array_like
        Measured output vector. It should
        have dimension (Nd, Ny).
    u : array_like
        Measured input vector. It should
        have dimension (Nd, Nu).
    params : array_like
        Parameter array. It should
        have dimension (Nparams,).

    Returns
    -------
    jac : array_like
        Multidimensional jacobian matrix. Array
        containing one-step-ahead prediction
        derivatives in relation to ``params``.
        Dimension (Nd - max(M, N) + 1, Ny, Nparams).
    """

    # Check input
    mdl, y, u, params = _predict_arg_check(mdl, y, u, params)

    # Define dimensions
    maxMN = max(mdl.M, mdl.N)
    Nd = y.shape[0]

    # Compute derivatives
    jac = np.zeros([Nd - maxMN + 1, mdl.Ny, mdl.Nparams])
    for i in range(Nd - maxMN + 1):
        n = i + maxMN
        yvec = y[n-mdl.N:n, :][::-1, :]
        uvec = u[n-mdl.M:n-mdl.delay+1, :][::-1, :]
        jac[i, :, :] = mdl.derivatives(yvec, uvec, params, deriv_y=False,
                                          deriv_u=False, deriv_params=True)

    return jac


def _predict_arg_check(mdl, y, u, params):
    """
    Check input arguments.
    """

    params = np.atleast_1d(params)
    y = np.atleast_2d(y)
    u = np.atleast_2d(u)

    if params.shape != (mdl.Nparams,):
        raise ValueError("Wrong params vector size.")
    Ndy, Ny = y.shape
    Ndu, Nu = u.shape
    if Nu != mdl.Nu:
        raise ValueError("Wrong u vector size.")
    if Ny != mdl.Ny:
        raise ValueError("Wrong y vector size.")
    if Ndu < max(mdl.M, mdl.N):
        raise ValueError("Nd should be greater than M")
    if Ndu != Ndy:
        raise ValueError("Vector y and u should have the same \
                         dimensions along axis 0")

    return mdl, y, u, params
