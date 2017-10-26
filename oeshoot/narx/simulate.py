"""
Free-run simulation.
"""
from __future__ import division, print_function, absolute_import
import numpy as np

__all__ = [
    'simulate',
    'simulate_derivatives'
]


def simulate(mdl, y0, u, params):
    """
    Compute free-run simulation for a given
    input sequence.

    Parameters
    ----------
    mdl : NarxModel
        NARX model object.
    y0 : array_like
        Initial Condition. It should
        have dimension (N, Ny).
    u : array_like
        Input Sequence. Dimension (Nd, Nu).
    params : array_like
        Parameter array. It should
        have dimension (Nparams,).

    Returns
    -------
    ys : array_like
        Array containing free-run simulation.
        Dimension (Nd - M + 1, Ny).
    """

    # Check input
    mdl, y0, u, params = _simulate_arg_check(mdl, y0, u, params)

    # Define dimensions
    Nd = u.shape[0]

    # Assemble ys
    ys = np.zeros([Nd - mdl.M + mdl.N + 1, mdl.Ny])
    ys[:mdl.N, :] = y0

    # Compute the output recursively
    for i in range(Nd - mdl.M + 1):
        ny = i + mdl.N
        nu = i + mdl.M
        yvec = ys[ny-mdl.N:ny, :][::-1, :]
        uvec = u[nu-mdl.M:nu-mdl.delay+1, :][::-1, :]
        ys[ny, :] = mdl.__call__(yvec, uvec, params)

    # Returns
    return ys[mdl.N:]


def simulate_derivatives(mdl, y0, u, params,
                         deriv_params=True, deriv_y0=True):
    """
    Compute the derivatives of the free-run simulation
    sequence for a given input sequence.

    Parameters
    ----------
    mdl : NarxModel
        NARX model object.
    y0 : array_like
        Initial Condition. It should
        have dimension (N, Ny).
    u : array_like
        Input Sequence. Dimension (Nd, Nu).
    params : array_like
        Parameter array. It should
        have dimension (Nparams,).
    deriv_params : boolean, optional
        Specify if the derivatives in relation to params
        should be returned by this function. By default,
        it is True.
    deriv_y0 : boolean, optional
        Specify if the derivatives in relation to params
        should be returned by this function. By default,
        it is True.

    Returns
    -------
    jac_params : array_like
        Multidimensional jacobian matrix. Array
        containing free-run simulation sequence
        derivatives in relation to ``params``.
        Dimension (Nd - M + 1, Ny, Nparams).
    jac_y0 : array_like
        Multidimensional jacobian matrix. Array
        containing free-run simulation sequence
        derivatives in relation to ``y0``.
        Dimension (Nd - M + 1, Ny, N, Ny).
    """

    # Check input
    mdl, y0, u, params = _simulate_arg_check(mdl, y0, u, params)

    # Define dimensions
    Nd = u.shape[0]

    # Initialize ys
    ys = np.zeros([Nd - mdl.M + mdl.N + 1, mdl.Ny])
    ys[:mdl.N, :] = y0

    # Initialize jacobian matrix
    if deriv_params:
        jac_params = np.zeros([Nd - mdl.M + 1, mdl.Ny, mdl.Nparams])

    if deriv_y0:
        jac_y0 = np.zeros([Nd - mdl.M + 1, mdl.Ny, mdl.N, mdl.Ny])

    # Compute jacobian
    for n in range(Nd-mdl.M+1):
        ny = n + mdl.N
        nu = n + mdl.M
        yvec = ys[ny-mdl.N:ny, :][::-1, :]
        uvec = u[nu-mdl.M:nu-mdl.delay+1, :][::-1, :]

        ys[ny, :] = mdl.__call__(yvec, uvec, params)
        dy, dparams = mdl.derivatives(yvec, uvec, params,
                                      deriv_u=False)

        if n == 0:
            if deriv_params:
                jac_params[n, :, :] = dparams
            if deriv_y0:
                jac_y0[n, :, :, :] = dy
        elif n < mdl.N:
            if deriv_params:
                sum = np.zeros([mdl.Ny, mdl.Nparams])
                for i in range(n):
                    sum += np.dot(dy[:, i, :], jac_params[n-i-1, :, :])
                jac_params[n, :, :] = dparams+sum
            if deriv_y0:
                sum = np.zeros([mdl.Ny, mdl.N, mdl.Ny])
                for i in range(n):
                    sum += np.tensordot(dy[:, i, :],
                                        jac_y0[n-i-1, :, :, :],
                                        axes=1)
                jac_y0[n, :, :mdl.N-n, :] \
                    = dy[:, n:, :]
                jac_y0[n, :, :, :] += sum
        else:
            if deriv_params:
                sum = np.zeros([mdl.Ny, mdl.Nparams])
                for i in range(mdl.N):
                    sum += np.dot(dy[:, i, :], jac_params[n-i-1, :, :])
                jac_params[n, :, :] = dparams+sum
            if deriv_y0:
                sum = np.zeros([mdl.Ny, mdl.N, mdl.Ny])
                for i in range(mdl.N):
                    sum += np.tensordot(dy[:, i, :],
                                        jac_y0[n-i-1, :, :, :],
                                        axes=1)
                jac_y0[n, :, :, :] += sum

    # Reverse jac_y0 to compensate that y0 is crescent
    # while the paper formula is for decrescent initial
    # conditions
    jac_y0 = jac_y0[:, :, ::-1, :]

    # Returns
    if deriv_params and deriv_y0:
        return jac_params, jac_y0
    elif deriv_params:
        return jac_params
    else:
        return jac_y0


def _simulate_arg_check(mdl, y0, u, params):
    """
    Check input arguments.
    """
    params = np.atleast_1d(params)
    y0 = np.atleast_2d(y0)
    u = np.atleast_2d(u)

    if params.shape != (mdl.Nparams,):
        raise ValueError("Wrong params vector size.")
    if y0.shape != (mdl.N, mdl.Ny):
        raise ValueError("Wrong y0 vector size.")
    Nd, Nu = u.shape
    if Nu != mdl.Nu:
        raise ValueError("Wrong u vector size.")
    if Nd < mdl.M:
        raise ValueError("Nd should be greater than M")

    return mdl, y0, u, params
