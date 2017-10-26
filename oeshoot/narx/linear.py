"""
Linear autoregressive model with exogenous inputs.
"""

from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.linalg import block_diag
from .narx import NarxModel


__all__ = [
    'Linear'
]


class Linear(NarxModel):
    """
    Create linear autoregressive model with exogenous inputs (ARX).
    ``y[n] = a1*y[n-1]+...+a_N*y[n-N]+b_1*u[n-delay]+...+b_M*u[n-M]``

    Init Parameters
    ---------------
    N : int
        Number of past output taken in account by the ARX model.
    M : int
        Number of past input taken in account by the ARX model.
    Ny : int, optional
        Number of outputs. By default ``Ny = 1``.
    Nu : int, optional
        Number of inputs. By default ``Nu = 1``.
    delay : int, optional
        System output delay. By default ``delay = 1``.

    Atributes
    ---------
    N : int
        Maximum lag of past output.
    M : int
        Maximum lag of past input.
    delay : int
        Output delay.
    Ny : int
        Number of outputs.
    Nu : int
        Number of inputs.
    Nparams : int
        Number of Parameters.

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

    def __init__(self, N, M, Ny=1, Nu=1, delay=1):
        Nparams = Ny*(N*Ny+(M-delay+1)*Nu)
        NarxModel.__init__(self, Nparams, N, M, Ny, Nu, delay)

    def __call__(self, y, u, params):

        # Check inputs
        y, u, params = self._arg_check(y, u, params)

        # Compute next input
        y = np.reshape(y, (self.N*self.Ny, 1))
        u = np.reshape(u, (self.Mu*self.Nu, 1))
        params_y = np.reshape(params[:self.N*self.Ny**2],
                              (self.Ny, self.N*self.Ny))
        params_u = np.reshape(params[self.N*self.Ny**2:],
                              (self.Ny, self.Mu*self.Nu))
        ynext = np.dot(params_y, y)+np.dot(params_u, u)

        # Guarantee right ynext dimension
        ynext = ynext.flatten()

        return ynext

    def derivatives(self, y, u, params, deriv_y=True,
                    deriv_u=True, deriv_params=True):

        # Check inputs
        y, u, params = self._arg_check(y, u, params)

        returns = []
        # Compute dy
        if deriv_y:
            dy = np.reshape(params[:self.N*self.Ny**2],
                            (self.Ny, self.N, self.Ny))
            returns.append(dy)

        # Compute du
        if deriv_u:
            du = np.reshape(params[self.N*self.Ny**2:],
                            (self.Ny, self.Mu, self.Nu))
            returns.append(du)

        # Compute dparams
        if deriv_params:
            # assemble y-related and u-related params derivatives
            yrep = [np.reshape(y, (self.N*self.Ny,))]*self.Ny
            urep = [np.reshape(u, (self.Mu*self.Nu,))]*self.Ny
            dparams_y = block_diag(*yrep)
            dparams_u = block_diag(*urep)

            # Guarantee right dimension (import when N=0, Mu=0 or  Nu=0)
            dparams_y = dparams_y.reshape((self.Ny, self.Ny*self.N*self.Ny))
            dparams_u = dparams_u.reshape((self.Ny, self.Ny*self.Mu*self.Nu))

            # Concatenate
            dparams = np.concatenate((dparams_y,
                                      dparams_u),
                                     axis=1)
            returns.append(dparams)

        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)
