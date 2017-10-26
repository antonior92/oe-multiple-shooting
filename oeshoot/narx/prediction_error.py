"""
Prediction Error.
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from .predict import predict, predict_derivatives
from ..error import ErrorFunction

__all__ = [
    'PredictionError'
]


class PredictionError(ErrorFunction):
    """
    Compute models prediction error for
    a given set of measured input, outputs
    and a parameter vector.

    Init Parameters
    ---------------
    mdl : NarxModel
        NARX model for which the prediction error
        will be computed
    y : array_like
        Measured output vector. It should
        have dimension (Nd, Ny).
    u : array_like
        Measured input vector. It should
        have dimension (Nd, Nu).

    Call Parameters
    ---------------
    params : array_like
        Parameter array. It should
        have dimension (Nparams,).

    Call Returns
    -------
    error : array_like
        Unidimensional array containing prediction
        errors. The array dimension
        is ((Nd - max(M, N))*Ny,).
    """

    def __init__(self, mdl, y, u):

        self.mdl, self.y, self.u = self._arg_check(mdl, y, u)

    def __call__(self, params):

        # Compute one-step-ahead prediction
        y1 = predict(self.mdl, self.y, self.u, params)

        # Remove last element
        y1 = y1[:-1, :]

        # Compute error
        error = y1-self.y[-y1.shape[0]:, :]

        # Reshape it
        error = error.reshape((error.shape[0]*error.shape[1],))

        # Returns
        return error

    def derivatives(self, params):
        """
        Compute the derivatives of one-step-ahead error
        in relation to the parameter vector ``params``.

        Parameters
        ----------
        params : array_like
            Parameter array. It should
            have dimension (Nparams,).

        Returns
        -------
        jac : array_like
            Multidimensional jacobian matrix. Array
            containing prediction error
            derivatives in relation to ``params``.
            Dimension ((Nd - max(M, N))*Ny, Nparams).
        """

        # Compute one-step-ahead derivatives
        jac = predict_derivatives(self.mdl, self.y, self.u, params)

        # Remove last element
        jac = jac[:-1, :]

        # Reshape it
        jac = jac.reshape((jac.shape[0]*jac.shape[1],
                           jac.shape[2]))

        # Returns
        return jac

    def _arg_check(self, mdl, y, u):
        """
        Check input arguments.
        """

        y = np.atleast_2d(y)
        u = np.atleast_2d(u)

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

        return mdl, y, u
