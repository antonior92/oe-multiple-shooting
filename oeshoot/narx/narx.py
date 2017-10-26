"""
Abstract base class that define basic features for NARX models.
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from abc import (ABCMeta, abstractmethod)
from six import add_metaclass
import numdifftools as nd

__all__ = [
    'NarxModel'
]


@add_metaclass(ABCMeta)
class NarxModel(object):
    """
    Define a NARX model:
    ``y[n] = F(y[n-1],..., y[n-N], u[n-delay],..., u[n-M], params)``
    """

    def __init__(self, Nparams, N, M, Ny=1, Nu=1, delay=1):
        if M != 0 and Nu != 0 and delay > M:
            raise ValueError("delay should be smaller than M.")
        self.Nparams = Nparams
        self.N = N
        self.Ny = Ny
        self.Nu = Nu
        if Nu != 0:
            self.delay = delay
            self.Mu = M-delay+1
            self.M = M
        else:
            self.delay = 0
            self.Mu = 0
            self.M = 0

    @abstractmethod
    def __call__(self, y, u, params):
        """
        Given current system state and inputs compute next state.

        Parameters
        ----------
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
        return

    def derivatives(self, y, u, params, deriv_y=True,
                    deriv_u=True, deriv_params=True):
        """
        Given current system state and inputs compute
        derivatives at a given point

        Parameters
        ----------
        y : array_like
            Array containing previous system outputs.
            It should have dimension (N, Ny).
        u : array_like
            Array containing system inputs.
            it should have dimension (M-delay+1, Nu).
        params : array_like
            Parameter array. It should
            have dimension (Nparams,).
        deriv_y : boolean, optional
            Specify if the derivatives in relation to y
            should be returned by this function. By default,
            it is True.
        deriv_u : boolean, optional
            Specify if the derivatives in relation to u
            should be returned by this function. By default,
            it is True.
        deriv_params : boolean, optional
            Specify if the derivatives in relation to the
            parameters should be returned by this function.
            By default, it is True.

        Returns
        -------
        dy : array_like, optional
            Array containing model derivatives in relation
            to y. Dimension (Ny, N, Ny)
        du : array_like, optional
            Array containing model derivatives in relation
            to u. Dimension (Ny, M-delay+1, Nu)
        dparams : array_like, optional
            Array containing model derivatives in relation
            to params. Dimension (Ny, Nparams)
        """

        return self._numeric_derivatives(y, u, params, deriv_y, deriv_u,
                                         deriv_params)

    def _numeric_derivatives(self, y, u, params, deriv_y=True,
                             deriv_u=True, deriv_params=True):
        """
        Given current system state and inputs compute
        an numeric aproximation of the derivatives at a
        given point
        """

        # Check Arguments
        y, u, params = self._arg_check(y, u, params)

        # Use numdifftools to estimate derivatives
        returns = []
        if deriv_y:
            def fun_y(x):
                return self.__call__(x.reshape(self.N, self.Ny), u, params)
            if self.N != 0:
                dy = nd.Jacobian(fun_y)(y.flatten()).reshape((self.Ny, self.N,
                                                              self.Ny))
            else:
                dy = np.reshape([], (self.Ny, self.N, self.Ny))
            returns.append(dy)

        if deriv_u:
            def fun_u(x):
                return self.__call__(y, x.reshape(self.Mu, self.Nu), params)
            if self.Mu != 0 and self.Nu != 0:
                du = nd.Jacobian(fun_u)(u.flatten()).reshape((self.Ny, self.Mu,
                                                              self.Nu))
            else:
                du = np.reshape([], (self.Ny, self.Mu, self.Nu))
            returns.append(du)

        if deriv_params:
            def fun_params(x):
                return self.__call__(y, u, x)
            dparams = nd.Jacobian(fun_params)(params).reshape((self.Ny,
                                                               self.Nparams))
            returns.append(dparams)

        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)

    def _arg_check(self, y, u, params):
        """
        Check input arguments.
        """

        params = np.atleast_1d(params)
        if params.shape != (self.Nparams,):
            raise ValueError("Wrong params vector size.")
        y = np.reshape(y, (self.N, self.Ny))
        u = np.reshape(u, (self.Mu, self.Nu))

        return y, u, params

    def marshalling_input(self, y, u):
        """
        Generate input for a generic function
        from dinamic data. Usefull when
        the input for a NARX model need to be
        passed on to template functions.

        Parameters
        ----------
        y : array_like
            Array containing previous system outputs.
            It should have dimension (N, Ny).
        u : array_like
            Array containing system inputs.
            it should have dimension (M-delay+1, Nu).

        Returns
        -------
        x : array_like
            Unidimensional vector containing
            concatenation of values from y
            and u.
        """
        y = np.atleast_1d(y)
        u = np.atleast_1d(u)
        x = np.hstack((y.flatten(), u.flatten()))

        return x
