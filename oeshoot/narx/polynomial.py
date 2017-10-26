"""
Polynomial NARX model.
"""

from __future__ import division, print_function, absolute_import
import numpy as np
from .narx import NarxModel


__all__ = [
    'Polynomial',
    'Monomial'
]


class Polynomial(NarxModel):
    """
    Polynomial NARX model.
    ``y[n] = F(y[n-1],..., y[n-N], u[n-delay],..., u[n-M], params)``

    Init Parameters
    ----------
    struct : list of Monomials or list of list of Monomials
        List of all monomials contained in the polynomial model.
        For a single-output model this struct should be a list of
        Monomials. For a multiple-output model this struct should
        be a list contain a list of monomials for each output.

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
    Nl : int
        Degree of nonlinearity. Polynomial degree for
        the single-output case and maximum polynomial degree
        for the multiple-output case.
    struct : list of Monomials or list of list of Monomials
        List of all monomials contained in the polynomial model.
        For a single-output model this struct should be a list of
        Monomials. For a multiple-output model this struct should
        be a list contain a list of monomials for each output.

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

    def __init__(self, struct):

        # Check type of struc elements and evaluate Ny
        if all(isinstance(n, Monomial) for n in struct):
            Ny = 1
        elif all(all(isinstance(n, Monomial) for n in s) for s in struct):
            Ny = len(struct)
        else:
            raise TypeError("struc should be a list of Monomials or a \
                             list of list of Monomials")

        # Get parameters from monomials
        if Ny == 1:
            Nparams = len(struct)
            N = max(monomial.N for monomial in struct)
            Nu = max(monomial.Nu for monomial in struct)
            Ny_monomial = max(monomial.Ny for monomial in struct)
            M = max(monomial.M for monomial in struct)
            delay = min(monomial.minlag_u for monomial in struct)

        else:
            Nparams = np.sum(len(s) for s in struct)
            N = max(max(monomial.N for monomial in s) for s in struct)
            Nu = max(max(monomial.Nu for monomial in s) for s in struct)
            Ny_monomial = max(max(monomial.Ny for monomial in s)
                              for s in struct)
            M = max(max(monomial.M for monomial in s) for s in struct)
            delay = min(min(monomial.minlag_u for monomial in s)
                        for s in struct)

        # Check for incoherence on the output monomial terms.
        if Ny < Ny_monomial:
            raise ValueError("Monomial contain undefined output term.")

        # Save struct
        self.struct = struct

        # Call superclass constructor
        NarxModel.__init__(self, Nparams, N, M, Ny, Nu, delay)

    def __call__(self, y, u, params):

        # Check inputs
        y, u, params = self._arg_check(y, u, params)

        # Compute next input
        if self.Ny == 1:
            ynext = np.atleast_1d(np.sum(params[i]*self.struct[i](y, u)
                                         for i in range(len(self.struct))))
        else:
            ynext = np.zeros(self.Ny)
            l = 0
            for j in range(len(self.struct)):
                ynext[j] = np.sum(params[l+i]*self.struct[j][i](y, u)
                                  for i in range(len(self.struct[j])))
                l += len(self.struct[j])

        return ynext

    def derivatives(self, y, u, params, deriv_y=True,
                    deriv_u=True, deriv_params=True):

        # Check inputs
        y, u, params = self._arg_check(y, u, params)

        returns = []
        # Compute dy
        if deriv_y:
            dy = np.zeros((self.Ny, self.N, self.Ny))
            if self.Ny == 1:
                for i in range(len(self.struct)):
                    dy[0, :, :] += params[i] * \
                                   self.struct[i].derivatives(y, u,
                                                              deriv_y=True,
                                                              deriv_u=False)
            else:
                l = 0
                for j in range(len(self.struct)):
                    for i in range(len(self.struct[j])):
                        dy[j, :, :] += params[l+i] * \
                                   self.struct[j][i].derivatives(y, u,
                                                                 deriv_y=True,
                                                                 deriv_u=False)
                    l += len(self.struct[j])

            # Append return value
            returns.append(dy)

        # Compute du
        if deriv_u:
            du = np.zeros((self.Ny, self.Mu, self.Nu))
            if self.Ny == 1:
                for i in range(len(self.struct)):
                    du[0, :, :] += params[i] * \
                                   self.struct[i].derivatives(y, u,
                                                              deriv_u=True,
                                                              deriv_y=False)
            else:
                l = 0
                for j in range(len(self.struct)):
                    for i in range(len(self.struct[j])):
                        du[j, :, :] += params[l+i] * \
                                   self.struct[j][i].derivatives(y, u,
                                                                 deriv_u=True,
                                                                 deriv_y=False)
                    l += len(self.struct[j])

            # Append return value
            returns.append(du)

        # Compute dparams
        if deriv_params:
            dparams = np.zeros((self.Ny, self.Nparams))
            l = 0
            if self.Ny == 1:
                for i in range(len(self.struct)):
                    dparams[0, l] = self.struct[i](y, u)
                    l += 1
            else:
                for j in range(len(self.struct)):
                    for i in range(len(self.struct[j])):
                        dparams[j, l] = self.struct[j][i](y, u)
                        l += 1

            # Append return value
            returns.append(dparams)

        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)


class Monomial(object):
    """
    Monomial containing input and output terms.

    Init Parameters
    ----------
    nterms_in_y : int
        Number of output terms in the monomial.
    nterms_in_u : int
        Number of input terms in the monomial.
    lags_y : array_like of ints
        List of lags for each one of the ``y``
        terms in the monomial.
    lags_u : array_like of ints
        List of lags for each one of the ``u``
        terms in the monomial.
    expn_y : array_like of ints
        List of exponents for each one of the ``y``
        terms in the monomial. By default
        consider that all ``y`` terms are
        raised to the power of 1.
    expn_u : array_like of ints
        List of exponents for each one of the ``u``
        terms in the monomial. By default
        consider that all ``u`` terms are
        raised to the power of 1.
    number_y : array_like of ints, optional
        List containing the output number correspondent
        to each one of the ``y`` terms. By default
        consider that all ``y`` terms refer to
        output 0 (which is the case for single-input
        models).
    number_u : array_like of ints, optional
        List containing the input number correspondent
        to each one of the terms. By default
        consider that all ``u`` terms  refer to
        input 0 (which is the case for single-output
        models)

    Atributes
    ---------
    N : int
        Maximum lag of past output.
    M : int
        Maximum lag of past input.
    Ny : int
        Number of outputs.
    Nu : int
        Number of inputs.
    minlag_u : int
        Minimum input lag.
    Nl : int
        Monomial degree.
    nterms_in_y : int
        Number of output terms in the monomial.
    nterms_in_u : int
        Number of input terms in the monomial.
    lags_y : array_like of ints
        List of lags for each one of the ``y``
        terms in the monomial.
    lags_u : array_like of ints
        List of lags for each one of the ``u``
        terms in the monomial.
    expn_y : array_like of ints
        List of exponents for each one of the ``y``
        terms in the monomial.
    expn_u : array_like of ints
        List of exponents for each one of the ``u``
        terms in the monomial.
    number_y : array_like of ints, optional
        List containing the output number correspondent
        to each one of the ``y`` terms. 
    number_u : array_like of ints, optional
        List containing the input number correspondent
        to each one of the terms. 

    Call Parameters
    ---------------
    y : array_like
        Array containing previous system outputs.
        Its dimension should be (N, Ny), extra
        elements  will be ignored.
    u : array_like
        Array containing system inputs.
        Its dimension should be (M-delay+1, Nu),
        extra elements  will be ignored.
    """

    def __init__(self, nterms_in_y, nterms_in_u,
                 lags_y, lags_u, expn_y=None, expn_u=None,
                 number_y=None, number_u=None):

        # Check parameters
        if len(lags_y) != nterms_in_y:
            raise ValueError("lags_y should have one element per y term.")
        if len(lags_u) != nterms_in_u:
            raise ValueError("lags_u should have one element per u term.")

        # Get number of dynamic terms
        self.N = max(lags_y) if nterms_in_y != 0 else 0

        self.M = max(lags_u) if nterms_in_u != 0 else 0
        self.minlag_u = min(lags_u) if nterms_in_u != 0 else np.Inf

        # Get number of input/outputs
        if number_y is None:
            number_y = [0]*nterms_in_y
            self.Ny = 1 if nterms_in_y != 0 else 0
        else:
            self.Ny = max(number_y)+1 if nterms_in_y != 0 else 0
            if len(number_y) != nterms_in_y:
                raise ValueError("number_y should have one \
                                  element per y term.")
        if number_u is None:
            number_u = [0]*nterms_in_u
            self.Nu = 1 if nterms_in_u != 0 else 0
        else:
            self.Nu = max(number_u)+1 if nterms_in_u != 0 else 0

            if len(number_u) != nterms_in_u:
                raise ValueError("number_u should have one \
                                  element per u term.")

        # Get degree of each term
        if expn_y is None:
            expn_y = [1]*nterms_in_y
        elif len(expn_y) != nterms_in_y:
            raise ValueError("expn_y should have one element per y term.")
        if expn_u is None:
            expn_u = [1]*nterms_in_u
        elif len(expn_u) != nterms_in_u:
            raise ValueError("expn_u should have one element per u term.")

        # Get monomial degree
        self.Nl = sum(expn_u)+sum(expn_y)

        # Save monomial struct
        self.nterms_in_y = nterms_in_y
        self.nterms_in_u = nterms_in_u
        self.lags_y = np.asarray(lags_y)
        self.lags_u = np.asarray(lags_u)
        self.expn_y = np.asarray(expn_y)
        self.expn_u = np.asarray(expn_u)
        self.number_y = np.asarray(number_y)
        self.number_u = np.asarray(number_u)

    def _arg_check(self, y, u, delay):
        y = np.atleast_2d(y)
        u = np.atleast_2d(u)
        return y, u

    def __call__(self, y, u, delay=1):
        """
        Evaluate monomial for a given set of input and outputs.

        Parameters
        ----------
        y : array_like
            Array containing previous system outputs.
            Its dimension should be (N, Ny), extra
            elements  will be ignored.
        u : array_like
            Array containing system inputs.
            Its dimension should be (M-delay+1, Nu),
            extra elements  will be ignored.
        delay : int, optional
            Model output delay. This parameter dictates
            ``u`` dimensions. By default ``delay=1``.

        Returns
        -------
        psi : float
            Monomial evaluation.
        """

        # Check Inputs
        y, u = self._arg_check(y, u, delay)

        # Compute monomial
        prod = 1
        for i in range(self.nterms_in_y):
            prod *= np.power(y[self.lags_y[i]-1, self.number_y[i]],
                             self.expn_y[i])
        for j in range(self.nterms_in_u):
            prod *= np.power(u[self.lags_u[j]-delay,
                               self.number_u[j]],
                             self.expn_u[j])

        return prod

    def derivatives(self, y, u, delay=1,
                    deriv_y=True, deriv_u=True):
        """
        Evaluate monomial first derivative for a given set of
        input and outputs.

        Parameters
        ----------
        y : array_like
            Array containing previous system outputs.
            Its dimension should be (N, Ny), extra
            elements  will be ignored.
        u : array_like
            Array containing system inputs.
            Its dimension should be (M-delay+1, Nu),
            extra elements  will be ignored.
        delay : int, optional
            Model output delay. This parameter dictates
            ``u`` dimensions. By default ``delay=1``.
        deriv_y : boolean, optional
            Specify if the derivatives in relation to y
            should be returned by this function. By default,
            it is True.
        deriv_u : boolean, optional
            Specify if the derivatives in relation to u
            should be returned by this function. By default,
            it is False.

        Returns
        -------
        dy : array_like, optional
            Array containing monomial derivatives in relation
            to y. Same dimension as input ``y``.
        du : array_like, optional
            Array containing monomial derivatives in relation
            to u. Same dimension as input ``u``.
        """

        # Check Inputs
        y, u = self._arg_check(y, u, delay)

        if deriv_y:
            # Initialize array
            dy = np.zeros_like(y, dtype=np.float64)

            for n in range(self.nterms_in_y):
                # Compute monomial derivative
                prod = 1
                for i in range(self.nterms_in_y):
                    if i != n:
                        prod *= np.power(y[self.lags_y[i]-1,
                                           self.number_y[i]],
                                         self.expn_y[i])
                    else:
                        prod *= self.expn_y[i]*np.power(y[self.lags_y[i]-1,
                                                          self.number_y[i]],
                                                        self.expn_y[i]-1)
                for j in range(self.nterms_in_u):
                    prod *= np.power(u[self.lags_u[j]-delay,
                                       self.number_u[j]],
                                     self.expn_u[j])

                # Store it on the array
                dy[self.lags_y[n]-1, self.number_y[n]] = prod

        if deriv_u:
            du = np.zeros_like(u, dtype=np.float64)

            for n in range(self.nterms_in_u):
                # Compute monomial derivative
                prod = 1
                for i in range(self.nterms_in_y):
                    prod *= np.power(y[self.lags_y[i]-1,
                                       self.number_y[i]],
                                     self.expn_y[i])
                for j in range(self.nterms_in_u):
                    if j != n:
                        prod *= np.power(u[self.lags_u[j]-delay,
                                           self.number_u[j]],
                                         self.expn_u[j])
                    else:
                        prod *= self.expn_u[j]*np.power(u[self.lags_u[j]-delay,
                                                          self.number_u[j]],
                                                        self.expn_u[j]-1)

                # Store it on the array
                du[self.lags_u[n]-delay, self.number_u[n]] = prod

        # Returns
        if deriv_y and deriv_u:
            return dy, du
        elif deriv_y:
            return dy
        else:
            return du
