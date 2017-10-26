"""
Simulation Error.
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from .simulate import simulate, simulate_derivatives
from ..error import ErrorFunction

__all__ = [
    'SimulationError',
    'generate_simulation_intervals',
    'initial_conditions_from_data',
    'assemble_extended_params',
    'disassemble_extended_params'
]


class SimulationError(ErrorFunction):
    """
    Compute model simulation error for
    a given set of measured inputs, outputs
    and a parameter vector.

    Init Parameters
    ---------------
    mdl : NarxModel
        NARX model for which the simulation error
        will be computed
    y : array_like
        Measured output vector. It should
        have dimension (Nd, Ny).
    u : array_like
        Measured input vector. It should
        have dimension (Nd, Nu).
    maxlength : int, optional
        Maximum length of simulation intervals.
        The data will be break into intervals of
        this length os smaller. By default, it is 20.
    penalty_weigth : float, optional
        Coeficient that weights the importace of the coherence
        between diferent simulations intervals. By defaut,
        it is 100000.

    Call Parameters
    ---------------
    extended_params : array_like
        Parameter array containing the concatenation of the
        parameters and the initial conditions for each
        of the ``Nshoots`` simulation intervals. It should
        have dimension (Nparams+N*Ny*Nshoots,).

    Call Returns
    ------------
    error : array_like
        Unidimensional array containing simulation
        errors. The array dimension
        is ((Nd - max(M, N))*Ny+N*Ny*(Nshoots-1),).
    """

    def __init__(self, mdl, y, u, maxlength=20, penalty_weigth=100000):

        # Check Input
        self.mdl, self.y, self.u, self.maxlength, self.penalty_weigth \
            = self._arg_check(mdl, y, u, maxlength, penalty_weigth)

        # Get data length
        self.Nd = self.y.shape[0]

        # Generate list of tuples containing the begining and
        # and the end of each interval
        self.multiple_shoots = generate_simulation_intervals(self.Nd,
                                                             self.maxlength,
                                                             max(self.mdl.M,
                                                                 self.mdl.N))
        self.Nshoots = len(self.multiple_shoots)

    def __call__(self, extended_params):

        extended_params = np.atleast_1d(extended_params)

        # Auxiliar variables
        Nparams = self.mdl.Nparams
        N = self.mdl.N
        M = self.mdl.M
        Ny = self.mdl.Ny

        # Check extended parameter shape
        if extended_params.shape != (self.mdl.Nparams+N*Ny*self.Nshoots,):
            raise ValueError("Wrong params vector size.")

        for i, interval in enumerate(self.multiple_shoots):

            # Get y and u for this shoot
            y_shoot = self.y[interval[0]:interval[1], :]
            u_shoot = self.u[interval[0]:interval[1], :]
            y0_shoot \
                = extended_params[(Nparams+i*N*Ny):
                                  (Nparams+(i+1)*N*Ny)].reshape(N, Ny)

            # Remove first elements from u_shoot if M < N
            if M < N:
                u_shoot = u_shoot[N-M:]

            # Compute the error on the overlap between two shoots
            if i != 0:
                error_overlap = (ys[-N:, :]-y0_shoot) * \
                                np.sqrt(self.penalty_weigth)
                error_overlap = error_overlap.flatten()

            # Free run simulation
            ys = simulate(self.mdl, y0_shoot, u_shoot,
                          extended_params[:Nparams])

            # Get right dimensions
            ys = ys[:-1, :]
            error_size = ys.shape[0]
            error = ys - y_shoot[(-1*error_size):, :]

            # Reshape error
            error = error.flatten()

            # Stack it
            if i == 0:
                error_multiple_shoots = error
            else:
                error_multiple_shoots = np.hstack((error_multiple_shoots,
                                                   error_overlap))
                error_multiple_shoots = np.hstack((error_multiple_shoots,
                                                   error))

        return error_multiple_shoots

    def derivatives(self, extended_params, sparse=False):
        """
        Compute the derivatives of the simulation error
        in relation to the extended parameter vector.

        Parameters
        ----------
        extended_params : array_like
            Parameter array containing the concatenation of the
            parameters and the initial conditions for each
            of the ``Nshoots`` simulation intervals. It should
            have dimension (Nparams+N*Ny*Nshoots,).
        sparse : boolean, optional
            If True return a sparse CSR matrix, otherwise return
            a numpy array. By default, False.

        Returns
        -------
        jac : sparse
            Sparse matrix containing sequence derivatives in relation to params.
            Dimension:
            ((Nd-max(M, N))*Ny+N*Ny*(Nshoots-1), Nparams+N*Ny*Nshoots).
        """

        extended_params = np.atleast_1d(extended_params)

        # Auxiliar variables
        Nparams = self.mdl.Nparams
        N = self.mdl.N
        M = self.mdl.M
        maxMN = max(M, N)
        Ny = self.mdl.Ny
        Nshoots = self.Nshoots
        Nd = self.Nd

        # Check extended parameter shape
        if extended_params.shape != (Nparams+N*Ny*self.Nshoots,):
            raise ValueError("Wrong params vector size.")

        # Create sparse matrix
        jac = lil_matrix(((Nd-maxMN)*Ny+N*Ny*(Nshoots-1),
                          Nparams+N*Ny*Nshoots))

        # Row index for asign values
        rstart = 0

        # Jacobian overlap block matrix number of rows
        jac_overlap_nrows = N*Ny

        for i, interval in enumerate(self.multiple_shoots):

            # Jacobian one shoot block matrix number of rows
            jac_nrows = (interval[1]-interval[0] - maxMN)*Ny

            # Get u for this shoot
            u_shoot = self.u[interval[0]:interval[1], :]

            # Get initial conditions
            y0_shoot \
                = extended_params[(Nparams+i*N*Ny):
                                  (Nparams+(i+1)*N*Ny)].reshape(N, Ny)

            # Remove first elements from u_shoot if M < N
            if M < N:
                u_shoot = u_shoot[N-M:]

            # Compute the error on the overlap between two shoots
            if i != 0:

                # Define end of block matrix
                rend = rstart+jac_overlap_nrows

                # Assemble overlap block matrix
                jac[rstart:rend, :Nparams] \
                    = np.sqrt(self.penalty_weigth) * \
                    dparams[-jac_overlap_nrows:, :]

                jac[rstart:rend, (Nparams+(i-1)*N*Ny):(Nparams+i*N*Ny)] \
                    = np.sqrt(self.penalty_weigth) * \
                    dy0[-jac_overlap_nrows:, :]

                jac[rstart:rend, (Nparams+i*N*Ny):(Nparams+(i+1)*N*Ny)] \
                    = -1*np.sqrt(self.penalty_weigth) * np.eye(N*Ny)

                # Advance in rows
                rstart = rend

            # Free run simulation
            dparams, dy0 \
                = simulate_derivatives(self.mdl, y0_shoot, u_shoot,
                                       extended_params[:Nparams])

            # Define end of block matrix
            rend = rstart+jac_nrows

            # Remove last term and change dimension
            dparams = dparams[:-1, :, :].reshape((-1, Nparams))
            dy0 = dy0[:-1, :, :, :].reshape((-1, N*Ny))

            # Assemble matrix
            jac[rstart:rend, :Nparams] = dparams
            jac[rstart:rend, (Nparams+i*N*Ny):(Nparams+(i+1)*N*Ny)] \
                = dy0

            # advance in rows
            rstart = rend

        if sparse:
            return jac.tocsr()
        else:
            return jac.toarray()

    def _arg_check(self, mdl, y, u, length, penalty_weigth):
        """
        Check input arguments.
        """

        y = np.atleast_2d(y)
        u = np.atleast_2d(u)

        Ndy, Ny = y.shape
        Ndu, Nu = u.shape
        Nd = Ndu
        if Nu != mdl.Nu:
            raise ValueError("Wrong u vector size.")
        if Ny != mdl.Ny:
            raise ValueError("Wrong y vector size.")
        if Ndu < max(mdl.M, mdl.N):
            raise ValueError("Nd should be greater than M")
        if Ndu != Ndy:
            raise ValueError("Vector y and u should have the same \
                             dimensions along axis 0")

        # Check if length is larger than Nd
        if length > Nd:
            length = Nd

        # Return
        return mdl, y, u, length, penalty_weigth

    def lsq_estimate_parameters(self, initial_guess, use_jacobian=True,
                                use_sparse=True, *args, **kwargs):

        initial_conditions \
            = initial_conditions_from_data(self.y, self.multiple_shoots,
                                           self.mdl.N,
                                           max(self.mdl.M, self.mdl.N))

        extended_initial_guess \
            = assemble_extended_params(initial_guess, initial_conditions)

        if use_jacobian:
            if use_sparse:
                result = least_squares(self.__call__, extended_initial_guess,
                                       lambda x: self.derivatives(x, sparse=True),
                                       *args, **kwargs)
            else:
                result = least_squares(self.__call__, extended_initial_guess,
                                       self.derivatives,
                                       *args, **kwargs)
        else:
            result = least_squares(self.__call__, extended_initial_guess,
                                   "2-point", *args, **kwargs)

        extended_params = result["x"]
        info = result

        params, optimized_initial_conditions \
            = disassemble_extended_params(extended_params, self.mdl.Nparams,
                                          self.mdl.N, self.mdl.Ny,
                                          self.Nshoots)

        info["multiple_shoots"] = self.multiple_shoots
        info["initial_conditions"] = optimized_initial_conditions

        return params, info


def generate_simulation_intervals(Nd, maxlength, overlap):
    """
    Generate a list of tuples containing the begining
    and the end of any simulation intervals.

    Parameters
    ----------
    Nd : int
        Number of samples from the data set.
    maxlength : int
        Maximum length of simulation intervals.
        The data will be break into intervals of
        this length os smaller.
    overlap : int
        Overlap between two consecutives intervals.

    Returns
    -------
    multiple_shoots : list of tuples
        List of tuples containing the begining
        and the end of simulation intervals.
    """

    Nshoots = Nd // maxlength
    if Nd % maxlength != 0:
        Nshoots += 1

        # Compute actual simulation lenght
        simulation_length \
            = maxlength - \
            (maxlength - Nd % maxlength) // Nshoots

        # Compute a second simulation length and the number of times
        # this diferent length will be applied
        diferent_length = simulation_length-1
        if Nd % simulation_length == 0:
            Nshoots_diferent_length = 0
        else:
            Nshoots_diferent_length \
                = simulation_length - Nd % simulation_length

    else:
        Nshoots_diferent_length = 0
        diferent_length = maxlength
        simulation_length = maxlength

    # Check if diferent_length is large enough
    if diferent_length < overlap:
            raise ValueError("length too small.")

    # Generate list
    multiple_shoots = []
    firstvalue = 0
    for i in range(Nshoots - Nshoots_diferent_length):
        lastvalue = firstvalue+simulation_length + \
                    overlap
        multiple_shoots += [(firstvalue, min(lastvalue, Nd))]
        firstvalue += simulation_length

    for i in range(Nshoots_diferent_length):
        lastvalue = firstvalue + diferent_length + \
                    overlap
        multiple_shoots += [(firstvalue, min(lastvalue, Nd))]
        firstvalue += diferent_length

    return multiple_shoots


def initial_conditions_from_data(y, multiple_shoots, N, overlap):
    """
    For a data set subdivided in multiple intervals, get
    first values of each subinterval to be used as
    initial conditions.

    Parameters
    ----------
    y : array_like
        Array containing measured data from which
        the initial conditions will be extracted.
        It should have dimension (Nd, Ny).
    multiple_shoots : list of tuples
        List of tuples containing the begining
        and the end of simulation intervals.
        The list should have length Nshoots
    N : int
        Order of the NARX model being considered
    overlap : int
        Overlap between two consecutives intervals.

    Returns
    -------
    initial_conditions : array_like
        Array containing one set of initial conditions
        per simulation intervals. Dimension
        (Nshoots, N, Ny)
    """

    # Check input
    y = np.atleast_2d(y)

    # Get dimensions from input
    Nd, Ny = y.shape
    Nshoots = len(multiple_shoots)

    # Initialize matrix
    initial_conditions = np.zeros((Nshoots, N, Ny))

    # Get initial conditions from data
    for i, interval in enumerate(multiple_shoots):

        y_shoot = y[interval[0]:interval[1]]
        initial_conditions[i, :, :] = y_shoot[(overlap-N):overlap]

    return initial_conditions


def assemble_extended_params(params, initial_conditions):
    """
    Given a set of model parameters and a set of initial
    initial conditions assemble an extended parameter
    vector.

    Parameters
    ----------
    params : array_like
        Array containing model parameters. It should
        have dimension (Nparams,).
    initial_conditions : array_like
        Array containing one set of initial conditions
        per simulation intervals. Dimension
        (Nshoots, N, Ny)

    Returns
    -------
    extended_params : array_like
        Parameter array containing the concatenation of the
        parameters and the initial conditions for each
        of the ``Nshoots`` simulation intervals. It should
        have dimension (Nparams+N*Ny*Nshoots,).
    """
    # Get right dimension
    params = np.atleast_1d(params)
    initial_conditions = np.atleast_3d(initial_conditions)

    # Assemble extended_params
    return np.hstack((params, initial_conditions.flatten()))


def disassemble_extended_params(extended_params, Nparams, N, Ny, Nshoots):
    """
    Given a set of model parameters and a set of initial
    initial conditions assemble an extended parameter
    vector.

    Parameters
    ----------
    extended_params : array_like
        Parameter array containing the concatenation of the
        parameters and the initial conditions for each
        of the ``Nshoots`` simulation intervals. It should
        have dimension (Nparams+N*Ny*Nshoots,).
    Nparams : int
        Number of parameters
    N : int
        Model order
    Ny : int
        Model number of outputs
    Nshoots : int
        Number of simulations intervals.


    Returns
    -------
    params : array_like
        Array containing model parameters. It should
        have dimension (Nparams,).
    initial_conditions : array_like
        Array containing one set of initial conditions
        per simulation intervals. Dimension
        (Nshoots, N, Ny)
    """
    # Get right dimension
    extended_params = np.atleast_1d(extended_params)

    # Disassemble extended_params
    params = extended_params[:Nparams]
    initial_conditions = extended_params[Nparams:].reshape(Nshoots, N, Ny)
    return params, initial_conditions
