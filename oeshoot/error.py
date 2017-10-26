"""
Abstract base class that define error function.
"""

from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import least_squares
from abc import (ABCMeta, abstractmethod)
from six import add_metaclass

@add_metaclass(ABCMeta)
class ErrorFunction(object):
    """
    Define an error function.
    """

    @abstractmethod
    def __call__(self, params):

        return

    @abstractmethod
    def derivatives(self, params):

        return

    def lsq_estimate_parameters(self, initial_guess, use_jacobian=True,
                                *args, **kwargs):
        """
        Minimize error function using non-linear least squares algorithm.

        Given the error ``e(params)`` and the loss function rho(s)
        (a scalar function), `lsq_estimate_parameters` finds a local minimum
        of the cost function:
            minimize  0.5 * sum(rho(e(params)**2), i = 0, ..., m - 1)
            subject to lb <= x <= ub
        The purpose of the loss function rho(s) is to reduce the influence of
        outliers on the solution.

        Parameters
        ----------
        initial_guess: array_like with shape (n,) or float
            Initial guess on independent variables ``params``.
        use_jacobian : boolean, optional
            If use_jacobian is ``False`` the two point aproximation
            of the jacobian will be used, otherwise, if ``True``, the
            exact computed jacobian will be used (default).
        bounds : 2-tuple of array_like, optional
            Lower and upper bounds on independent variables. Defaults to
            no bounds. Each array must match the size of `initial_guess`
            or be a scalar, in the latter case a bound will be the same
            for all variables. Use ``np.inf`` with an appropriate sign
            to disable bounds on all or some variables.
        method : {'trf', 'dogbox', 'lm'}, optional
            Algorithm to perform minimization.
                * 'trf' : Trust Region Reflective algorithm,
                  particularly suitable for large sparse
                  problems with bounds. Generally robust method.
                * 'dogbox' : dogleg algorithm with rectangular
                  trust regions, typical use case is small problems
                  with bounds. Not recommended for problems with
                  rank-deficient Jacobian.
                * 'lm' : Levenberg-Marquardt algorithm as
                  implemented in MINPACK. Doesn't handle bounds
                  and sparse Jacobians. Usually the most
                  efficient method for small unconstrained problems.
            Default is 'trf'. See Notes for more information.
        ftol : float, optional
            Tolerance for termination by the change of
            the cost function. Default is 1e-8. The
            optimization process is stopped when  ``dF < ftol * F``,
            and there was an adequate agreement between a
            local quadratic model and the true model in the last step.
        xtol : float, optional
            Tolerance for termination by the change of the independent variables.
            Default is 1e-8. The exact condition depends on the `method` used:
                * For 'trf' and 'dogbox' : ``norm(dx) < xtol * (xtol + norm(x))``
                * For 'lm' : ``Delta < xtol * norm(xs)``, where ``Delta`` is
                  a trust-region radius and ``xs`` is the value of ``x``
                  scaled according to `x_scale` parameter (see below).
        gtol : float, optional
            Tolerance for termination by the norm of the gradient. Default is 1e-8.
            The exact condition depends on a `method` used:
                * For 'trf' : ``norm(g_scaled, ord=np.inf) < gtol``, where
                  ``g_scaled`` is the value of the gradient scaled to account for
                  the presence of the bounds [STIR]_.
                * For 'dogbox' : ``norm(g_free, ord=np.inf) < gtol``, where
                  ``g_free`` is the gradient with respect to the variables which
                  are not in the optimal state on the boundary.
                * For 'lm' : the maximum absolute value of the cosine of angles
                  between columns of the Jacobian and the residual vector is less
                  than `gtol`, or the residual vector is zero.
        x_scale : array_like or 'jac', optional
            Characteristic scale of each variable. Setting `x_scale` is equivalent
            to reformulating the problem in scaled variables ``xs = x / x_scale``.
            An alternative view is that the size of a trust region along j-th
            dimension is proportional to ``x_scale[j]``. Improved convergence may
            be achieved by setting `x_scale` such that a step of a given size
            along any of the scaled variables has a similar effect on the cost
            function. If set to 'jac', the scale is iteratively updated using the
            inverse norms of the columns of the Jacobian matrix (as described in
            [JJMore]_).
        loss : str or callable, optional
            Determines the loss function. The following keyword values are allowed:
                * 'linear' (default) : ``rho(z) = z``. Gives a standard
                  least-squares problem.
                * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
                  approximation of l1 (absolute value) loss. Usually a good
                  choice for robust least squares.
                * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
                  similarly to 'soft_l1'.
                * 'cauchy' : ``rho(z) = ln(1 + z)``. Severely weakens outliers
                  influence, but may cause difficulties in optimization process.
                * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
                  a single residual, has properties similar to 'cauchy'.
            If callable, it must take a 1-d ndarray ``z=f**2`` and return an
            array_like with shape (3, m) where row 0 contains function values,
            row 1 contains first derivatives and row 2 contains second
            derivatives. Method 'lm' supports only 'linear' loss.
        f_scale : float, optional
            Value of soft margin between inlier and outlier residuals, default
            is 1.0. The loss function is evaluated as follows
            ``rho_(f**2) = C**2 * rho(f**2 / C**2)``, where ``C`` is `f_scale`,
            and ``rho`` is determined by `loss` parameter. This parameter has
            no effect with ``loss='linear'``, but for other `loss` values it is
            of crucial importance.
        max_nfev : None or int, optional
            Maximum number of function evaluations before the termination.
            If None (default), the value is chosen automatically:
                * For 'trf' and 'dogbox' : 100 * n.
                * For 'lm' :  100 * n if `jac` is callable and 100 * n * (n + 1)
                  otherwise (because 'lm' counts function calls in Jacobian
                  estimation).
        diff_step : None or array_like, optional
            Determines the relative step size for the finite difference
            approximation of the Jacobian. The actual step is computed as
            ``x * diff_step``. If None (default), then `diff_step` is taken to be
            a conventional "optimal" power of machine epsilon for the finite
            difference scheme used [NR]_.
        tr_solver : {None, 'exact', 'lsmr'}, optional
            Method for solving trust-region subproblems, relevant only for 'trf'
            and 'dogbox' methods.
                * 'exact' is suitable for not very large problems with dense
                  Jacobian matrices. The computational complexity per iteration is
                  comparable to a singular value decomposition of the Jacobian
                  matrix.
                * 'lsmr' is suitable for problems with sparse and large Jacobian
                  matrices. It uses the iterative procedure
                  `scipy.sparse.linalg.lsmr` for finding a solution of a linear
                  least-squares problem and only requires matrix-vector product
                  evaluations.
            If None (default) the solver is chosen based on the type of Jacobian
            returned on the first iteration.
        tr_options : dict, optional
            Keyword options passed to trust-region solver.
                * ``tr_solver='exact'``: `tr_options` are ignored.
                * ``tr_solver='lsmr'``: options for `scipy.sparse.linalg.lsmr`.
                  Additionally  ``method='trf'`` supports  'regularize' option
                  (bool, default is True) which adds a regularization term to the
                  normal equation, which improves convergence if the Jacobian is
                  rank-deficient [Byrd]_ (eq. 3.4).
        jac_sparsity : {None, array_like, sparse matrix}, optional
            Defines the sparsity structure of the Jacobian matrix for finite
            difference estimation, its shape must be (m, n). If the Jacobian has
            only few non-zero elements in *each* row, providing the sparsity
            structure will greatly speed up the computations [Curtis]_. A zero
            entry means that a corresponding element in the Jacobian is identically
            zero. If provided, forces the use of 'lsmr' trust-region solver.
            If None (default) then dense differencing will be used. Has no effect
            for 'lm' method.
        verbose : {0, 1, 2}, optional
            Level of algorithm's verbosity:
                * 0 (default) : work silently.
                * 1 : display a termination report.
                * 2 : display progress during iterations (not supported by 'lm'
                  method).
        args, kwargs : tuple and dict, optional
            Additional arguments passed to `fun` and `jac`. Both empty by default.
            The calling signature is ``fun(x, *args, **kwargs)`` and the same for
            `jac`.
        Returns
        -------
        params : array_like
            Solution found.
        info : OptimizeResult
            Subclass of dict object with the following atributes:

                cost : float
                    Value of the cost function at the solution.
                fun : ndarray, shape (m,)
                    Vector of residuals at the solution.
                jac : ndarray, sparse matrix or LinearOperator, shape (m, n)
                    Modified Jacobian matrix at the solution, in the sense that J^T J
                    is a Gauss-Newton approximation of the Hessian of the cost function.
                    The type is the same as the one used by the algorithm.
                grad : ndarray, shape (m,)
                    Gradient of the cost function at the solution.
                optimality : float
                    First-order optimality measure. In unconstrained problems, it is always
                    the uniform norm of the gradient. In constrained problems, it is the
                    quantity which was compared with `gtol` during iterations.
                active_mask : ndarray of int, shape (n,)
                    Each component shows whether a corresponding constraint is active
                    (that is, whether a variable is at the bound):
                        *  0 : a constraint is not active.
                        * -1 : a lower bound is active.
                        *  1 : an upper bound is active.
                    Might be somewhat arbitrary for 'trf' method as it generates a sequence
                    of strictly feasible iterates and `active_mask` is determined within a
                    tolerance threshold.
                nfev : int
                    Number of function evaluations done. Methods 'trf' and 'dogbox' do not
                    count function calls for numerical Jacobian approximation, as opposed
                    to 'lm' method.
                njev : int or None
                    Number of Jacobian evaluations done. If numerical Jacobian
                    approximation is used in 'lm' method, it is set to None.
                status : int
                    The reason for algorithm termination:
                        * -1 : improper input parameters status returned from MINPACK.
                        *  0 : the maximum number of function evaluations is exceeded.
                        *  1 : `gtol` termination condition is satisfied.
                        *  2 : `ftol` termination condition is satisfied.
                        *  3 : `xtol` termination condition is satisfied.
                        *  4 : Both `ftol` and `xtol` termination conditions are satisfied.
                message : str
                    Verbal description of the termination reason.
                success : bool
                    True if one of the convergence criteria is satisfied (`status` > 0).

        Notes
        -----
        Method 'lm' (Levenberg-Marquardt) calls a wrapper over least-squares
        algorithms implemented in MINPACK (lmder, lmdif). It runs the
        Levenberg-Marquardt algorithm formulated as a trust-region type algorithm.
        The implementation is based on paper [JJMore]_, it is very robust and
        efficient with a lot of smart tricks. It should be your first choice
        for unconstrained problems. Note that it doesn't support bounds. Also
        it doesn't work when m < n.

        Method 'trf' (Trust Region Reflective) is motivated by the process of
        solving a system of equations, which constitute the first-order optimality
        condition for a bound-constrained minimization problem as formulated in
        [STIR]_. The algorithm iteratively solves trust-region subproblems
        augmented by a special diagonal quadratic term and with trust-region shape
        determined by the distance from the bounds and the direction of the
        gradient. This enhancements help to avoid making steps directly into bounds
        and efficiently explore the whole space of variables. To further improve
        convergence, the algorithm considers search directions reflected from the
        bounds. To obey theoretical requirements, the algorithm keeps iterates
        strictly feasible. With dense Jacobians trust-region subproblems are
        solved by an exact method very similar to the one described in [JJMore]_
        (and implemented in MINPACK). The difference from the MINPACK
        implementation is that a singular value decomposition of a Jacobian
        matrix is done once per iteration, instead of a QR decomposition and series
        of Givens rotation eliminations. For large sparse Jacobians a 2-d subspace
        approach of solving trust-region subproblems is used [STIR]_, [Byrd]_.
        The subspace is spanned by a scaled gradient and an approximate
        Gauss-Newton solution delivered by `scipy.sparse.linalg.lsmr`. When no
        constraints are imposed the algorithm is very similar to MINPACK and has
        generally comparable performance. The algorithm works quite robust in
        unbounded and bounded problems, thus it is chosen as a default algorithm.

        Method 'dogbox' operates in a trust-region framework, but considers
        rectangular trust regions as opposed to conventional ellipsoids [Voglis]_.
        The intersection of a current trust region and initial bounds is again
        rectangular, so on each iteration a quadratic minimization problem subject
        to bound constraints is solved approximately by Powell's dogleg method
        [NumOpt]_. The required Gauss-Newton step can be computed exactly for
        dense Jacobians or approximately by `scipy.sparse.linalg.lsmr` for large
        sparse Jacobians. The algorithm is likely to exhibit slow convergence when
        the rank of Jacobian is less than the number of variables. The algorithm
        often outperforms 'trf' in bounded problems with a small number of
        variables.

        Robust loss functions are implemented as described in [BA]_. The idea
        is to modify a residual vector and a Jacobian matrix on each iteration
        such that computed gradient and Gauss-Newton Hessian approximation match
        the true gradient and Hessian approximation of the cost function. Then
        the algorithm proceeds in a normal way, i.e. robust loss functions are
        implemented as a simple wrapper over standard least-squares algorithms.

        This methods call internally the scipy implementation of non-linear
        least squares ``least_squares``.

        References
        ----------
        .. [STIR] M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,
                  and Conjugate Gradient Method for Large-Scale Bound-Constrained
                  Minimization Problems," SIAM Journal on Scientific Computing,
                  Vol. 21, Number 1, pp 1-23, 1999.
        .. [NR] William H. Press et. al., "Numerical Recipes. The Art of Scientific
                Computing. 3rd edition", Sec. 5.7.
        .. [Byrd] R. H. Byrd, R. B. Schnabel and G. A. Shultz, "Approximate
                  solution of the trust region problem by minimization over
                  two-dimensional subspaces", Math. Programming, 40, pp. 247-263,
                  1988.
        .. [Curtis] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
                    sparse Jacobian matrices", Journal of the Institute of
                    Mathematics and its Applications, 13, pp. 117-120, 1974.
        .. [JJMore] J. J. More, "The Levenberg-Marquardt Algorithm: Implementation
                    and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
                    Notes in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
        .. [Voglis] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region
                    Dogleg Approach for Unconstrained and Bound Constrained
                    Nonlinear Optimization", WSEAS International Conference on
                    Applied Mathematics, Corfu, Greece, 2004.
        .. [NumOpt] J. Nocedal and S. J. Wright, "Numerical optimization,
                    2nd edition", Chapter 4.
        .. [BA] B. Triggs et. al., "Bundle Adjustment - A Modern Synthesis",
                Proceedings of the International Workshop on Vision Algorithms:
                Theory and Practice, pp. 298-372, 1999.
        """

        if use_jacobian:
            result = least_squares(self.__call__, initial_guess,
                                   self.derivatives,
                                   *args, **kwargs)
        else:
            result = least_squares(self.__call__, initial_guess, "2-point",
                                   *args, **kwargs)

        params = result["x"]

        return params, result
