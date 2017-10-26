from __future__ import division, print_function, absolute_import
import numpy as np
from oeshoot.narx import Linear
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)


class TestLinearInit(TestCase):

    def test_Nparams_siso(self):
        mdl = Linear(N=2, M=4)
        assert_equal(mdl.Nparams, 6)

    def test_Nparams_mimo(self):
        mdl = Linear(N=3, M=6, Ny=2, Nu=2, delay=3)
        assert_equal(mdl.Nparams, 28)

    def test_Nparams_ar_model(self):
        mdl = Linear(N=3, M=0, Ny=1, Nu=0)
        assert_equal(mdl.Nparams, 3)

    def test_Nparams_fir_model(self):
        mdl = Linear(N=0, M=2, Ny=1, Nu=1)
        assert_equal(mdl.Nparams, 2)


class TestLinearCall(TestCase):

    def test_ynext_siso(self):
        mdl = Linear(N=2, M=2)
        ynext = mdl(y=[1, 2], u=[2, 3], params=[1, 2, 3, 4])
        assert_array_almost_equal(ynext, [23])

    def test_ynext_tito(self):
        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2], [2, 3]]
        u = [[2, 3], [3, 4]]
        ynext = mdl(y=y, u=u,
                    params=[1, 2, 0, 0,
                            0, 0, 5, 6,
                            3, 4, 0, 0,
                            0, 0, 7, 8])
        assert_array_almost_equal(ynext, [23, 81])

    def test_ynext_ar_model(self):
        mdl = Linear(N=3, M=0, Ny=1, Nu=0)
        ynext = mdl(y=[1, 2, 3], u=[], params=[1, 2, 3])
        assert_array_almost_equal(ynext, [14])

    def test_ynext_fir_model(self):
        mdl = Linear(N=0, M=2, Ny=1, Nu=1)
        ynext = mdl(y=[], u=[2, 1], params=[1, 2])
        assert_array_almost_equal(ynext, [4])


class TestLinearDerivatives(TestCase):

    def test_siso_ynext_derivatives(self):
        mdl = Linear(N=2, M=2)
        dy, du, dparams = mdl.derivatives(y=[1, 2], u=[2, 3],
                                          params=[10, 20, 30, 40])
        assert_array_almost_equal(dy, [[[10], [20]]])
        assert_array_almost_equal(du, [[[30], [40]]])
        assert_array_almost_equal(dparams, [[1, 2, 2, 3]])

    def test_tito_ynext_derivatives(self):
        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2], [2, 3]]
        u = [[2, 3], [3, 4]]
        dy, du, dparams = mdl.derivatives(y=y, u=u,
                                          params=[10, 20, 0, 0,
                                                  0, 0, 50, 60,
                                                  30, 40, 0, 0,
                                                  0, 0, 70, 80])
        assert_array_almost_equal(dy, [[[10, 20],
                                        [0, 0]],
                                       [[0, 0],
                                        [50, 60]]])
        assert_array_almost_equal(du, [[[30, 40],
                                        [0, 0]],
                                       [[0, 0],
                                        [70, 80]]])
        assert_array_almost_equal(dparams,
                                  [[1, 2, 2, 3, 0, 0, 0, 0, 2,
                                    3, 3, 4, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 2, 2, 3, 0,
                                    0, 0, 0, 2, 3, 3, 4]])

    def test_siso_numerical_approx(self):
        # Define model
        N = 2
        M = 2
        Ny = 1
        Nu = 1
        mdl = Linear(N, M)
        Nparams = mdl.Nparams

        # Define ynext and derivatives
        y = np.array([1.23, 2.41])
        u = np.array([2.54, 23])
        params = [1, 5.46, 3.23, 4]
        dy, du, dparams = mdl.derivatives(y, u,
                                          params)

        # Compute the same derivatives numerically
        dy_numeric, du_numeric, \
            dparams_numeric = mdl._numeric_derivatives(y, u,
                                                       params)

        # Check
        assert_array_almost_equal(dy, dy_numeric)
        assert_array_almost_equal(du, du_numeric)
        assert_array_almost_equal(dparams, dparams_numeric)

    def test_mimo_numerical_approx_threeinputs_threeoutputs(self):
        # Define model
        N = 2
        M = 2
        Ny = 3
        Nu = 3
        mdl = Linear(N, M, Ny, Nu)
        Nparams = mdl.Nparams

        # Define ynext and derivatives
        y = np.array([[1.23, 2.41, 3], [1.23, 2.41, 5]])
        u = np.array([[4.53, 6.41, 10], [2.23, 2.41, 51]])
        params = np.arange(Nparams)
        dy, du, dparams = mdl.derivatives(y, u,
                                          params)

        # Compute the same derivatives numerically
        dy_numeric, du_numeric, \
            dparams_numeric = mdl._numeric_derivatives(y, u,
                                                       params)

        # Check
        assert_array_almost_equal(dy, dy_numeric)
        assert_array_almost_equal(du, du_numeric)
        assert_array_almost_equal(dparams, dparams_numeric)

    def test_mimo_numerical_approx_twoinputs_threeoutputs(self):
        # Define model
        N = 3
        M = 2
        Ny = 2
        Nu = 3
        mdl = Linear(N, M, Ny, Nu)
        Nparams = mdl.Nparams

        # Define ynext and derivatives
        y = np.array([[12, 2.41], [3, 1.23], [2.41, 5]])
        u = np.array([[4.5, 6.4, 10], [2.23, 2.41, 521]])
        params = np.arange(Nparams)
        dy, du, dparams = mdl.derivatives(y, u,
                                          params)

        # Compute the same derivatives numerically
        dy_numeric, du_numeric, \
            dparams_numeric = mdl._numeric_derivatives(y, u,
                                                       params)

        # Check
        assert_array_almost_equal(dy, dy_numeric)
        assert_array_almost_equal(du, du_numeric)
        assert_array_almost_equal(dparams, dparams_numeric)

    def test_ar_model_derivatives(self):
        mdl = Linear(N=3, M=0, Ny=1, Nu=0)
        y = [1, 2, 3]
        u = []
        params = [1, 2, 3]

        dy, du, dparams = mdl.derivatives(y, u, params)

        assert_array_almost_equal(dy, [[[1], [2], [3]]])
        assert_array_almost_equal(du, np.reshape([], (1, 0, 0)))
        assert_array_almost_equal(dparams, [[1, 2, 3]])

    def test_ar_model_numerical_derivatives(self):
        mdl = Linear(N=3, M=0, Ny=1, Nu=0)
        y = [1, 2, 3]
        u = []
        params = [1, 2, 3]

        dy, du, dparams = mdl.derivatives(y, u, params)

        dy_numeric, du_numeric, \
            dparams_numeric = mdl._numeric_derivatives(y, u, params)

        assert_array_almost_equal(dy, dy_numeric)
        assert_array_almost_equal(du, du_numeric)
        assert_array_almost_equal(dparams, dparams_numeric)

    def test_fir_model_derivatives(self):
        mdl = Linear(N=0, M=2, Ny=1, Nu=1)
        y = []
        u = [2, 1]
        params = [1, 2]

        dy, du, dparams = mdl.derivatives(y, u, params)

        assert_array_almost_equal(du, [[[1], [2]]])
        assert_array_almost_equal(dy, np.reshape([], (1, 0, 1)))
        assert_array_almost_equal(dparams, [[2, 1]])

    def test_fir_model_numerical_derivatives(self):
        mdl = Linear(N=0, M=2, Ny=1, Nu=1)
        y = []
        u = [2, 1]
        params = [1, 2]

        dy, du, dparams = mdl.derivatives(y, u, params)

        dy_numeric, du_numeric, \
            dparams_numeric = mdl._numeric_derivatives(y, u, params)

        assert_array_almost_equal(dy, dy_numeric)
        assert_array_almost_equal(du, du_numeric)
        assert_array_almost_equal(dparams, dparams_numeric)
