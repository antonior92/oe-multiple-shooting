from __future__ import division, print_function, absolute_import
import numpy as np
from oeshoot.narx import (Linear, Monomial, Polynomial,
                           predict, predict_derivatives)
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)
import numdifftools as nd


# Auxiliar Functions
def predict_numeric_derivatives(mdl, y, u, params):

    # Check Arguments
    y = np.atleast_2d(y)
    u = np.atleast_2d(u)
    params = np.atleast_1d(params)

    # Use numdifftools to estimate derivatives
    def fun(x):
        return predict(mdl, y, u, x).flatten()
    jac = nd.Jacobian(fun)(params).reshape((-1, mdl.Ny,
                                            mdl.Nparams))

    return jac


class TestPredict(TestCase):

    def test_check_parameters_size(self):

        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2], [2, 3], [3, 4]]
        u = [[2, 3], [3, 4], [5, 6]]
        params = [1, 2, 0, 0,
                  0, 0, 5, 6,
                  3, 4, 0, 0]
        assert_raises(ValueError, predict, mdl, y, u, params)

    def test_check_wrong_y_size(self):

        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        u = [[2, 3], [3, 4], [5, 6]]
        params = [1, 2, 0, 0,
                  0, 0, 5, 6,
                  3, 4, 0, 0,
                  0, 0, 1, 2]
        assert_raises(ValueError, predict, mdl, y, u, params)

    def test_check_wrong_u_size(self):

        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2], [2, 3], [3, 4]]
        u = [[2, 3, 3], [3, 4, 2], [5, 6, 4]]
        params = [1, 2, 0, 0,
                  0, 0, 5, 6,
                  3, 4, 0, 0,
                  0, 0, 1, 2]
        assert_raises(ValueError, predict, mdl, y, u, params)

    def test_check_diferent_Nd(self):

        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2], [2, 3], [3, 4]]
        u = [[2, 3], [3, 3], [4, 2], [5, 6]]
        params = [1, 2, 0, 0,
                  0, 0, 5, 6,
                  3, 4, 0, 0,
                  0, 0, 1, 2]
        assert_raises(ValueError, predict, mdl, y, u, params)

    def test_siso_prediction(self):

        mdl = Linear(N=3, M=3)
        y = [[0],
             [4],
             [13],
             [24.8],
             [30.2],
             [27.96],
             [11.76],
             [-16.568],
             [-50.384],
             [-64.7776],
             [-57.784],
             [-26.03872],
             [22.23296],
             [76.507136],
             [98.513024],
             [86.7539072],
             [96.2449152],
             [154.54439424],
             [244.89924352],
             [290.512711168]]
        u = [[1],
             [1],
             [1],
             [1],
             [1],
             [-2],
             [-2],
             [-2],
             [-2],
             [-2],
             [3],
             [3],
             [3],
             [3],
             [3],
             [10],
             [10],
             [10],
             [10],
             [10]]
        params = [1, -0.8, 0.2, 4, 5, 6]

        y1 = predict(mdl, y[:-1], u[:-1], params)
        assert_array_almost_equal(y1, y[3:])

    def test_mimo_prediction(self):

        mdl = Linear(N=3, M=3, Ny=2, Nu=2)
        y = [[0, 0],
             [4, 8],
             [13, 26],
             [24.8, 49.6],
             [30.2, 60.4],
             [27.96, 63.92],
             [11.76, 61.52],
             [-16.568, 55.464],
             [-50.384, 34.032],
             [-64.7776, 8.9648],
             [-57.784, -10.1680],
             [-26.03872, -5.53344],
             [22.23296, 34.39392],
             [76.507136, 112.787072],
             [98.513024, 180.165248],
             [86.7539072, 216.8143744],
             [96.2449152, 243.2395904],
             [154.54439424, 260.82114048],
             [244.89924352, 271.59234304],
             [290.512711168, 231.583348736]]
        u = [[1, 1],
             [1, 1],
             [1, 1],
             [1, 1],
             [1, 3],
             [-2, 3],
             [-2, 3],
             [-2, 3],
             [-2, 1],
             [-2, 1],
             [3, 1],
             [3, 1],
             [3, 5],
             [3, 5],
             [3, 5],
             [10, 5],
             [10, -2],
             [10, -2],
             [10, -2],
             [10, -2]]
        params = [1, 0, -0.8, 0, 0.2, 0,
                  0, 1, 0, -0.8, 0, 0.2,
                  4, 0, 5, 0, 6, 0,
                  4, 4, 5, 5, 6, 6]

        y1 = predict(mdl, y[:-1], u[:-1], params)
        assert_array_almost_equal(y1, y[3:])

    def test_ar_model_predict(self):

        m1 = Monomial(1, 0, [1], [], [1], [])
        m2 = Monomial(1, 0, [1], [], [2], [])

        m = [m1, m2]
        mdl = Polynomial(m)

        y = [[0.500000000000000],
             [0.925000000000000],
             [0.256687500000000],
             [0.705956401171875],
             [0.768053255020420],
             [0.659145574149943],
             [0.831288939045395],
             [0.518916263804854],
             [0.923676047365561],
             [0.260844845488171],
             [0.713377804660565],
             [0.756538676169480],
             [0.681495258228080],
             [0.803120043590673],
             [0.585037484942277],
             [0.898243916772361],
             [0.338186596189094],
             [0.828120762684376],
             [0.526646030853067],
             [0.922372959447176]]
        u = np.reshape([], (20, 0))
        params = [3.7, -3.7]

        y1 = predict(mdl, y[:-1], u[:-1], params)
        assert_array_almost_equal(y1, y[1:])

    def test_fir_model_predict(self):

        m1 = Monomial(0, 1, [], [1], [], [1])
        m2 = Monomial(0, 1, [], [1], [], [2])

        m = [m1, m2]
        mdl = Polynomial(m)

        y = [[0],
             [-1],
             [-6],
             [-15],
             [-28],
             [-45],
             [-66],
             [-91],
             [-120],
             [-153],
             [-190],
             [-231],
             [-276],
             [-325],
             [-378],
             [-435],
             [-496],
             [-561],
             [-630],
             [-703]]
        u = np.arange(1, 21).reshape((20, 1))
        params = [1, -2]

        y1 = predict(mdl, y[:-1], u[:-1], params)
        assert_array_almost_equal(y1, y[1:])


class TestPredictDerivatives(TestCase):

    def test_check_parameters_size(self):

        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2], [2, 3], [3, 4]]
        u = [[2, 3], [3, 4], [5, 6]]
        params = [1, 2, 0, 0,
                  0, 0, 5, 6,
                  3, 4, 0, 0]
        assert_raises(ValueError, predict_derivatives, mdl, y, u, params)

    def test_check_wrong_y_size(self):

        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        u = [[2, 3], [3, 4], [5, 6]]
        params = [1, 2, 0, 0,
                  0, 0, 5, 6,
                  3, 4, 0, 0,
                  0, 0, 1, 2]
        assert_raises(ValueError, predict_derivatives, mdl, y, u, params)

    def test_check_wrong_u_size(self):

        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2], [2, 3], [3, 4]]
        u = [[2, 3, 3], [3, 4, 2], [5, 6, 4]]
        params = [1, 2, 0, 0,
                  0, 0, 5, 6,
                  3, 4, 0, 0,
                  0, 0, 1, 2]
        assert_raises(ValueError, predict_derivatives, mdl, y, u, params)

    def test_check_diferent_Nd(self):

        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2], [2, 3], [3, 4]]
        u = [[2, 3], [3, 3], [4, 2], [5, 6]]
        params = [1, 2, 0, 0,
                  0, 0, 5, 6,
                  3, 4, 0, 0,
                  0, 0, 1, 2]
        assert_raises(ValueError, predict_derivatives, mdl, y, u, params)

    def test_siso_predict_numerical_derivativesative(self):

        mdl = Linear(N=3, M=3)
        y = [[0],
             [4],
             [13],
             [24.8],
             [30.2],
             [27.96],
             [11.76],
             [-16.568],
             [-50.384],
             [-64.7776],
             [-57.784],
             [-26.03872],
             [22.23296],
             [76.507136],
             [98.513024],
             [86.7539072],
             [96.2449152],
             [154.54439424],
             [244.89924352],
             [290.512711168]]
        u = [[1],
             [1],
             [1],
             [1],
             [1],
             [-2],
             [-2],
             [-2],
             [-2],
             [-2],
             [3],
             [3],
             [3],
             [3],
             [3],
             [10],
             [10],
             [10],
             [10],
             [10]]
        params = [1, -0.8, 0.2, 4, 5, 6]

        jac = predict_derivatives(mdl, y[:-1], u[:-1], params)
        jac_numeric = predict_numeric_derivatives(mdl, y[:-1], u[:-1], params)

        assert_array_almost_equal(jac, jac_numeric)

    def test_mimo_predict_numerical_derivative(self):

        mdl = Linear(N=3, M=3, Ny=2, Nu=2)
        y = [[0, 0],
             [4, 8],
             [13, 26],
             [24.8, 49.6],
             [30.2, 60.4],
             [27.96, 63.92],
             [11.76, 61.52],
             [-16.568, 55.464],
             [-50.384, 34.032],
             [-64.7776, 8.9648],
             [-57.784, -10.1680],
             [-26.03872, -5.53344],
             [22.23296, 34.39392],
             [76.507136, 112.787072],
             [98.513024, 180.165248],
             [86.7539072, 216.8143744],
             [96.2449152, 243.2395904],
             [154.54439424, 260.82114048],
             [244.89924352, 271.59234304],
             [290.512711168, 231.583348736]]
        u = [[1, 1],
             [1, 1],
             [1, 1],
             [1, 1],
             [1, 3],
             [-2, 3],
             [-2, 3],
             [-2, 3],
             [-2, 1],
             [-2, 1],
             [3, 1],
             [3, 1],
             [3, 5],
             [3, 5],
             [3, 5],
             [10, 5],
             [10, -2],
             [10, -2],
             [10, -2],
             [10, -2]]
        params = [1, 0, -0.8, 0, 0.2, 0,
                  0, 1, 0, -0.8, 0, 0.2,
                  4, 0, 5, 0, 6, 0,
                  4, 4, 5, 5, 6, 6]

        jac = predict_derivatives(mdl, y[:-1], u[:-1], params)
        jac_numeric = predict_numeric_derivatives(mdl, y[:-1], u[:-1], params)

        assert_array_almost_equal(jac, jac_numeric)

    def test_ar_model_predict_derivative(self):

        m1 = Monomial(1, 0, [1], [], [1], [])
        m2 = Monomial(1, 0, [1], [], [2], [])

        m = [m1, m2]
        mdl = Polynomial(m)

        y = [[0.500000000000000],
             [0.925000000000000],
             [0.256687500000000],
             [0.705956401171875],
             [0.768053255020420],
             [0.659145574149943],
             [0.831288939045395],
             [0.518916263804854],
             [0.923676047365561],
             [0.260844845488171],
             [0.713377804660565],
             [0.756538676169480],
             [0.681495258228080],
             [0.803120043590673],
             [0.585037484942277],
             [0.898243916772361],
             [0.338186596189094],
             [0.828120762684376],
             [0.526646030853067],
             [0.922372959447176]]
        u = np.reshape([], (20, 0))
        params = [3.7, -3.7]

        jac = predict_derivatives(mdl, y[:-1], u[:-1], params)
        jac_numeric = predict_numeric_derivatives(mdl, y[:-1], u[:-1], params)
        assert_array_almost_equal(jac, jac_numeric)

    def test_fir_model_predict_derivative(self):

        m1 = Monomial(0, 1, [], [1], [], [1])
        m2 = Monomial(0, 1, [], [1], [], [2])

        m = [m1, m2]
        mdl = Polynomial(m)

        y = [[0],
             [-1],
             [-6],
             [-15],
             [-28],
             [-45],
             [-66],
             [-91],
             [-120],
             [-153],
             [-190],
             [-231],
             [-276],
             [-325],
             [-378],
             [-435],
             [-496],
             [-561],
             [-630],
             [-703]]
        u = np.arange(1, 21).reshape((20, 1))
        params = [1, -2]

        jac = predict_derivatives(mdl, y[:-1], u[:-1], params)
        jac_numeric = predict_numeric_derivatives(mdl, y[:-1], u[:-1], params)
        assert_array_almost_equal(jac, jac_numeric)
