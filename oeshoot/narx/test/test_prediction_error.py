from __future__ import division, print_function, absolute_import
import numpy as np
from oeshoot.narx import Linear, PredictionError
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)
import numdifftools as nd


def error_numeric_derivatives(error, params):

    # Use numdifftools to estimate derivatives
    jac = nd.Jacobian(error)(params)

    return jac


class TestPredictionErrorInit(TestCase):

    def test_check_wrong_y_size(self):

        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        u = [[2, 3], [3, 4], [5, 6]]
        assert_raises(ValueError, PredictionError, mdl, y, u)

    def test_check_wrong_u_size(self):

        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2], [2, 3], [3, 4]]
        u = [[2, 3, 3], [3, 4, 2], [5, 6, 4]]
        assert_raises(ValueError, PredictionError, mdl, y, u)

    def test_check_diferent_Nd(self):

        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2], [2, 3], [3, 4]]
        u = [[2, 3], [3, 3], [4, 2], [5, 6]]
        assert_raises(ValueError, PredictionError, mdl, y, u)


class TestPredictionErrorCall(TestCase):

    def test_siso_prediction_error(self):

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
        u = [[1+0.5],
             [1+0.5],
             [1+0.5],
             [1+0.5],
             [1+0.5],
             [-2+0.3],
             [-2+0.3],
             [-2+0.3],
             [-2+0.3],
             [-2+0.3],
             [3+0.3],
             [3+0.3],
             [3+0.3],
             [3+0.3],
             [3+0.3],
             [10+1],
             [10+1],
             [10+1],
             [10+1],
             [10+1]]
        params = [1, -0.8, 0.2, 4, 5, 6]

        expected_error = [7.5,
                          7.5,
                          7.5,
                          6.7,
                          5.7,
                          4.5,
                          4.5,
                          4.5,
                          4.5,
                          4.5,
                          4.5,
                          4.5,
                          4.5,
                          7.3,
                          10.8,
                          15,
                          15]

        e = PredictionError(mdl, y, u)
        error = e(params)
        assert_array_almost_equal(error, expected_error)

    def test_mimo_prediction_error(self):

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
        u = [[1+0.5, 1+0.5],
             [1+0.5, 1+0.5],
             [1+0.5, 1+0.5],
             [1+0.5, 1+0.5],
             [1+0.5, 3+0.5],
             [-2+0.3, 3+0.3],
             [-2+0.3, 3+0.3],
             [-2+0.3, 3+0.3],
             [-2+0.3, 1+0.3],
             [-2+0.3, 1+0.3],
             [3+0.3, 1+0.3],
             [3+0.3, 1+0.3],
             [3+0.3, 5+0.3],
             [3+0.3, 5+0.3],
             [3+0.3, 5+0.3],
             [10+1, 5+1],
             [10+1, -2+1],
             [10+1, -2+1],
             [10+1, -2+1],
             [10+1, -2+1]]
        params = [1, 0, -0.8, 0, 0.2, 0,
                  0, 1, 0, -0.8, 0, 0.2,
                  4, 0, 5, 0, 6, 0,
                  4, 4, 5, 5, 6, 6]

        expected_error = [7.5, 15,
                          7.5, 15,
                          7.5, 15,
                          6.7, 13.4,
                          5.7, 11.4,
                          4.5, 9,
                          4.5, 9,
                          4.5, 9,
                          4.5, 9,
                          4.5, 9,
                          4.5, 9,
                          4.5, 9,
                          4.5, 9,
                          7.3, 14.6,
                          10.8, 21.6,
                          15, 30,
                          15, 30]

        error = PredictionError(mdl, y, u)(params)
        assert_array_almost_equal(error, expected_error)


class TestPredictionErrorDerivatives(TestCase):

    def test_siso_prediction_error_derivatives(self):

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
        u = [[1+0.5],
             [1+0.5],
             [1+0.5],
             [1+0.5],
             [1+0.5],
             [-2+0.3],
             [-2+0.3],
             [-2+0.3],
             [-2+0.3],
             [-2+0.3],
             [3+0.3],
             [3+0.3],
             [3+0.3],
             [3+0.3],
             [3+0.3],
             [10+1],
             [10+1],
             [10+1],
             [10+1],
             [10+1]]
        params = [1, -0.8, 0.2, 4, 5, 6]

        error = PredictionError(mdl, y, u)
        jac = error.derivatives(params)
        jac_numeric \
            = error_numeric_derivatives(error, params)
        assert_array_almost_equal(jac, jac_numeric)

    def test_mimo_prediction_error(self):

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
        u = [[1+0.5, 1+0.5],
             [1+0.5, 1+0.5],
             [1+0.5, 1+0.5],
             [1+0.5, 1+0.5],
             [1+0.5, 3+0.5],
             [-2+0.3, 3+0.3],
             [-2+0.3, 3+0.3],
             [-2+0.3, 3+0.3],
             [-2+0.3, 1+0.3],
             [-2+0.3, 1+0.3],
             [3+0.3, 1+0.3],
             [3+0.3, 1+0.3],
             [3+0.3, 5+0.3],
             [3+0.3, 5+0.3],
             [3+0.3, 5+0.3],
             [10+1, 5+1],
             [10+1, -2+1],
             [10+1, -2+1],
             [10+1, -2+1],
             [10+1, -2+1]]
        params = [1, 0, -0.8, 0, 0.2, 0,
                  0, 1, 0, -0.8, 0, 0.2,
                  4, 0, 5, 0, 6, 0,
                  4, 4, 5, 5, 6, 6]

        error = PredictionError(mdl, y, u)
        jac = error.derivatives(params)
        jac_numeric \
            = error_numeric_derivatives(error, params)
        assert_array_almost_equal(jac, jac_numeric)


class TestPredictionErrorLsqEstimateParameters(TestCase):

    def test_siso_estimate_parameters(self):
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
        true_params = [1, -0.8, 0.2, 4, 5, 6]
        initial_guess = [0, 0, 0, 0, 0, 0]

        error = PredictionError(mdl, y, u)
        params, info = error.lsq_estimate_parameters(initial_guess)

        assert_array_almost_equal(params, true_params)

    def test_mimo_estimate_parameters(self):
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
        true_params = [1, 0, -0.8, 0, 0.2, 0,
                       0, 1, 0, -0.8, 0, 0.2,
                       4, 0, 5, 0, 6, 0,
                       4, 4, 5, 5, 6, 6]
        initial_guess = [0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0]

        error = PredictionError(mdl, y, u)

        error = PredictionError(mdl, y, u)
        params, info = error.lsq_estimate_parameters(initial_guess)

        assert_array_almost_equal(params, true_params)
