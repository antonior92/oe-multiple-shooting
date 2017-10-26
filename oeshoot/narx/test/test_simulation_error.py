from __future__ import division, print_function, absolute_import
import numpy as np
from oeshoot.narx import (Linear, Monomial, Polynomial,
                           SimulationError,
                           generate_simulation_intervals,
                           initial_conditions_from_data,
                           assemble_extended_params,
                           disassemble_extended_params)
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)
import numdifftools as nd


def error_numeric_derivatives(error, params):

    # Use numdifftools to estimate derivatives
    jac = nd.Jacobian(error, step=10**(-6))(params)

    return jac


class TestSimulationErrorArgCheck(TestCase):

    def test_check_wrong_y_size(self):

        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        u = [[2, 3], [3, 4], [5, 6]]
        assert_raises(ValueError, SimulationError, mdl, y, u)

    def test_check_wrong_u_size(self):

        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2], [2, 3], [3, 4]]
        u = [[2, 3, 3], [3, 4, 2], [5, 6, 4]]
        assert_raises(ValueError, SimulationError, mdl, y, u)

    def test_check_diferent_Nd(self):

        mdl = Linear(N=2, M=2, Ny=2, Nu=2)
        y = [[1, 2], [2, 3], [3, 4]]
        u = [[2, 3], [3, 3], [4, 2], [5, 6]]
        assert_raises(ValueError, SimulationError, mdl, y, u)


class TestSimulationErrorGenerateIntervals(TestCase):

    def test_generate_simulation_intervals_integer(self):

        sim_list = generate_simulation_intervals(100, 25, 6)

        assert_equal(sim_list, [(0, 31), (25, 56), (50, 81), (75, 100)])

    def test_generate_simulation_intervals_fractional(self):

        sim_list = generate_simulation_intervals(100, 23, 6)

        assert_equal(sim_list, [(0, 26), (20, 46),
                                (40, 66), (60, 86),
                                (80, 100)])

    def test_generate_simulation_intervals_fractional_2(self):
        sim_list = generate_simulation_intervals(299, 50, 6)

        assert_equal(sim_list, [(0, 56), (50, 106),
                                (100, 156), (150, 206),
                                (200, 256), (250, 299)])

    def test_generate_simulation_intervals_fractional_3(self):

        sim_list = generate_simulation_intervals(286, 50, 6)

        assert_equal(sim_list, [(0, 54), (48, 102),
                                (96, 150), (144, 198),
                                (192, 245), (239, 286)])

    def test_generate_simulation_intervals_full(self):

        sim_list = generate_simulation_intervals(286, 286, 6)

        assert_equal(sim_list, [(0, 286)])

    def test_generate_simulation_intervals_toosmall(self):

        assert_raises(ValueError, generate_simulation_intervals, 8, 2, 3)


class TestSimulationErrorCall(TestCase):

    def test_siso_simulation_error_one_shoot(self):

        mdl = Linear(N=3, M=3)
        y = [[0],
             [4],
             [13],
             [24.8-1.34],
             [30.2-1.34],
             [27.96-1.34],
             [11.76-1.34],
             [-16.568-1.34],
             [-50.384-1.34],
             [-64.7776-1.34],
             [-57.784-1.34],
             [-26.03872-5.34],
             [22.23296-5.34],
             [76.507136-5.34],
             [98.513024-5.34],
             [86.7539072-5.34],
             [96.2449152-5.34],
             [154.54439424-5.34],
             [244.89924352-5.34],
             [290.512711168-5.34]]
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

        expected_error = [1.34,
                          1.34,
                          1.34,
                          1.34,
                          1.34,
                          1.34,
                          1.34,
                          1.34,
                          5.34,
                          5.34,
                          5.34,
                          5.34,
                          5.34,
                          5.34,
                          5.34,
                          5.34,
                          5.34]

        e = SimulationError(mdl, y, u, 20)
        error = e(params+[0, 4, 13])
        assert_array_almost_equal(error, expected_error)

    def test_mimo_simulation_error_one_shoot(self):

        mdl = Linear(N=3, M=3, Ny=2, Nu=2)
        y = [[0, 0],
             [4, 8],
             [13, 26],
             [24.8-1.23, 49.6-2],
             [30.2-1.23, 60.4-2],
             [27.96-1.23, 63.92-2],
             [11.76-1.23, 61.52-2],
             [-16.568-1.23, 55.464-2],
             [-50.384-1.23, 34.032-2],
             [-64.7776-1.23, 8.9648-2],
             [-57.784-1.23, -10.1680-2],
             [-26.03872-1.23, -5.53344-2],
             [22.23296-12, 34.39392-2],
             [76.507136-12, 112.787072-2.45],
             [98.513024-12, 180.165248-2.45],
             [86.7539072-12, 216.8143744-2.45],
             [96.2449152-12, 243.2395904-2.45],
             [154.54439424-12, 260.82114048-2.45],
             [244.89924352-12, 271.59234304-2.45],
             [290.512711168-12, 231.583348736-2.45]]
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

        expected_error = [1.23, 2,
                          1.23, 2,
                          1.23, 2,
                          1.23, 2,
                          1.23, 2,
                          1.23, 2,
                          1.23, 2,
                          1.23, 2,
                          1.23, 2,
                          12, 2,
                          12, 2.45,
                          12, 2.45,
                          12, 2.45,
                          12, 2.45,
                          12, 2.45,
                          12, 2.45,
                          12, 2.45]

        error = SimulationError(mdl, y, u)(params +
                                           [0, 0, 4,
                                            8, 13, 26])
        assert_array_almost_equal(error, expected_error)

    def test_ar_model_simulation_error_one_shoot(self):

        m1 = Monomial(1, 0, [1], [], [1], [])
        m2 = Monomial(1, 0, [1], [], [2], [])

        m = [m1, m2]
        mdl = Polynomial(m)

        y = [[0.500000000000000],
             [0.925000000000000-1.4],
             [0.256687500000000-1.4],
             [0.705956401171875-1.4],
             [0.768053255020420-1.4],
             [0.659145574149943-1.4],
             [0.831288939045395-1.4],
             [0.518916263804854-1.4],
             [0.923676047365561-1.4],
             [0.260844845488171-1.4],
             [0.713377804660565-1.4],
             [0.756538676169480-1.4],
             [0.681495258228080-1],
             [0.803120043590673-1],
             [0.585037484942277-1],
             [0.898243916772361-1],
             [0.338186596189094-1],
             [0.828120762684376-1],
             [0.526646030853067-1],
             [0.922372959447176-1]]
        u = np.reshape([], (20, 0))
        params = [3.7, -3.7]

        expected_error = [1.4,
                          1.4,
                          1.4,
                          1.4,
                          1.4,
                          1.4,
                          1.4,
                          1.4,
                          1.4,
                          1.4,
                          1.4,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1]

        error = SimulationError(mdl, y, u)(params +
                                           [0.5])
        assert_array_almost_equal(error, expected_error)

    def test_siso_simulation_error_multiple_shoot(self):

        mdl = Linear(N=3, M=3)
        y = [[0],
             [4],
             [13],
             [24.8-1.34],
             [30.2-1.34],
             [27.96-1.34],
             [11.76-1.34],
             [-16.568-1.34],
             [-50.384-1.34],
             [-64.7776-1.34],
             [-57.784-1.34],
             [-26.03872-5.34],
             [22.23296-5.34],
             [76.507136-5.34],
             [98.513024-5.34],
             [86.7539072-5.34],
             [96.2449152-5.34],
             [154.54439424-5.34],
             [244.89924352-5.34],
             [290.512711168-5.34]]
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

        expected_error = [1.34,
                          1.34,
                          1.34,
                          1.34,
                          1.34,
                          0,  # overlap error
                          0,  # overlap error
                          0,  # overlap error
                          1.34,
                          1.34,
                          1.34,
                          5.34,
                          5.34,
                          0,  # overlap error
                          0,  # overlap error
                          0,  # overlap error
                          5.34,
                          5.34,
                          5.34,
                          5.34,
                          5.34,
                          0,  # overlap error
                          0,  # overlap error
                          0,  # overlap error
                          5.34,
                          5.34]

        e = SimulationError(mdl, y, u, 5)
        error = e(params +
                  [0, 4, 13] +
                  [27.96, 11.76, -16.568] +
                  [-57.784, -26.03872, 22.23296] +
                  [86.7539072, 96.2449152, 154.54439424])
        assert_array_almost_equal(error, expected_error)

    def test_mimo_simulation_error_multiple_shoot(self):

        mdl = Linear(N=3, M=3, Ny=2, Nu=2)
        y = [[0, 0],
             [4, 8],
             [13, 26],
             [24.8-1.23, 49.6-2],
             [30.2-1.23, 60.4-2],
             [27.96-1.23, 63.92-2],
             [11.76-1.23, 61.52-2],
             [-16.568-1.23, 55.464-2],
             [-50.384-1.23, 34.032-2],
             [-64.7776-1.23, 8.9648-2],
             [-57.784-1.23, -10.1680-2],
             [-26.03872-1.23, -5.53344-2],
             [22.23296-12, 34.39392-2],
             [76.507136-12, 112.787072-2.45],
             [98.513024-12, 180.165248-2.45],
             [86.7539072-12, 216.8143744-2.45],
             [96.2449152-12, 243.2395904-2.45],
             [154.54439424-12, 260.82114048-2.45],
             [244.89924352-12, 271.59234304-2.45],
             [290.512711168-12, 231.583348736-2.45]]
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

        expected_error = [1.23, 2,
                          1.23, 2,
                          1.23, 2,
                          1.23, 2,
                          1.23, 2,
                          0, 0,
                          0, 0,
                          0, 0,
                          1.23, 2,
                          1.23, 2,
                          1.23, 2,
                          1.23, 2,
                          12, 2,
                          0, 0,
                          0, 0,
                          0, 0,
                          12, 2.45,
                          12, 2.45,
                          12, 2.45,
                          12, 2.45,
                          12, 2.45,
                          0, 0,
                          0, 0,
                          0, 0,
                          12, 2.45,
                          12, 2.45]

        error = SimulationError(mdl, y, u, 5)(params +
                                              [0, 0,
                                               4, 8,
                                               13, 26] +
                                              [27.96, 63.92,
                                               11.76, 61.52,
                                               -16.568, 55.464] +
                                              [-57.784, -10.1680,
                                               -26.03872, -5.53344,
                                               22.23296, 34.39392] +
                                              [86.7539072,
                                               216.8143744,
                                               96.2449152,
                                               243.2395904,
                                               154.54439424,
                                               260.82114048])

        assert_array_almost_equal(error, expected_error)

    def test_ar_model_simulation_error_multiple_shoot(self):

        m1 = Monomial(1, 0, [1], [], [1], [])
        m2 = Monomial(1, 0, [1], [], [2], [])

        m = [m1, m2]
        mdl = Polynomial(m)

        y = [[0.500000000000000],
             [0.925000000000000-1.4],
             [0.256687500000000-1.4],
             [0.705956401171875-1.4],
             [0.768053255020420-1.4],
             [0.659145574149943-1.4],
             [0.831288939045395-1.4],
             [0.518916263804854-1.4],
             [0.923676047365561-1.4],
             [0.260844845488171-1.4],
             [0.713377804660565-1.4],
             [0.756538676169480-1.4],
             [0.681495258228080-1],
             [0.803120043590673-1],
             [0.585037484942277-1],
             [0.898243916772361-1],
             [0.338186596189094-1],
             [0.828120762684376-1],
             [0.526646030853067-1],
             [0.922372959447176-1]]
        u = np.reshape([], (20, 0))
        params = [3.7, -3.7]

        expected_error = [1.4,
                          1.4,
                          1.4,
                          1.4,
                          1.4,
                          0,
                          1.4,
                          1.4,
                          1.4,
                          1.4,
                          1.4,
                          0,
                          1.4,
                          1,
                          1,
                          1,
                          1,
                          0,
                          1,
                          1,
                          1,
                          1]

        error = SimulationError(mdl, y, u, 5)(params +
                                              [0.5] +
                                              [0.659145574149943] +
                                              [0.713377804660565] +
                                              [0.898243916772361])
        assert_array_almost_equal(error, expected_error)


class TestSimulationErrorDerivatives(TestCase):

    def test_siso_simulation_error_one_shoot_derivatives(self):

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

        extended_params = params+[0, 4, 13]

        error = SimulationError(mdl, y, u)
        dparams = error.derivatives(extended_params)
        dparams_numeric \
            = error_numeric_derivatives(error, extended_params)
        assert_array_almost_equal(dparams, dparams_numeric)

    def test_mimo_simulation_error_one_shoot_derivatives(self):

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
        extended_params = params + [0, 0, 4,
                                    8, 13, 26]

        error = SimulationError(mdl, y, u)
        dparams = error.derivatives(extended_params)
        dparams_numeric \
            = error_numeric_derivatives(error, extended_params)
        assert_array_almost_equal(dparams, dparams_numeric)

    def test_ar_model_simulation_error_one_shoot(self):

        m1 = Monomial(1, 0, [1], [], [1], [])
        m2 = Monomial(1, 0, [1], [], [2], [])

        m = [m1, m2]
        mdl = Polynomial(m)

        y = [[0.500000000000000],
             [0.925000000000000-1.4],
             [0.256687500000000-1.4],
             [0.705956401171875-1.4],
             [0.768053255020420-1.4],
             [0.659145574149943-1.4],
             [0.831288939045395-1.4],
             [0.518916263804854-1.4],
             [0.923676047365561-1.4],
             [0.260844845488171-1.4],
             [0.713377804660565-1.4],
             [0.756538676169480-1.4],
             [0.681495258228080-1],
             [0.803120043590673-1],
             [0.585037484942277-1],
             [0.898243916772361-1],
             [0.338186596189094-1],
             [0.828120762684376-1],
             [0.526646030853067-1],
             [0.922372959447176-1]]
        u = np.reshape([], (20, 0))
        params = [3.7, -3.7]
        extended_params = params + [0.5]

        error = SimulationError(mdl, y, u)
        dparams = error.derivatives(extended_params)
        dparams_numeric \
            = error_numeric_derivatives(error, extended_params)
        assert_array_almost_equal(dparams, dparams_numeric, decimal=2)

    def test_siso_simulation_error_multiple_shoots_derivatives(self):

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

        extended_params = params + \
                          [0, 4, 13] + \
                          [27.96, 11.76, -16.568] + \
                          [-57.784, -26.03872, 22.23296] + \
                          [86.7539072, 96.2449152, 154.54439424]

        error = SimulationError(mdl, y, u, 5)
        dparams = error.derivatives(extended_params)
        dparams_numeric \
            = error_numeric_derivatives(error, extended_params)
        assert_array_almost_equal(dparams, dparams_numeric)

    def test_mimo_simulation_error_multiple_shoots_derivatives(self):

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
        extended_params = params + [0, 0,
                                    4, 8,
                                    13, 26] + \
                                    [27.96, 63.92,
                                     11.76, 61.52,
                                     -16.568, 55.464] + \
                                     [-57.784, -10.1680,
                                      -26.03872, -5.53344,
                                      22.23296, 34.39392] + \
                                      [86.7539072,
                                       216.8143744,
                                       96.2449152,
                                       243.2395904,
                                       154.54439424,
                                       260.82114048]

        error = SimulationError(mdl, y, u, 5)
        dparams = error.derivatives(extended_params)
        dparams_numeric \
            = error_numeric_derivatives(error, extended_params)
        assert_array_almost_equal(dparams, dparams_numeric)

    def test_ar_model_simulation_error_multiple_shoot(self):

        m1 = Monomial(1, 0, [1], [], [1], [])
        m2 = Monomial(1, 0, [1], [], [2], [])

        m = [m1, m2]
        mdl = Polynomial(m)

        y = [[0.500000000000000],
             [0.925000000000000-1.4],
             [0.256687500000000-1.4],
             [0.705956401171875-1.4],
             [0.768053255020420-1.4],
             [0.659145574149943-1.4],
             [0.831288939045395-1.4],
             [0.518916263804854-1.4],
             [0.923676047365561-1.4],
             [0.260844845488171-1.4],
             [0.713377804660565-1.4],
             [0.756538676169480-1.4],
             [0.681495258228080-1],
             [0.803120043590673-1],
             [0.585037484942277-1],
             [0.898243916772361-1],
             [0.338186596189094-1],
             [0.828120762684376-1],
             [0.526646030853067-1],
             [0.922372959447176-1]]
        u = np.reshape([], (20, 0))
        params = [3.7, -3.7]
        extended_params = params + [0.5] + \
                          [0.659145574149943] + \
                          [0.713377804660565] + \
                          [0.898243916772361]

        error = SimulationError(mdl, y, u, 5)
        dparams = error.derivatives(extended_params)
        dparams_numeric \
            = error_numeric_derivatives(error, extended_params)
        assert_array_almost_equal(dparams, dparams_numeric, decimal=3)


class TestInitialConditionsFromData(TestCase):

    def test_initial_conditions_from_data_siso(self):
        multiple_shoots = generate_simulation_intervals(100, 25, 3)
        y = np.arange(100).reshape((100, 1))

        initial_conditions = initial_conditions_from_data(y, multiple_shoots, 3, 4)

        assert_equal(initial_conditions, [[[1], [2], [3]],
                                          [[26], [27], [28]],
                                          [[51], [52], [53]],
                                          [[76], [77], [78]]])

    def test_initial_conditions_from_data_mimo(self):
        multiple_shoots = generate_simulation_intervals(200, 50, 5)
        y = np.arange(400).reshape((200, 2))

        initial_conditions = initial_conditions_from_data(y, multiple_shoots, 3, 5)

        assert_equal(initial_conditions, [[[4, 5],
                                           [6, 7],
                                           [8, 9]],
                                          [[104, 105],
                                           [106, 107],
                                           [108, 109]],
                                          [[204, 205],
                                           [206, 207],
                                           [208, 209]],
                                          [[304, 305],
                                           [306, 307],
                                           [308, 309]]])


class TestAssembleExtendedParams(TestCase):

    def test_assemble_extended_params(self):

        params = [1, 2, 3, 4, 5]
        initial_conditions = [[[4, 5],
                               [6, 7],
                               [8, 9]],
                              [[104, 105],
                               [106, 107],
                               [108, 109]],
                              [[204, 205],
                               [206, 207],
                               [208, 209]],
                              [[304, 305],
                               [306, 307],
                               [308, 309]]]
        expected_extended_params = [1, 2, 3, 4, 5, 4, 5,
                                    6, 7, 8, 9, 104, 105,
                                    106, 107, 108, 109,
                                    204, 205, 206, 207,
                                    208, 209, 304, 305,
                                    306, 307, 308, 309]

        extended_params = assemble_extended_params(params,
                                                   initial_conditions)

        assert_equal(extended_params, expected_extended_params)


class TestDisassembleExtendedParams(TestCase):

    def test_disassemble_extended_params(self):

        expected_params = [1, 2, 3, 4, 5]
        expected_initial_conditions = [[[4, 5],
                                        [6, 7],
                                        [8, 9]],
                                       [[104, 105],
                                        [106, 107],
                                        [108, 109]],
                                       [[204, 205],
                                        [206, 207],
                                        [208, 209]],
                                       [[304, 305],
                                        [306, 307],
                                        [308, 309]]]
        extended_params = [1, 2, 3, 4, 5, 4, 5,
                           6, 7, 8, 9, 104, 105,
                           106, 107, 108, 109,
                           204, 205, 206, 207,
                           208, 209, 304, 305,
                           306, 307, 308, 309]

        params, initial_conditions \
            = disassemble_extended_params(extended_params, 5, 3, 2, 4)

        assert_equal(params, expected_params)
        assert_equal(initial_conditions, expected_initial_conditions)


class TestLsqEstimateParameters(TestCase):

    def test_siso_estimate_parameters_single_shoot(self):
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

        error = SimulationError(mdl, y, u, maxlength=20)
        params, info = error.lsq_estimate_parameters(initial_guess)

        multiple_shoots = info["multiple_shoots"]
        initial_conditions = info["initial_conditions"]

        assert_array_almost_equal(params, true_params)
        assert_array_almost_equal(multiple_shoots, [(0, 20)])
        assert_array_almost_equal(initial_conditions, [[[0], [4], [13]]], decimal=3)

    def test_siso_estimate_parameters_multiple_shoot(self):

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

        error = SimulationError(mdl, y, u, maxlength=5)
        params, info = error.lsq_estimate_parameters(initial_guess)

        multiple_shoots = info["multiple_shoots"]
        initial_conditions = info["initial_conditions"]

        assert_array_almost_equal(params, true_params, decimal=3)
        assert_array_almost_equal(multiple_shoots, [(0, 8),
                                                    (5, 13),
                                                    (10, 18),
                                                    (15, 20)], decimal=3)
        assert_array_almost_equal(initial_conditions, [[[0],
                                                        [4],
                                                        [13]],
                                                       [[27.96],
                                                        [11.76],
                                                        [-16.568]],
                                                       [[-57.784],
                                                        [-26.03872],
                                                        [22.23296]],
                                                       [[86.7539072],
                                                        [96.2449152],
                                                        [154.54439424]]], decimal=3)

    def test_mimo_estimate_parameters_single_shoot(self):
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

        error = SimulationError(mdl, y, u, maxlength=20)
        params, info = error.lsq_estimate_parameters(initial_guess,
                                                     use_sparse=False)

        multiple_shoots = info["multiple_shoots"]
        initial_conditions = info["initial_conditions"]

        assert_array_almost_equal(params, true_params)
        assert_array_almost_equal(multiple_shoots, [(0, 20)], decimal=3)
        assert_array_almost_equal(initial_conditions, [[[0, 0],
                                                        [4, 8],
                                                        [13, 26]]], decimal=3)

    def test_mimo_estimate_parameters_multiple_shoot(self):
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

        error = SimulationError(mdl, y, u, maxlength=5)
        params, info = error.lsq_estimate_parameters(initial_guess, use_sparse=False)

        multiple_shoots = info["multiple_shoots"]
        initial_conditions = info["initial_conditions"]

        assert_array_almost_equal(params, true_params)
        assert_array_almost_equal(multiple_shoots, [(0, 8),
                                                    (5, 13),
                                                    (10, 18),
                                                    (15, 20)])
        assert_array_almost_equal(initial_conditions, [[[0, 0],
                                                        [4, 8],
                                                        [13, 26]],
                                                       [[27.96, 63.92],
                                                        [11.76, 61.52],
                                                        [-16.568, 55.464]],
                                                       [[-57.784, -10.1680],
                                                        [-26.03872, -5.53344],
                                                        [22.23296, 34.39392]],
                                                       [[86.7539072,
                                                         216.8143744],
                                                        [96.2449152,
                                                         243.2395904],
                                                        [154.54439424,
                                                         260.82114048]]])


