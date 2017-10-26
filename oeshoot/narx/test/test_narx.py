from __future__ import division, print_function, absolute_import
import numpy as np
from oeshoot.narx import NarxModel
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)


class NarxModelForTest(NarxModel):

    def __call__(self, y, u, params):
        return y[0, :]


class TestDerivativeDefault(TestCase):

    def test_derivative_default(self):

        mdl = NarxModelForTest(2, 1, 1, 2, 1)

        dy, du, dparams = mdl.derivatives([[1, 2]], [[1]], [1, 2])

        assert_array_almost_equal(dy, [[[1, 0]], [[0, 1]]])
        assert_array_almost_equal(du, [[[0]], [[0]]])
        assert_array_almost_equal(dparams, [[0, 0], [0, 0]])


class TestInit(TestCase):

    def test_wrong_M_for_the_given_delay(self):

        assert_raises(ValueError, NarxModelForTest, 1, 1, 1, 1, 1, 5)

class TestMarshallingInput(TestCase):

    def test_marshalling_input(self):

        mdl = NarxModelForTest(2, 1, 1, 2, 1)

        x = mdl.marshalling_input([[1, 2], [3, 4], [5, 6]],
                                  [[11, 12, 13], [14, 15, 16]])
        assert_equal(x, [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16])


class TestNarxArgCheck(TestCase):

    def test_check_parameters_size(self):

        mdl = NarxModelForTest(16, 2, 2, 2, 2)
        y = [[1, 2], [2, 3]]
        u = [[2, 3], [3, 4]]
        params = [1, 2, 0, 0,
                  0, 0, 5, 6,
                  3, 4, 0, 0]
        assert_raises(ValueError, mdl._arg_check, y, u, params)

    def test_check_wrong_y_size(self):

        mdl = NarxModelForTest(16, 2, 2, 2, 2)
        y = [[1, 2, 3], [2, 3, 2]]
        u = [[2, 3], [3, 4]]
        params = [1, 2, 0, 0,
                  0, 0, 5, 6,
                  3, 4, 0, 0,
                  3, 4, 0, 0]
        assert_raises(ValueError, mdl._arg_check, y, u, params)

    def test_check_wrong_u_size(self):

        mdl = NarxModelForTest(16, 2, 2, 2, 2)
        y = [[1, 2], [2, 3]]
        u = [[2, 3, 30], [3, 4, 2]]
        params = [1, 2, 0, 0,
                  0, 0, 5, 6,
                  3, 4, 0, 0,
                  3, 4, 0, 0]
        assert_raises(ValueError, mdl._arg_check, y, u, params)

    def test_1d_input(self):

        mdl = NarxModelForTest(4, 2, 2)
        y = [1, 2]
        u = [3, 4]
        params = [1, 2, 3, 4]

        y_new, u_new, params = mdl._arg_check(y, u, params)
        assert_array_almost_equal(y_new, [[1], [2]])
        assert_array_almost_equal(u_new, [[3], [4]])
