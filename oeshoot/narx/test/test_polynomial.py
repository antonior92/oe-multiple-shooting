from __future__ import division, print_function, absolute_import
import numpy as np
from oeshoot.narx import Polynomial, Monomial
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)
import numdifftools as nd


# Auxiliar Function
def monomial_numeric_derivatives(monomial, y, u, delay=1,
                                 deriv_y=True, deriv_u=True):
    """
    Evaluate monomial numeric aproximation of the
    derivatives at a given point
    """

    # Check Inputs
    y, u = monomial._arg_check(y, u, delay)

    if deriv_y:
        def fun_y(x):
            return monomial(x.reshape(y.shape), u, delay)
        dy = nd.Jacobian(fun_y)(y.flatten()).reshape(y.shape)

    if deriv_u:
        def fun_u(x):
            return monomial(y, x.reshape(u.shape), delay)
        du = nd.Jacobian(fun_u)(u.flatten()).reshape(u.shape)

    # Returns
    if deriv_y and deriv_u:
        return dy, du
    elif deriv_y:
        return dy
    else:
        return du


class TestPolynomialInit(TestCase):

    def test_siso_atribute_settings(self):
        m1 = Monomial(3, 2, [1, 2, 4], [2, 3],
                      [1, 2, 2], [1, 1])
        m2 = Monomial(1, 0, [1], [], [1], [])
        m3 = Monomial(2, 1, [2, 3], [2], [1, 2], [3])

        m = [m1, m2, m3]
        p = Polynomial(m)
        assert_equal(p.Nparams, 3)
        assert_equal(p.N, 4)
        assert_equal(p.M, 3)
        assert_equal(p.delay, 2)
        assert_equal(p.Ny, 1)
        assert_equal(p.Nu, 1)

    def test_mimo_atribute_settings(self):
        m1 = Monomial(3, 2, [1, 2, 4], [2, 3],
                      [1, 2, 2], [1, 1], [0, 1, 1], [0, 1])
        m2 = Monomial(1, 0, [1], [], [1], [])
        m3 = Monomial(2, 1, [2, 3], [2], [1, 2], [3], [1, 1],
                      [1])

        m = [[m1, m2], [m3]]
        p = Polynomial(m)
        assert_equal(p.Nparams, 3)
        assert_equal(p.N, 4)
        assert_equal(p.M, 3)
        assert_equal(p.delay, 2)
        assert_equal(p.Ny, 2)
        assert_equal(p.Nu, 2)

    def test_noinput_atribute_settings(self):
        m1 = Monomial(1, 0, [1], [], [1], [])
        m2 = Monomial(1, 0, [1], [], [2], [])

        m = [m1, m2]
        p = Polynomial(m)
        assert_equal(p.Nparams, 2)
        assert_equal(p.N, 1)
        assert_equal(p.M, 0)
        assert_equal(p.delay, 0)
        assert_equal(p.Mu, 0)
        assert_equal(p.Ny, 1)
        assert_equal(p.Nu, 0)

    def test_fir_atribute_settings(self):
        m1 = Monomial(0, 1, [], [1], [], [1])
        m2 = Monomial(0, 1, [], [1], [], [2])

        m = [m1, m2]
        p = Polynomial(m)
        assert_equal(p.Nparams, 2)
        assert_equal(p.N, 0)
        assert_equal(p.M, 1)
        assert_equal(p.delay, 1)
        assert_equal(p.Mu, 1)
        assert_equal(p.Ny, 1)
        assert_equal(p.Nu, 1)

    def test_zero_delay_atribute_settings(self):
        m1 = Monomial(0, 1, [], [0], [], [2])

        m = [m1]
        p = Polynomial(m)
        assert_equal(p.Nparams, 1)
        assert_equal(p.N, 0)
        assert_equal(p.M, 0)
        assert_equal(p.delay, 0)
        assert_equal(p.Mu, 1)
        assert_equal(p.Ny, 1)
        assert_equal(p.Nu, 1)

    def test_exception_raised_tipeerror(self):
        m1 = Monomial(3, 2, [1, 2, 4], [2, 3],
                      [1, 2, 2], [1, 1])
        m2 = Monomial(1, 0, [1], [], [1], [])
        m3 = Monomial(2, 1, [2, 3], [1], [1, 2], [3])

        m = [m1, m2, m3, 4]
        assert_raises(TypeError, Polynomial, m)

    def test_exception_raised_unknown_output(self):
        m1 = Monomial(3, 2, [1, 2, 4], [2, 3],
                      [1, 2, 2], [1, 1], [0, 1, 1], [0, 1])
        m2 = Monomial(1, 0, [1], [], [1], [])
        m3 = Monomial(2, 1, [2, 3], [2], [1, 2], [3], [1, 1],
                      [1])

        m = [m1, m2, m3]
        assert_raises(ValueError, Polynomial, m)


class TestPolynomialCall(TestCase):

    def test_siso_polynomial(TestCase):
        m1 = Monomial(2, 2, [1, 2], [1, 2],
                      [1, 2], [1, 1])
        m2 = Monomial(1, 0, [1], [], [1], [])
        m3 = Monomial(2, 1, [2, 3], [2], [1, 2], [3])

        m = [m1, m2, m3]

        p = Polynomial(m)

        y = [[1], [2], [3]]
        u = [[3], [4]]
        params = [1, 2, 3]

        assert_equal(p(y, u, params), [3506])

    def test_mimo_polynomial(TestCase):
        m1 = Monomial(2, 2, [1, 2], [1, 2],
                      [1, 2], [1, 1],
                      [0, 1], [0, 1])
        m2 = Monomial(1, 0, [1], [], [1], [], [1])
        m3 = Monomial(2, 1, [2, 3], [2], [1, 2], [3], [0, 0], [1])

        m = [[m1, m2], [m3]]

        p = Polynomial(m)

        y = [[1, 2], [2, 3], [3, 4]]
        u = [[3, 5], [4, 7]]
        params = [1, 2, 3]

        assert_equal(p(y, u, params), [193, 18522])

    def test_noinput_polynomial(self):
        m1 = Monomial(1, 0, [1], [], [1], [])
        m2 = Monomial(1, 0, [1], [], [2], [])

        m = [m1, m2]
        p = Polynomial(m)

        y = [[1]]
        u = [[]]
        params = [1, -2]

        assert_equal(p(y, u, params), [-1])

    def test_fir_polynomial(self):
        m1 = Monomial(0, 1, [], [1], [], [1])
        m2 = Monomial(0, 1, [], [1], [], [2])

        m = [m1, m2]
        p = Polynomial(m)

        y = [[]]
        u = [[1]]
        params = [1, -2]

        assert_equal(p(y, u, params), [-1])

    def test_zero_delay_polynomial(self):
        m1 = Monomial(0, 1, [], [0], [], [2])

        m = [m1]
        p = Polynomial(m)

        y = [[]]
        u = [[2]]
        params = [2]

        assert_equal(p(y, u, params), [8])


class TestPolynomialDerivatives(TestCase):

    def test_siso_derivatives(TestCase):
        m1 = Monomial(2, 2, [1, 2], [1, 2],
                      [1, 2], [1, 1])
        m2 = Monomial(1, 0, [1], [], [1], [])
        m3 = Monomial(2, 1, [2, 3], [2], [1, 2], [3])

        m = [m1, m2, m3]

        p = Polynomial(m)

        y = [[1], [2], [3]]
        u = [[3], [4]]
        params = [1, 2, 3]

        dy, du, dparams = p.derivatives(y, u, params)

        assert_array_almost_equal(dy, [[[50], [1776], [2304]]])
        assert_array_almost_equal(du, [[[16], [2604]]])
        assert_array_almost_equal(dparams, [[48, 1, 1152]])

    def test_siso_numeric_derivatives(TestCase):
        m1 = Monomial(2, 2, [1, 2], [1, 2],
                      [1, 2], [1, 1])
        m2 = Monomial(1, 0, [1], [], [1], [])
        m3 = Monomial(2, 1, [2, 3], [2], [1, 2], [3])
        m4 = Monomial(3, 1, [2, 3, 5], [3], [1, 2, 2], [2])
        m5 = Monomial(0, 1, [], [3], [], [2])

        m = [m1, m2, m3, m4, m5]

        p = Polynomial(m)

        y = [[12], [2.23], [4.51], [2.1], [3.24]]
        u = [[3.3], [4.45], [3.4]]
        params = [1.8, 2.2, 3.232, 32.3, 3.34]

        dy, du, dparams = p.derivatives(y, u, params)
        dy_numeric, du_numeric, \
            dparams_numeric = p._numeric_derivatives(y, u, params)

        assert_array_almost_equal(dy, dy_numeric)
        assert_array_almost_equal(du, du_numeric)
        assert_array_almost_equal(dparams, dparams_numeric)

    def test_mimo_polynomial(TestCase):
        m1 = Monomial(2, 2, [1, 2], [1, 2],
                      [1, 2], [1, 1],
                      [0, 1], [0, 1])
        m2 = Monomial(1, 0, [1], [], [1], [], [1])
        m3 = Monomial(2, 1, [2, 3], [2], [1, 2], [3], [0, 0], [1])

        m = [[m1, m2], [m3]]

        p = Polynomial(m)

        y = [[1, 2], [2, 3], [3, 4]]
        u = [[3, 5], [4, 7]]
        params = [1, 2, 3]

        dy, du, dparams = p.derivatives(y, u, params)

        assert_array_almost_equal(dy, [[[189, 2], [0, 126], [0, 0]],
                                       [[0, 0], [9261, 0], [12348, 0]]])
        assert_array_almost_equal(du, [[[63, 0], [0, 27]],
                                       [[0, 0], [0, 7938]]])
        assert_array_almost_equal(dparams, [[189, 2, 0],
                                            [0, 0, 6174]])

    def test_mimo_numeric_derivatives(TestCase):
        m1 = Monomial(2, 2, [1, 2], [1, 2],
                      [1, 2], [1, 1], [1, 0])
        m2 = Monomial(1, 0, [1], [], [1], [], [1])
        m3 = Monomial(2, 1, [2, 3], [2], [1, 2], [3], [0, 0], [2])
        m4 = Monomial(3, 1, [2, 3, 5], [3], [1, 2, 2], [2], [0, 1, 0], [0])
        m5 = Monomial(0, 1, [], [3], [], [2], [], [2])

        m = [[m1, m2, m3, m4, m5], [m1, m2]]

        p = Polynomial(m)

        y = [[12, 1.23], [2.23, 23.3], [4.51, 3.4],
             [2.1, 34.54], [3.24, 324.23]]
        u = [[3.3, 132, 1.2], [4.45, 32.32, 2.23], [3.4, 1.3214, 2.132]]
        params = [1.8, 2.2, 3.232, 32.3, 3.34, 3.3425, 13]

        dy, du, dparams = p.derivatives(y, u, params)
        dy_numeric, du_numeric, \
            dparams_numeric = p._numeric_derivatives(y, u, params)

        assert_array_almost_equal(dy, dy_numeric)
        assert_array_almost_equal(du, du_numeric)
        assert_array_almost_equal(dparams, dparams_numeric)

    def test_noinput_derivative(self):
        m1 = Monomial(1, 0, [1], [], [1], [])
        m2 = Monomial(1, 0, [1], [], [2], [])

        m = [m1, m2]
        p = Polynomial(m)

        y = [[1]]
        u = [[]]
        params = [1, -2]

        dy, du, dparams = p.derivatives(y, u, params)

        assert_array_almost_equal(dy, [[[-3]]])
        assert_array_almost_equal(du, np.reshape([], (1, 0, 0)))
        assert_array_almost_equal(dparams, [[1, 1]])

    def test_noinput_numeric_derivative(self):
        m1 = Monomial(1, 0, [1], [], [1], [])
        m2 = Monomial(1, 0, [1], [], [2], [])

        m = [m1, m2]
        p = Polynomial(m)

        y = [[1]]
        u = [[]]
        params = [1, -2]

        dy, du, dparams = p.derivatives(y, u, params)
        dy_numeric, du_numeric, \
            dparams_numeric = p._numeric_derivatives(y, u, params)

        assert_array_almost_equal(dy, dy_numeric)
        assert_array_almost_equal(du, du_numeric)
        assert_array_almost_equal(dparams, dparams_numeric)

    def test_fir_derivative(self):
        m1 = Monomial(0, 1, [], [1], [], [1])
        m2 = Monomial(0, 1, [], [1], [], [2])

        m = [m1, m2]
        p = Polynomial(m)

        y = [[]]
        u = [[1]]
        params = [1, -2]

        dy, du, dparams = p.derivatives(y, u, params)

        assert_array_almost_equal(du, [[[-3]]])
        assert_array_almost_equal(dy, np.reshape([], (1, 0, 1)))
        assert_array_almost_equal(dparams, [[1, 1]])

    def test_fir_numeric_derivative(self):
        m1 = Monomial(0, 1, [], [1], [], [1])
        m2 = Monomial(0, 1, [], [1], [], [2])

        m = [m1, m2]
        p = Polynomial(m)

        y = [[]]
        u = [[1]]
        params = [1, -2]

        dy, du, dparams = p.derivatives(y, u, params)
        dy_numeric, du_numeric, \
            dparams_numeric = p._numeric_derivatives(y, u, params)

        assert_array_almost_equal(dy, dy_numeric)
        assert_array_almost_equal(du, du_numeric)
        assert_array_almost_equal(dparams, dparams_numeric)

    def test_zero_delay_derivative(self):
        m1 = Monomial(0, 1, [], [0], [], [2])

        m = [m1]
        p = Polynomial(m)

        y = [[]]
        u = [[1]]
        params = [2]

        dy, du, dparams = p.derivatives(y, u, params)

        assert_array_almost_equal(du, [[[4]]])
        assert_array_almost_equal(dy, np.reshape([], (1, 0, 1)))
        assert_array_almost_equal(dparams, [[1]])

    def test_zero_delay_numeric_derivative(self):
        m1 = Monomial(0, 1, [], [0], [], [2])

        m = [m1]
        p = Polynomial(m)

        y = [[]]
        u = [[1]]
        params = [2]

        dy, du, dparams = p.derivatives(y, u, params)
        dy_numeric, du_numeric, \
            dparams_numeric = p._numeric_derivatives(y, u, params)

        assert_array_almost_equal(dy, dy_numeric)
        assert_array_almost_equal(du, du_numeric)
        assert_array_almost_equal(dparams, dparams_numeric)


class TestMonomialInit(TestCase):

    def test_siso_atribute_setting(self):
        m = Monomial(3, 2, [1, 2, 4], [2, 3],
                     [1, 2, 2], [1, 1])
        assert_equal(m.N, 4)
        assert_equal(m.M, 3)
        assert_equal(m.minlag_u, 2)
        assert_equal(m.Ny, 1)
        assert_equal(m.Nu, 1)
        assert_equal(m.Nl, 7)

    def test_mimo_atribute_setting(self):
        m = Monomial(3, 2, [1, 2, 4], [2, 3],
                     [1, 2, 2], [1, 1],
                     [0, 1, 2], [3, 0])
        assert_equal(m.N, 4)
        assert_equal(m.M, 3)
        assert_equal(m.minlag_u, 2)
        assert_equal(m.Ny, 3)
        assert_equal(m.Nu, 4)
        assert_equal(m.Nl, 7)

    def test_singleterm_siso_atribute_setting(self):

        m = Monomial(1, 0, [1], [], [1], [])
        assert_equal(m.N, 1)
        assert_equal(m.M, 0)
        assert_equal(m.minlag_u, np.Inf)
        assert_equal(m.Ny, 1)
        assert_equal(m.Nu, 0)
        assert_equal(m.Nl, 1)

    def test_singleterm_mimo_atribute_setting(self):

        m = Monomial(0, 1, [], [2], [], [2], [], [3])
        assert_equal(m.N, 0)
        assert_equal(m.M, 2)
        assert_equal(m.minlag_u, 2)
        assert_equal(m.Ny, 0)
        assert_equal(m.Nu, 4)
        assert_equal(m.Nl, 2)

    def test_exception_raised(self):
        assert_raises(ValueError, Monomial, 3, 2,
                      [1, 2], [2, 3],
                      [1, 2, 2], [1, 1],
                      [0, 1, 2], [3, 0])
        assert_raises(ValueError, Monomial, 3, 2,
                      [1, 2, 4], [2, 3, 4],
                      [1, 2, 2], [1, 1],
                      [0, 1, 2], [3, 0])
        assert_raises(ValueError, Monomial, 3, 2,
                      [1, 2, 4], [2, 3],
                      [1, 2, 2, 3], [1, 1],
                      [0, 1, 2], [3, 0])
        assert_raises(ValueError, Monomial, 3, 2,
                      [1, 2, 4], [2, 3],
                      [1, 2, 2], [1, 1, 1],
                      [0, 1, 2], [3, 0])
        assert_raises(ValueError, Monomial, 3, 2,
                      [1, 2, 4], [2, 3],
                      [1, 2, 2], [1, 1],
                      [0, 1, 2], [3])
        assert_raises(ValueError, Monomial, 3, 2,
                      [1, 2, 4], [2, 3],
                      [1, 2, 2], [1, 1],
                      [0, 1, 2, 0], [3, 1])


class TestMonomialCall(TestCase):

    def test_singleterm_siso_monomial_evaluation(self):

        m = Monomial(1, 0, [2], [], [2], [])
        y = [[1.12], [2.12]]
        u = []
        assert_array_almost_equal(m(y, u), 2.12**2)

    def test_singleterm_mimo_monomial_evaluation(self):

        m = Monomial(0, 1, [], [2], [], [2], [], [3])

        y = []
        u = [[1, 2, 3, 7], [4, 5, 6, 8]]

        assert_array_almost_equal(m(y, u), 8**2)

    def test_siso_monomial_evaluation(self):

        m = Monomial(3, 2, [1, 2, 3], [4, 5], [1, 1, 1], [2, 2])
        y = [[1.12], [2.12], [3]]
        u = [[2], [2]]
        assert_array_almost_equal(m(y, u, 4), 1.12*2.12*3*(2**2)*(2**2))

        m = Monomial(3, 2, [1, 2, 3], [4, 5])
        y = [[1.12], [2.12], [3]]
        u = [[2], [2]]
        assert_array_almost_equal(m(y, u, 4), 1.12*2.12*3*2*2)

    def test_mimo_monomial_evaluation(self):

        m = Monomial(3, 2, [1, 2, 3], [4, 5], [1, 1, 1], [2, 2],
                     [1, 2, 3], [0, 1])
        y = [[1.12, 3, 4, 5], [2.12, 7, 8, 9], [3, 10, 11, 12]]
        u = [[2, 1], [2, 3.2]]
        assert_array_almost_equal(m(y, u, 4), 3*8*12*(2**2)*(3.2**2))


class TestMonomialDerivatives(TestCase):

    def test_singleterm_siso_monomial_derivatives(self):

        m = Monomial(1, 0, [2], [], [2], [])
        y = [[1.12], [2.12]]
        u = []
        dy, du = m.derivatives(y, u, 1, True, True)
        assert_array_almost_equal(dy, [[0], [2.12*2]])

    def test_singleterm_siso_numeric_derivatives(self):

        m = Monomial(1, 0, [3], [], [4], [])
        y = [[1.12], [123.123], [1.2]]
        u = []
        dy = m.derivatives(y, u, 1, deriv_u=False)
        dy_numeric = monomial_numeric_derivatives(m, y, u, 1, deriv_u=False)
        assert_array_almost_equal(dy, dy_numeric)

    def test_singleterm_mimo_monomial_derivatives(self):

        m = Monomial(0, 1, [], [2], [], [3], [], [3])
        y = []
        u = [[1, 2, 3, 7], [4, 5, 6, 8]]
        dy, du = m.derivatives(y, u, 1, True, True)
        assert_array_almost_equal(du, [[0, 0, 0, 0], [0, 0, 0, 3*8**2]])

    def test_singleterm_mimo_numeric_derivatives(self):

        m = Monomial(0, 1, [], [2], [], [4], [], [3])
        y = []
        u = [[1, 2.12, 3, 7], [4, 5, 6, 8]]
        du = m.derivatives(y, u, 1, deriv_y=False)
        du_numeric = monomial_numeric_derivatives(m, y, u, 1, deriv_y=False)
        assert_array_almost_equal(du, du_numeric)

    def test_siso_monomial_derivatives(self):

        m = Monomial(3, 2, [1, 2, 3], [4, 5], [1, 1, 1], [3, 3])
        y = [[1.12], [2.12], [3]]
        u = [[2], [2]]
        dy, du = m.derivatives(y, u, 4)
        assert_array_almost_equal(dy, [[2.12*3*(2**3)*(2**3)],
                                       [1.12*3*(2**3)*(2**3)],
                                       [1.12*2.12*(2**3)*(2**3)]])
        assert_array_almost_equal(du, [[1.12*2.12*3*(3*2**2)*(2**3)],
                                       [1.12*2.12*3*(2**3)*(3*2**2)]])

    def test_siso_numeric_derivatives(self):

        m = Monomial(2, 3, [1, 2], [4, 5, 6])
        y = [[1.12], [2.12]]
        u = [[21.1], [2.1], [3.4]]
        dy, du = m.derivatives(y, u, 4)
        dy_numeric, du_numeric = monomial_numeric_derivatives(m, y, u, 4)
        assert_array_almost_equal(dy, dy_numeric)
        assert_array_almost_equal(du, du_numeric)

    def test_mimo_monomial_derivatives(self):

        m = Monomial(3, 2, [1, 2, 3], [4, 5], [1, 1, 1], [2, 2],
                     [1, 2, 3], [0, 1])
        y = [[1.12, 3, 4, 5], [2.12, 7, 8, 9], [3, 10, 11, 12]]
        u = [[2, 1], [2, 3.2]]
        dy, du = m.derivatives(y, u, 4)
        assert_array_almost_equal(dy, [[0, 8*12*(2**2)*(3.2**2), 0, 0],
                                       [0, 0, 3*12*(2**2)*(3.2**2), 0],
                                       [0, 0, 0, 3*8*(2**2)*(3.2**2)]])
        assert_array_almost_equal(du, [[3*8*12*(2*2)*(3.2**2), 0],
                                       [0, 3*8*12*(2**2)*(3.2*2)]])

    def test_mimo_numeric_derivatives(self):

        m = Monomial(3, 2, [1, 2, 3], [4, 5], [1, 2, 1], [2, 2],
                     [0, 2, 3], [0, 1])
        y = [[1.12, 3, 4, 5], [2.12, 7, 8, 9], [3, 10, 11, 12]]
        u = [[2, 1], [2, 3.2]]
        dy, du = m.derivatives(y, u, 4)
        dy_numeric, du_numeric = monomial_numeric_derivatives(m, y, u, 4)
        assert_array_almost_equal(dy, dy_numeric)
        assert_array_almost_equal(du, du_numeric)
