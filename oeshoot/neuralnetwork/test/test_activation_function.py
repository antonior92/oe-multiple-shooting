from __future__ import division, print_function, absolute_import
import numpy as np
from oeshoot.neuralnetwork import (Identity, Logistic,
                                HyperbolicTangent)
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)
import numdifftools as nd


class TestIdentity(TestCase):

    def test_identity(self):

        l = Identity()

        for x in np.linspace(-100, 100, 5):
            assert_array_almost_equal(l(x), x)

    def test_identity_vector(self):

        l = Identity()

        x = np.linspace(-100, 100, 5)
        z_numeric = np.zeros(x.shape)
        for i, xi in enumerate(x):
            z_numeric[i] = l(xi)

        assert_array_almost_equal(l(x), z_numeric)

    def test_identity_derivative(self):

        l = Identity()
        dl = nd.Derivative(l)

        for x in np.linspace(-100, 100, 5):
            assert_array_almost_equal(l.derivatives(x), dl(x))

    def test_identity_derivative_vector(self):

        l = Identity()
        dl = nd.Derivative(l)

        x = np.linspace(-100, 100, 5)
        derivative_numeric = np.zeros(x.shape)
        for i, xi in enumerate(x):
            derivative_numeric[i] = dl(xi)

        assert_array_almost_equal(l.derivatives(x), derivative_numeric)


class TestLogistic(TestCase):

    def test_logistic(self):

        l = Logistic()

        for x in np.linspace(-100, 100, 5):
            assert_array_almost_equal(l(x), 1/(1+np.exp(-x)))

    def test_logistic_vector(self):

        l = Logistic()

        x = np.linspace(-100, 100, 5)
        z_numeric = np.zeros(x.shape)
        for i, xi in enumerate(x):
            z_numeric[i] = l(xi)

        assert_array_almost_equal(l(x), z_numeric)

    def test_logistic_derivative(self):

        l = Logistic()
        dl = nd.Derivative(l)

        for x in np.linspace(-100, 100, 5):
            assert_array_almost_equal(l.derivatives(x), dl(x))

    def test_logistic_derivative_vector(self):

        l = Logistic()
        dl = nd.Derivative(l)

        x = np.linspace(-100, 100, 5)
        derivative_numeric = np.zeros(x.shape)
        for i, xi in enumerate(x):
            derivative_numeric[i] = dl(xi)

        assert_array_almost_equal(l.derivatives(x), derivative_numeric)


class TestHyperbolicTangent(TestCase):

    def test_hyperbolic_tangent(self):

        l = HyperbolicTangent()

        for x in np.linspace(-100, 100, 5):
            assert_array_almost_equal(l(x), np.tanh(x))

    def test_hyperbolic_tangent_vector(self):

        l = HyperbolicTangent()

        x = np.linspace(-100, 100, 5)
        z_numeric = np.zeros(x.shape)
        for i, xi in enumerate(x):
            z_numeric[i] = l(xi)

        assert_array_almost_equal(l(x), z_numeric)

    def test_hyperbolic_tangent_derivative(self):

        l = HyperbolicTangent()
        dl = nd.Derivative(l)

        for x in np.linspace(-100, 100, 5):
            assert_array_almost_equal(l.derivatives(x), dl(x))

    def test_hyperbolic_tangent_derivative_vector(self):

        l = HyperbolicTangent()
        dl = nd.Derivative(l)

        x = np.linspace(-100, 100, 5)
        derivative_numeric = np.zeros(x.shape)
        for i, xi in enumerate(x):
            derivative_numeric[i] = dl(xi)

        assert_array_almost_equal(l.derivatives(x), derivative_numeric)
