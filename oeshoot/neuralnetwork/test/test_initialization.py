from __future__ import division, print_function, absolute_import
import numpy as np
from oeshoot.neuralnetwork import (FeedforwardNetwork, LeCunInitializer,
                                    Logistic, scale_input_layer,
                                    scale_output_layer)
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)


class TestLeCunInitializer(TestCase):

    def test_raise_exception(self):

        assert_raises(ValueError, LeCunInitializer, distribution='blab')
        assert_raises(ValueError, LeCunInitializer, gain=-1)
        assert_raises(ValueError, LeCunInitializer, gain=0)

    def test_shape_check(self):

        net = FeedforwardNetwork(ninput=3, noutput=2,
                                 nhidden=[2, 3],
                                 activ_func=Logistic())
        weights, bias = LeCunInitializer()(net)

        # Check shape
        assert_equal(weights[0].shape, (2, 3))
        assert_equal(weights[1].shape, (3, 2))
        assert_equal(weights[2].shape, (2, 3))
        assert_equal(bias[0].shape, (2, 1))
        assert_equal(bias[1].shape, (3, 1))
        assert_equal(bias[2].shape, (2, 1))

    def test_normal_distribution_std_check(self):

        nexp = 1000

        net = FeedforwardNetwork(ninput=3, noutput=2,
                                 nhidden=[2, 3],
                                 activ_func=Logistic())

        mean_per_layer = np.zeros(net.nlayers)
        variance_per_layer = np.zeros(net.nlayers)
        for i in range(nexp):

            np.random.seed(i)
            weights, bias = LeCunInitializer()(net)

            for j in range(net.nlayers):
                w = weights[j]
                we = w.shape[0]*w.shape[1]
                mean_per_layer[j] += 1/(nexp*we)*np.sum(w)
                variance_per_layer[j] += 1/(nexp*we)*np.sum(w**2)

        std_per_layer = np.sqrt(variance_per_layer)

        assert_array_almost_equal(std_per_layer, [1/np.sqrt(3),
                                                  1/np.sqrt(2),
                                                  1/np.sqrt(3)], decimal=2)
        assert_array_almost_equal(mean_per_layer, [0, 0, 0], decimal=2)

    def test_uniform_distribution_std_check(self):

        nexp = 1000

        net = FeedforwardNetwork(ninput=3, noutput=2,
                                 nhidden=[2, 3],
                                 activ_func=Logistic())

        mean_per_layer = np.zeros(net.nlayers)
        variance_per_layer = np.zeros(net.nlayers)
        for i in range(nexp):

            np.random.seed(i*2)
            weights, bias = LeCunInitializer(distribution='uniform')(net)

            for j in range(net.nlayers):
                w = weights[j]
                we = w.shape[0]*w.shape[1]
                mean_per_layer[j] += 1/(nexp*we)*np.sum(w)
                variance_per_layer[j] += 1/(nexp*we)*np.sum(w**2)

        std_per_layer = np.sqrt(variance_per_layer)

        assert_array_almost_equal(std_per_layer, [1/np.sqrt(3),
                                                  1/np.sqrt(2),
                                                  1/np.sqrt(3)], decimal=2)
        assert_array_almost_equal(mean_per_layer, [0, 0, 0], decimal=2)
