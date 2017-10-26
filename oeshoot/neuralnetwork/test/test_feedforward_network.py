from __future__ import division, print_function, absolute_import
import numpy as np
from oeshoot.neuralnetwork import (FeedforwardNetwork, Logistic,
                                    HyperbolicTangent, Identity)
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less,
                           assert_raises, assert_equal, assert_,
                           run_module_suite, assert_allclose, assert_warns,
                           dec)
import numdifftools as nd


class TestFeedforwardNetworkInit(TestCase):

    def test_single_activation_function(self):

        activ_func = Logistic()
        net = FeedforwardNetwork(ninput=3, noutput=2,
                                 nhidden=[2, 3],
                                 activ_func=activ_func)

        assert_equal(net.ninput, 3)
        assert_equal(net.noutput, 2)
        assert_equal(net.nhidden, [2, 3])
        assert_equal(net.weights_shape, [(2, 3),
                                         (3, 2),
                                         (2, 3)])
        assert_equal(net.bias_shape, [(2, 1),
                                      (3, 1),
                                      (2, 1)])
        assert_equal(net.nparams, 25)
        assert_equal(net.nlayers, 3)
        assert_equal(isinstance(net.activ_func[0], Logistic), True)
        assert_equal(isinstance(net.activ_func[1], Logistic), True)
        assert_equal(isinstance(net.activ_func[2], Identity), True)

    def test_listof_activation_function(self):

        net = FeedforwardNetwork(ninput=3, noutput=2,
                                nhidden=[2, 3, 4],
                                activ_func=[Logistic(),
                                            HyperbolicTangent(),
                                            Identity(),
                                            Logistic()])

        assert_equal(net.ninput, 3)
        assert_equal(net.noutput, 2)
        assert_equal(net.nhidden, [2, 3, 4])
        assert_equal(net.weights_shape, [(2, 3),
                                         (3, 2),
                                         (4, 3),
                                         (2, 4)])
        assert_equal(net.bias_shape, [(2, 1),
                                      (3, 1),
                                      (4, 1),
                                      (2, 1)])
        assert_equal(net.nlayers, 4)
        assert_equal(net.nparams, 43)
        assert_equal(isinstance(net.activ_func[0], Logistic), True)
        assert_equal(isinstance(net.activ_func[1], HyperbolicTangent), True)
        assert_equal(isinstance(net.activ_func[2], Identity), True)
        assert_equal(isinstance(net.activ_func[3], Logistic), True)

    def test_raise_exception(self):

        assert_raises(ValueError,
                      FeedforwardNetwork,
                      ninput=3, noutput=2,
                      nhidden=[2, 3],
                      activ_func=[Logistic(),
                                  "bsdfad"])
        assert_raises(ValueError,
                      FeedforwardNetwork,
                      ninput=3, noutput=2,
                      nhidden=[2, 3],
                      activ_func=[Logistic()])


class TestDisassembleParams(TestCase):

    def test_disassemble_params(self):

        net = FeedforwardNetwork(ninput=3, noutput=2,
                                 nhidden=[2, 3],
                                 activ_func=Logistic())

        params = [1111, 1112, 1113,
                  1121, 1122, 1123,
                  121, 122,
                  2111, 2112,
                  2121, 2122,
                  2131, 2132,
                  221, 222, 223,
                  3111, 3112, 3113,
                  3121, 3122, 3123,
                  321, 322]

        w, b = net.disassemble_params(params)

        W0 = [[1111, 1112, 1113],
              [1121, 1122, 1123]]
        W1 = [[2111, 2112],
              [2121, 2122],
              [2131, 2132]]
        W2 = [[3111, 3112, 3113],
              [3121, 3122, 3123]]
        B0 = [[121], [122]]
        B1 = [[221], [222], [223]]
        B2 = [[321], [322]]

        assert_array_almost_equal(w[0], W0)
        assert_array_almost_equal(w[1], W1)
        assert_array_almost_equal(w[2], W2)
        assert_array_almost_equal(b[0], B0)
        assert_array_almost_equal(b[1], B1)
        assert_array_almost_equal(b[2], B2)

    def test_disassemble_params_raise_exception(self):

        net = FeedforwardNetwork(ninput=3, noutput=2,
                                 nhidden=[2, 3],
                                 activ_func=Logistic())

        params = [1111]

        assert_raises(ValueError, net.disassemble_params, params)


class TestAssembleParams(TestCase):

    def test_assemble_params(self):

        net = FeedforwardNetwork(ninput=3, noutput=2,
                                 nhidden=[2, 3],
                                 activ_func=Logistic())

        expected_params = [1111, 1112, 1113,
                           1121, 1122, 1123,
                           121, 122,
                           2111, 2112,
                           2121, 2122,
                           2131, 2132,
                           221, 222, 223,
                           3111, 3112, 3113,
                           3121, 3122, 3123,
                           321, 322]

        W0 = [[1111, 1112, 1113],
              [1121, 1122, 1123]]
        W1 = [[2111, 2112],
              [2121, 2122],
              [2131, 2132]]
        W2 = [[3111, 3112, 3113],
              [3121, 3122, 3123]]
        B0 = [[121], [122]]
        B1 = [[221], [222], [223]]
        B2 = [[321], [322]]

        params = net.assemble_params([W0, W1, W2], [B0, B1, B2])

        assert_array_almost_equal(params, expected_params)

    def test_assemble_params_raise_exception(self):

        net = FeedforwardNetwork(ninput=3, noutput=2,
                                 nhidden=[2, 3],
                                 activ_func=Logistic())

        expected_params = [1111, 1112, 1113,
                           1121, 1122, 1123,
                           121, 122,
                           2111, 2112,
                           2121, 2122,
                           2131, 2132,
                           221, 222, 223,
                           3111, 3112, 3113,
                           3121, 3122, 3123,
                           321, 322]

        W0 = [[1111, 1112, 1113],
              [1121, 1122, 1123]]
        W1 = [[2111, 2112],
              [2121, 2122],
              [2131, 2132]]
        W2 = [[3111, 3112, 3113],
              [3121, 3122, 3123]]
        B0 = [[121], [122]]
        B1 = [[221], [222], [223]]
        B2 = [[321], [322]]

        assert_raises(ValueError, net.assemble_params, [W0, W1], [B0, B1, B2])
        assert_raises(ValueError, net.assemble_params, [W0, W1, W2], [B1, B2])
        assert_raises(ValueError, net.assemble_params, [W0, W1, W2],
                      [[1, 2, 3], B1, B2])
        assert_raises(ValueError, net.assemble_params, [[1, 2, 3], W1, W2],
                      [B0, B1, B2])


class TestFeedforwardNetworkCall(TestCase):

    def test_two_hiddenlayer_three_input_two_output(self):

        l = Logistic()
        i = Identity()

        net = FeedforwardNetwork(ninput=3, noutput=2,
                                 nhidden=[2, 3],
                                 activ_func=Logistic())

        params = np.array([1.111, 1.112, 11.13,
                           112.1, 0.1122, 1123,
                           12.1, 1.22,
                           0.2111, 0.2112,
                           2.121, 2.122,
                           0.2131, 0.2132,
                           2.21, 2.22, 0.223,
                           3.111, 0.3112, 3.113,
                           0.3121, 0.3122, 0.3123,
                           0.321, 3.22])

        w, b = net.disassemble_params(params)

        innet = np.asarray([1, 2, 3]).reshape(3, 1)
        expected = i(np.dot(w[2], l(np.dot(w[1], l(np.dot(w[0], innet)+b[0]))+b[1]))+b[2]).flatten()
        assert_array_almost_equal(net(innet, params), expected, decimal=3)

        innet = np.asarray([1.78, -2, 3]).reshape(3, 1)
        expected = i(np.dot(w[2], l(np.dot(w[1], l(np.dot(w[0], innet)+b[0]))+b[1]))+b[2]).flatten()
        assert_array_almost_equal(net(innet, params), expected, decimal=3)

        innet = np.asarray([1, 25, 3]).reshape(3, 1)
        expected = i(np.dot(w[2], l(np.dot(w[1], l(np.dot(w[0], innet)+b[0]))+b[1]))+b[2]).flatten()
        assert_array_almost_equal(net(innet, params), expected, decimal=3)

    def test_raise_exception(self):

        net = FeedforwardNetwork(ninput=3, noutput=2,
                                 nhidden=[2, 3],
                                 activ_func=Logistic())

        params = np.array([1.111, 1.112, 11.13,
                           112.1, 0.1122, 1123,
                           12.1, 1.22,
                           0.2111, 0.2112,
                           2.121, 2.122,
                           0.2131, 0.2132,
                           2.21, 2.22, 0.223,
                           3.111, 0.3112, 3.113,
                           0.3121, 0.3122, 0.3123,
                           0.321, 3.22])

        assert_raises(ValueError, net, [1, 2, 3, 4], params)

        assert_raises(ValueError, net, [1, 2, 3], params[:-1])


class TestFeedforwardNetworkDerivatives(TestCase):

    def test_two_hiddenlayer_three_input_two_output(self):

        net = FeedforwardNetwork(ninput=3, noutput=2,
                                 nhidden=[2, 3],
                                 activ_func=Logistic())

        params = np.array([1.111, 1.112, 11.13,
                           1.121, 0.1122, 1123,
                           1.21, 1.22,
                           0.2111, 0.2112,
                           2.121, 2.122,
                           0.2131, 0.2132,
                           2.21, 2.22, 0.223,
                           3.111, 0.3112, 3.113,
                           0.3121, 0.3122, 0.3123,
                           0.321, 3.22])

        x = [0.1, 2.2, 0.34]

        def call_x(y):
            return net(y, params)

        def call_params(theta):
            return net(x, theta)

        dx_numeric = nd.Jacobian(call_x)(x)
        dparams_numeric = nd.Jacobian(call_params)(params)

        dx, dparams = net.derivatives(x, params)

        assert_array_almost_equal(dx, dx_numeric)
        assert_array_almost_equal(dparams, dparams_numeric)

    def test_three_hiddenlayer_three_input_two_output(self):

        net = FeedforwardNetwork(ninput=3, noutput=2,
                                 nhidden=[2, 3, 4],
                                 activ_func=[Logistic(),
                                             HyperbolicTangent(),
                                             Identity(),
                                             Logistic()])

        for i in range(5):
            params = np.random.normal(size=(net.nparams,))

            x = np.random.normal(size=(net.ninput,))

            def call_x(y):
                return net(y, params)

            def call_params(theta):
                return net(x, theta)

            dx_numeric = nd.Jacobian(call_x)(x)
            dparams_numeric = nd.Jacobian(call_params)(params)

            dx, dparams = net.derivatives(x, params)

            assert_array_almost_equal(dx, dx_numeric)
            assert_array_almost_equal(dparams, dparams_numeric)

    def test_raise_exception(self):

        net = FeedforwardNetwork(ninput=3, noutput=2,
                                 nhidden=[2, 3],
                                 activ_func=Logistic())

        params = np.array([1.111, 1.112, 11.13,
                           112.1, 0.1122, 1123,
                           12.1, 1.22,
                           0.2111, 0.2112,
                           2.121, 2.122,
                           0.2131, 0.2132,
                           2.21, 2.22, 0.223,
                           3.111, 0.3112, 3.113,
                           0.3121, 0.3122, 0.3123,
                           0.321, 3.22])

        assert_raises(ValueError, net.derivatives, [1, 2, 3, 4], params)

        assert_raises(ValueError, net.derivatives, [1, 2, 3], params[:-1])
