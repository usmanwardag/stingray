
from nose.tools import eq_

import numpy as np

from stingray import parametricmodels
from stingray import Const, PowerLaw, BrokenPowerLaw, Lorentzian
from stingray import CombinedModel

logmin = parametricmodels.logmin

class TestConstModel(object):
    def setUp(self):
        self.x = np.arange(1000)
        self.const = Const()

    def test_length(self):
        a = 2.0
        assert self.const(self.x,a).shape == self.x.shape

    def test_value(self):
        a = 2.0
        all(self.const(self.x, a)) == a


class TestPowerLawModel(object):

    def setUp(self):
        self.x = np.arange(1000)
        self.pl = PowerLaw()

    def test_shape(self):
        alpha = 2.0
        amplitude = 3.0

        assert self.pl(self.x, alpha, amplitude).shape == self.x.shape


    def test_value(self):
        pl_eqn = lambda x, i, a: np.exp(-i*np.log(x) + a)

        alpha = 2.0
        amplitude = 3.0

        for x in range(1,10):
            eq_(pl_eqn(x, alpha, amplitude), self.pl(x, alpha, amplitude))


class TestBentPowerLawModel(object):

    def setUp(self):
        self.x = np.arange(1000)
        self.bpl = BrokenPowerLaw()

    def test_shape(self):
        alpha1 = 1.0
        amplitude = 3.0
        alpha2 = 3.0
        x_break = 5.0

        c = self.bpl(self.x, alpha1, amplitude, alpha2, x_break)
        assert c.shape == self.x.shape


    def test_value(self):
        ## TODO: Need to write a meaningful test for this
        alpha1 = 1.0
        amplitude = 3.0
        alpha2 = 3.0
        x_break = 5.0

        bpl_eqn = lambda x, i, a: np.exp(-i*np.log(x) + a)
        pass


class TestLorentzianModel(object):

    def setUp(self):
        self.x = np.arange(1000)
        self.lorentzian = Lorentzian()

    def test_shape(self):
        gamma = 1.0
        amplitude = 2.0
        x0 = 200.0

        c = self.lorentzian(self.x, gamma, amplitude, x0)
        assert c.shape == self.x.shape


    def test_value(self):
        gamma = 1.0
        amplitude = 2.0
        x0 = 200.0

        qpo_func = lambda x, g, amp, cen: (np.exp(amp)/np.pi)*0.5*np.exp(g)/\
                                          ((x-cen)**2.0+(0.5*np.exp(g))**2.0)
        for x in range(1, 20):
            assert np.allclose(qpo_func(x, gamma, amplitude, x0),
                               self.lorentzian(x, x0, gamma, amplitude),
                               atol=1.e-10)


class TestCombinedModels(object):

    def setUp(self):
        self.x = np.arange(1000)


        ## number of parameters for the different models
        self.npar_const = 1
        self.npar_powerlaw = 2
        self.npar_bentpowerlaw = 4
        self.npar_lorentzian = 3


    def npar_equal(self, model1, model2):
        mod = CombinedModel([model1, model2])
        npar_model1 = self.__getattribute__("npar_"+mod.name[0])
        npar_model2 = self.__getattribute__("npar_"+mod.name[1])
        eq_(mod.npar, npar_model1+npar_model2)


    def test_model(self):
        """
        gamma = 0.5
        norm = 3.0
        x0 = 200.0
        a = 2.0
        """
        models = [Const,
                PowerLaw,
                BrokenPowerLaw,
                Lorentzian]

        for m1 in models:
            for m2 in models:
                self.npar_equal(m1, m2)



class TestConstPrior(object):

    def setUp(self):
        self.hyperpars = {"a_mean": 2.0, "a_var": 0.1}
        self.const = Const(self.hyperpars)

    def test_prior_nonzero(self):
        a = 2.0
        assert self.const.logprior(a) > logmin

    def test_prior_zero(self):
        a = 100.0
        assert self.const.logprior(a) == logmin



class TestPowerlawPrior(object):
    def setUp(self):
        self.hyperpars = {"alpha_min":-8.0, "alpha_max":5.0,
                          "amplitude_min": -10.0, "amplitude_max":10.0}

        alpha_norm = 1.0/(self.hyperpars["alpha_max"]-
                          self.hyperpars["alpha_min"])
        amplitude_norm = 1.0/(self.hyperpars["amplitude_max"]-
                              self.hyperpars["amplitude_min"])
        self.prior_norm = np.log(alpha_norm*amplitude_norm)

        self.pl = parametricmodels.PowerLaw(self.hyperpars)

    def test_prior_nonzero(self):
        alpha = 1.0
        amplitude = 2.0
        print(self.pl)
        assert self.pl.logprior(alpha, amplitude) == self.prior_norm

    def prior_zero(self, alpha, amplitude):
        assert self.pl.logprior(alpha, amplitude) == logmin

    def generate_prior_zero_tests(self):
        alpha_all = [1.0, 10.0]
        amplitude_all = [-20.0, 2.0]
        for alpha, amplitude in zip(alpha_all, amplitude_all):
            yield self.prior_zero, alpha, amplitude



class TestBentPowerLawPrior(object):

    def setUp(self):

        self.hyperpars = {"alpha1_min": -8.0, "alpha1_max":5.0,
                 "amplitude_min": -10., "amplitude_max":10.0,
                 "alpha2_min":-8.0, "alpha2_max":4.0,
                 "x_break_min":np.log(0.1), "x_break_max":np.log(500)}

        alpha1_norm = 1.0/(self.hyperpars["alpha1_max"]-
                           self.hyperpars["alpha1_min"])

        alpha2_norm = 1.0/(self.hyperpars["alpha2_max"]-
                           self.hyperpars["alpha2_min"])

        amplitude_norm = 1.0/(self.hyperpars["amplitude_max"]-
                              self.hyperpars["amplitude_min"])

        x_break_norm = 1.0/(self.hyperpars["x_break_max"]-
                            self.hyperpars["x_break_min"])

        self.prior_norm = np.log(alpha1_norm*alpha2_norm*
                                 amplitude_norm*x_break_norm)

        self.bpl = BrokenPowerLaw(self.hyperpars)


    def zero_prior(self, alpha1, amplitude, alpha2, x_break):
        assert self.bpl.logprior(alpha1, alpha2, x_break, amplitude) == logmin

    def nonzero_prior(self, alpha1, amplitude, alpha2, x_break):
        assert self.bpl.logprior(alpha1, alpha2, x_break, amplitude) == \
               self.prior_norm


    def test_prior(self):

        alpha1 = [1.0, 10.0]
        alpha2 = [1.0, 10.0]
        amplitude = [2.0, -20.0]
        x_break = [np.log(50.0), np.log(1000.0)]

        for i, a1 in enumerate(alpha1):
            for j, amp in enumerate(amplitude):
                for k, a2 in enumerate(alpha2):
                    for l, br in enumerate(x_break):
                        if i == 1 or j == 1 or k == 1 or l == 1:
                            yield self.zero_prior, a1, amp, a2, br
                        else:
                            yield self.nonzero_prior, a1, amp, a2, br




class TestLorentzianPrior(object):

    def setUp(self):

        self.hyperpars = {"gamma_min":-1.0, "gamma_max":5.0,
                     "amplitude_min":-10.0, "amplitude_max":10.0,
                     "x0_min":0.0, "x0_max":100.0}

        gamma_norm = 1.0/(self.hyperpars["gamma_max"]-
                          self.hyperpars["gamma_min"])

        amplitude_norm = 1.0/(self.hyperpars["amplitude_max"]-
                              self.hyperpars["amplitude_min"])

        x0_norm = 1.0/(self.hyperpars["x0_max"]-self.hyperpars["x0_min"])
        self.prior_norm = np.log(gamma_norm*amplitude_norm*x0_norm)
        self.lorentzian = Lorentzian(self.hyperpars)


    def zero_prior(self, gamma, amplitude, x0):
        assert self.lorentzian.logprior(x0, gamma, amplitude) == logmin

    def nonzero_prior(self, gamma, amplitude, x0):
        assert self.lorentzian.logprior(x0, gamma, amplitude) == self.prior_norm


    def test_prior(self):

        gamma = [2.0, -10.0]
        amplitude = [5.0, -20.0]
        x0 = [10.0, -5.0]

        for i,g in enumerate(gamma):
            for j,a in enumerate(amplitude):
                for k, x in enumerate(x0):
                    pars = [g, a, x]
                    if i == 1 or j == 1 or k == 1:
                        yield self.zero_prior, g, a, x
                    else:
                        yield self.nonzero_prior, g, a, x



