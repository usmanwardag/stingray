import numpy as np

from stingray import Powerspectrum, AveragedPowerspectrum
from stingray.parametricmodels import logmin

class Posterior(object):

    def __init__(self,x, y, model):
        self.x = x
        self.y = y

        ### model is a parametric model
        self.model = model

    def logprior(self, t0):
        print("If you're calling this method, something is wrong!")
        return 0.0


    ### use standard definition of the likelihood as the product of all
    def loglikelihood(self, t0, neg=False):
        print("If you're calling this method, something is wrong!")
        return 0.0

    def __call__(self, t0, neg=False):
        lpost = self.loglikelihood(t0) + self.logprior(t0)

        if neg == True:
            return -lpost
        else:
            return lpost


class PSDPosterior(Posterior):

    def __init__(self, ps, model):
        """
        Posterior distribution for power spectra.
        Uses an exponential distribution for the errors in the likelihood,
        or a $\chi^2$ distribution with $2M$ degrees of freedom, where $M$ is
        the number of frequency bins or power spectra averaged in each bin.


        Parameters
        ----------
        ps: {Powerspectrum | AveragedPowerspectrum} instance
            the Powerspectrum object containing the data

        model: instance of any subclass of parameterclass.ParametricModel
            The model for the power spectrum. Note that in order to define
            the posterior properly, the ParametricModel subclass must be
            instantiated with the hyperpars parameter set, or there won't
            be a prior to be calculated!


        Attributes
        ----------
        ps: {Powerspectrum | AveragedPowerspectrum} instance
            the Powerspectrum object containing the data

        m: int, optional, default is 1
            The number of averaged periodograms or frequency bins in ps.
            Useful for binned/averaged periodograms, since the value of
            m will change the likelihood function!

        x: numpy.ndarray
            The independent variable (list of frequencies) stored in ps.freq

        y: numpy.ndarray
            The dependent variable (list of powers) stored in ps.ps

        model: instance of any subclass of parameterclass.ParametricModel
               The model for the power spectrum. Note that in order to define
               the posterior properly, the ParametricModel subclass must be
               instantiated with the hyperpars parameter set, or there won't
               be a prior to be calculated!

        """
        self.ps = ps
        self.m = ps.m
        Posterior.__init__(self,ps.freq[1:], ps.ps[1:], model)

    def logprior(self, t0):
        """
        The logarithm of the prior distribution for the
        model defined in self.model.

        Parameters:
        ------------
        t0: {list | numpy.ndarray}
            The list with parameters for the model

        Returns:
        --------
        logp: float
            The logarithm of the prior distribution for the model and
            parameters given.
        """
        assert hasattr(self.model, "logprior")
        assert np.size(t0) == self.model.npar, "Input parameters must " \
                                               "match model parameters!"

        return self.model.logprior(*t0)


    def loglikelihood(self,t0, neg=False):
        """
        The log-likelihood for the model defined in self.model
        and the parameters in t0. Uses an exponential model for
        the errors.

        Parameters:
        ------------
        t0: {list | numpy.ndarray}
            The list with parameters for the model

        Returns:
        --------
        logl: float
            The logarithm of the likelihood function for the model and
            parameters given.

        """
        assert np.size(t0) == self.model.npar, "Input parameters must" \
                                               " match model parameters!"

        funcval = self.model(self.x, *t0)

        if self.m == 1:
            res = -np.sum(np.log(funcval)) - np.sum(self.y/funcval)
        else:
            res = -2.0*self.m*(np.sum(np.log(funcval)) +
                               np.sum(self.y/funcval) +
                               np.sum((2.0/float(2.*self.m) - 1.0)*
                                      np.log(self.y)))

        if np.isnan(res):
            #print("res is nan")
            res = logmin
        elif res == np.inf or np.isfinite(res) == False:
            #print("res is infinite!")
            res = logmin

        if neg:
            return -res
        else:
            return res

