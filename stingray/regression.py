
try:
    import matplotlib.pyplot as plt
except ImportError:
    can_plot = False


#### GENERAL IMPORTS ###
import numpy as np
import scipy
import scipy.optimize
import scipy.stats
import scipy.signal
import copy

from .parametricmodels import FixedCentroidLorentzian, CombinedModel

from . import PSDPosterior

try:
    from statsmodels.tools.numdiff import approx_hess
    comp_hessian = True
except ImportError:
    comp_hessian = False


### own imports
from . import Lightcurve
from . import Powerspectrum
from . import PSDPosterior, LightcurvePosterior, GaussianPosterior


class RegressionResults(object):


    def __init__(self, lpost, res, neg=True):
        """
        Helper class that will contain the results of the regression.
        Less fiddly than a dictionary.

        Parameters
        ----------
        lpost: instance of Posterior or one of its subclasses
            The object containing the function that is being optimized
            in the regression


        res: instance of scipy's OptimizeResult class
            The object containing the results from a optimization run

        """
        self.neg = neg
        self.result = res.fun
        self.p_opt = res.opt
        self.model = lpost.model


        self._compute_covariance(lpost, res)
        self._compute_model(lpost, res)
        self._compute_criteria(lpost, res)
        self._compute_statistics(lpost, res)

    def _compute_covariance(self, lpost, res):

        if hasattr(res, "hess_inv"):
            self.cov = res.hess_inv
            self.err = np.sqrt(np.diag(res.hess_inv))
        else:
            ### calculate Hessian approximating with finite differences
            print("Approximating Hessian with finite differences ...")
            phess = approx_hess(self.p_opt, lpost, neg=self.neg)

            self.cov = np.linalg.inv(phess)
            self.err = np.sqrt(np.diag(self.cov))

    def _compute_model(self, lpost, res):
        self.mfit = lpost.model(lpost.x, *self.p_opt)


    def _compute_criteria(self, lpost, res):

        self.deviance = -2.0*lpost.loglikelihood(self.p_opt, neg=False)

        ## Akaike Information Criterion
        self.aic = self.result+2.0*self.p_opt.shape[0]

        ### Bayesian Information Criterion
        self.bic = self.result + self.p_opt.shape[0]*np.log(lpost.x.shape[0])

        ### Deviance Information Criterion
        ## TODO: Add Deviance Information Criterion

    def _compute_statistics(self, lpost, res):
        try:
            self.mfit
        except AttributeError:
            self._compute_model(lpost, res)

        self.merit = np.sum(((lpost.y-self.mfit)/self.mfit)**2.0)
        self.dof = lpost.y.shape[0] - float(self.p_opt.shape[0])
        self.sexp = 2.0*len(lpost.x)*len(self.p_opt)
        self.ssd = np.sqrt(2.0*self.sexp)
        self.sobs = np.sum(lpost.y-self.mfit)

    def print_summary(self, lpost, res):

        print("The best-fit model parameters plus errors are:")
        for i,(x,y, p) in enumerate(zip(self.popt, self.err,
                                        lpost.model.parnames)):
            print("%i) Parameter %s: %.5f +/i %.f5"%(i, p, x, y))

        print("\n")

        print("Fitting statistics: ")
        print(" -- number of data points: " + str(len(lpost.x)))

        try:
            self.deviance
        except AttributeError:
            self._compute_criteria(lpost, res)

        print(" -- Deviance [-2 log L] D = " + str(self.deviance))
        print(" -- The Akaike Information Criterion of the model is: " +
              str(self.aic) + ".")

        print(" -- The Bayesian Information Criterion of the model is: " +
              str(self.bic) + ".")

        try:
            self.merit
        except AttributeError:
            self._compute_statistics(lpost, res)

        print(" -- The figure-of-merit function for this model is: " +
              str(self.merit) +
              " and the fit for " + str(self.dof) + " dof is " +
              str(self.merit/self.dof) + ".")

        print(" -- Summed Residuals S = " + str(self.sobs))
        print(" -- Expected S ~ " + str(self.sexp) + " +/- " + str(self.ssd))
        print(" -- merit function (SSE) M = " + str(self.merit) + "\n\n")

        return

class Regression(object):
    """ Maximum Likelihood Superclass. """
    def __init__(self, fitmethod='L-BFGS-B', max_post=True):
        """
        Linear regression of two-dimensional data.
        Note: optimization with bounds is not supported. If something like
        this is required, define (uniform) priors in the ParametricModel
        instances to be used below.

        Parameters:
        -----------
        fitmethod: string, optional, default "L-BFGS-B"
            Any of the strings allowed in scipy.optimize.minimize in
            the method keyword. Sets the fit method to be used.

        max_post: bool, optional, default True
            If True, then compute the Maximum-A-Posteriori estimate. If False,
            compute a Maximum Likelihood estimate.
        """

        self.fitmethod = fitmethod
        self.max_post = max_post


    def fit(self, lpost, t0, neg=True):
        """
        Do either a Maximum A Posteriori or Maximum Likelihood
        fit to the data.


        Parameters:
        -----------
        lpost: Posterior (or subclass) instance
            and instance of class Posterior or one of its subclasses
            that defines the function to be minized (either in loglikelihood
            or logposterior)

        t0 : {list | numpy.ndarray}
            List/array with set of initial parameters

        Returns:
        --------
        fitparams: dict
            A dictionary with the fit results
            TODO: Add description of keywords in the class!
        """

        if scipy.__version__ < "0.10.0":
            args = [neg]
        else:
            args = (neg,)


        ### different commands for different fitting methods,
        ### at least until scipy 0.11 is out

        funcval = 100.0
        i = 0
        while funcval == 100 or funcval == 200 or \
                funcval == 0.0 or not np.isfinite(funcval):

            if i > 20:
                raise Exception("Fitting unsuccessful!")
            ### perturb parameters slightly
            t0_p = np.random.multivariate_normal(t0, np.diag(np.abs(t0)/10.))

            ## if max_post is True, do the Maximum-A-Posteriori Fit
            if self.max_post:
                opt = scipy.optimize.minimize(lpost, t0_p,
                                              method=self.fitmethod,
                                              args=args)

            ## if max_post is False, then do a Maximum Likelihood Fit
            else:
                opt = scipy.optimize.minimize(lpost.loglikelihood, t0_p,
                                              method=self.fitmethod,
                                              args=args)



            funcval = opt.fun
            i+= 1


        res = RegressionResults(lpost, opt, neg=neg)

        return res


    def compute_lrt(self, lpost1, t1, lpost2, t2, neg=True):
        """
        This function computes the Likelihood Ratio Test between two
        nested models.

        Parameters
        ----------


        """

        ### fit data with both models
        res1 = self.fit(lpost1, t1, neg=neg)
        res2 = self.fit(lpost2, t2, neg=neg)

        ### compute log likelihood ratio as difference between the deviances
        lrt = res1.deviance - res2.deviance

        return lrt



class PSDRegression(Regression):

    ### ps = PowerSpectrum object with periodogram
    ### obs= if True, compute covariances and print summary to screen
    ###    

    def __init__(self, ps, fitmethod='L-BFGS-B', max_post=True):

        self.ps = ps
        Regression.__init__(self, fitmethod=fitmethod, max_post=max_post)


    def fit(self, model, t0, neg=True):

        self.lpost = PSDPosterior(self.ps, model, m=self.ps.m)
        res = Regression.fit(self, self.lpost, t0, neg=neg)

        res.maxpow, res.maxfreq, res.maxind = \
            self._compute_highest_outlier(self.lpost, res)

        return res


    def compute_lrt(self, model1, t1, model2, t2):

        lpost1 = PSDPosterior(self.ps, model1, m=self.ps.m)
        lpost2 = PSDPosterior(self.ps, model2, m=self.ps.m)

        lrt = Regression.compute_lrt(self, lpost1, t1, lpost2, t2)

        return lrt


    def _compute_highest_outlier(self, lpost, res, nmax=1):

        residuals = 2.0*lpost.y[:1]/res.mfit

#        if nmax > 1:

        ratio_sort = copy.copy(residuals)
        ratio_sort.sort()
        max_y = ratio_sort[-nmax:]

        max_x= np.zeros(max_y.shape[0])
        max_ind = np.zeros(max_y.shape[0])

        for i,my in enumerate(max_y):
            max_x[i], max_ind[i] = self._find_outlier(lpost.x, residuals, my)

#        else:
#            max_y = np.max(ratio)
#            max_x, max_ind = self._find_outlier(xdata, ratio, max_y)

        return max_y, max_x, max_ind

    def _find_outlier(self, xdata, ratio, max_y):
        max_ind = np.where(ratio == max_y)[0]+1
        #if np.size(max_ind) == 0:
        #    max_ind = None
        #    max_x = None
        #else:
        #if np.size(max_ind) > 1:
        #    max_ind = max_ind[0]
        max_x = xdata[max_ind]

        return max_x, max_ind


    ### plot two fits against each other
    def plotfits(self, res1, res2 = None, namestr='test', log=False):

        if not can_plot:
            print("No matplotlib imported. Can't plot!")
        else:
            ### make a figure
            f = plt.figure(figsize=(12,10))
            ### adjust subplots such that the space between the top and bottom of each are zero
            plt.subplots_adjust(hspace=0.0, wspace=0.4)


            ### first subplot of the grid, twice as high as the other two
            ### This is the periodogram with the two fitted models overplotted
            s1 = plt.subplot2grid((4,1),(0,0),rowspan=2)

            if log:
                logx = np.log10(self.ps.freq[1:])
                logy = np.log10(self.ps.ps[1:])
                logpar1 = np.log10(res1.mfit)

                p1, = s1.plot(logx, logy, color='black', linestyle='steps-mid')
                p2, = s1.plot(logx, logpar1, color='blue', lw=2)
                s1.set_xlim([min(logx), max(logx)])
                s1.set_ylim([min(logy)-1.0, max(logy)+1])
                if self.ps.norm == "leahy":
                    s1.set_ylabel('log(Leahy-Normalized Power)', fontsize=18)
                elif self.ps.norm == "rms":
                    s1.set_ylabel('log(RMS-Normalized Power)', fontsize=18)
                else:
                    s1.set_ylabel("log(Power)", fontsize=18)

            else:
                p1, = s1.plot(self.ps.freq[1:], self.ps.ps[1:],
                              color='black', linestyle='steps-mid')
                p2, = s1.plot(self.ps.freq[1:], res1.mfit,
                              color='blue', lw=2)

                s1.set_xscale("log")
                s1.set_yscale("log")

                s1.set_xlim([min(self.ps.freq[1:]), max(self.ps.ps[1:])])
                s1.set_ylim([min(self.ps.freq[1:])/10.0,
                             max(self.ps.ps[1:])*10.0])

                if self.ps.norm == "leahy":
                    s1.set_ylabel('Leahy-Normalized Power', fontsize=18)
                elif self.ps.norm == "rms":
                    s1.set_ylabel('RMS-Normalized Power', fontsize=18)
                else:
                    s1.set_ylabel("Power", fontsize=18)

            if res2 is not None:
                if log:
                    logpar2 = np.log10(res2.mfit)
                    p3, = s1.plot(logx, logpar2, color='red', lw=2)
                else:
                    p3, = s1.plot(self.ps.freq[1:], res2.mfit,
                                  color='red', lw=2)
                s1.legend([p1, p2, p3], ["data", "model 1 fit", "model 2 fit"])
            else:
                s1.legend([p1, p2], ["data", "model fit"])

            s1.set_title("Periodogram and fits for data set " + namestr,
                         fontsize=18)

            ### second subplot: power/model for Power law and straight line
            s2 = plt.subplot2grid((4,1),(2,0),rowspan=1)
            pldif = self.ps.ps[1:]/res1.mfit

            s2.set_ylabel("Residuals, \n" + res1.model.name + " model",
                          fontsize=18)

            if log:
                s2.plot(logx, pldif, color='black', linestyle='steps-mid')
                s2.plot(logx, np.ones(self.ps.freq[1:].shape[0]),
                        color='blue', lw=2)
                s2.set_xlim([min(logx), max(logx)])
                s2.set_ylim([min(pldif), max(pldif)])

            else:
                s2.plot(self.ps.freq[1:], pldif, color='black', linestyle='steps-mid')
                s2.plot(self.ps.ps[1:], np.ones(self.x.shape[0]), color='blue', lw=2)

                s2.set_xscale("log")
                s2.set_yscale("log")
                s2.set_xlim([min(self.ps.freq[1:]), max(self.ps.freq[1:])])
                s2.set_ylim([min(pldif), max(pldif)])

            if res2 is not None:
                bpldif = self.ps.ps[1:]/res2.mfit

            ### third subplot: power/model for bent power law and straight line
                s3 = plt.subplot2grid((4,1),(3,0),rowspan=1)

                if log:
                    s3.plot(logx, bpldif, color='black', linestyle='steps-mid')
                    s3.plot(logx, np.ones(len(self.ps.freq[1:])),
                            color='red', lw=2)
                    s3.axis([min(logx), max(logx), min(bpldif), max(bpldif)])
                    s3.set_xlabel("log(Frequency) [Hz]", fontsize=18)

                else:
                    s3.plot(self.ps.freq[1:], bpldif,
                            color='black', linestyle='steps-mid')
                    s3.plot(self.ps.freq[1:], np.ones(len(self.ps.freq[1:])),
                            color='red', lw=2)
                    s3.set_xscale("log")
                    s3.set_yscale("log")
                    s3.set_xlim([min(self.ps.freq[1:]), max(self.ps.freq[1:])])
                    s3.set_ylim([min(bpldif), max(bpldif)])
                    s3.set_xlabel("Frequency [Hz]", fontsize=18)

                s3.set_ylabel("Residuals, \n" + res.model.name + " model",
                              fontsize=18)

            else:
                if log:
                    s2.set_xlabel("log(Frequency) [Hz]", fontsize=18)
                else:
                    s2.set_xlabel("Frequency [Hz]", fontsize=18)

            ax = plt.gca()

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(14)

            ### make sure xticks are taken from first plots, but don't appear there
            plt.setp(s1.get_xticklabels(), visible=False)

            ### save figure in png file and close plot device
            plt.savefig(namestr + '_ps_fit.png', format='png')
            plt.close()

        return

