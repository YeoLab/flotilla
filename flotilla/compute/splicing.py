"""
Calculate modalities of splicing events.

This code is crazy, sometimes using classes and sometimes just objects because
for parallelization, you can't pickle anything that has state, like an
instancemethod
"""

from collections import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.misc import logsumexp


MODALITIES_NAMES = ['excluded', 'middle', 'included', 'bimodal',
                    'uniform']



class ModalityModel(object):
    """Object to model modalities from beta distributions"""
    def __init__(self, alphas, betas):
        if not isinstance(alphas, Iterable) and not isinstance(betas, Iterable):
            alphas = [alphas]
            betas = [betas]

        self.alphas = alphas if isinstance(alphas, Iterable) else np.ones(
            len(betas)) * alphas
        self.betas = betas if isinstance(betas, Iterable) else np.ones(
            len(alphas)) * betas

        self.rvs = [stats.beta(a, b) for a, b in
                    zip(self.alphas, self.betas)]
        self.scores = np.arange(len(self.rvs)).astype(float) + .1
        self.scaled_scores = self.scores / self.scores.max()
        self.prob_parameters = self.scaled_scores / self.scaled_scores.sum()

    def logliks(self, x):
        x = x.copy()
        x[x == 0] = 0.001
        x[x == 1] = 0.999

        return np.array([np.log(prob) + rv.logpdf(x).sum() for prob, rv in
                         zip(self.prob_parameters, self.rvs)])

    def logsumexp_logliks(self, x):
        return logsumexp(self.logliks(x))
    
class ModalityEstimator(object):
    """Use Bayesian methods to estimate modalities of splicing events"""

    # colors = dict(
    #     zip(['excluded', 'middle', 'included', 'bimodal', 'uniform'],
    #         sns.color_palette('deep', n_colors=5)))
    
    def __init__(self, step, vmax, logbf_thresh=3):
        """Initialize an object with models to estimate splicing modality

        Parameters
        ----------
        step : float
            Distance between parameter values
        vmax : float
            Maximum parameter value
        logbf_thresh : float
            Minimum threshold at which the bayes factor difference is defined
            to be significant
        """
        self.step = step
        self.vmax = vmax
        self.logbf_thresh = logbf_thresh

        self.parameters = np.arange(2, self.vmax + self.step,
                                    self.step).astype(float)
        self.exclusion_model = ModalityModel(1, self.parameters)
        self.inclusion_model = ModalityModel(self.parameters, 1)
        self.middle_model = ModalityModel(self.parameters, self.parameters)
        self.bimodal_model = ModalityModel(1 / self.parameters, 1 / self.parameters)
        
        self.models = {'included': self.inclusion_model,
                  'excluded': self.exclusion_model,
                  'bimodal': self.bimodal_model,
                  'middle': self.middle_model}

    def _loglik(self, event):
        return dict((name, m.logliks(event))
                    for name, m in self.models.iteritems())

    def _logsumexp(self, logliks):
        return pd.Series(dict((name, logsumexp(loglik))
                              for name, loglik in logliks.iteritems()))

    def _guess_modality(self, logsumexps):
        logsumexps['uniform'] = self.logbf_thresh
        return logsumexps.idxmax()

    def fit_transform(self, data):
        logsumexp_logliks = data.apply(lambda x:
                                       pd.Series({k: v._logsumexp(v.loglik_(x))
                                       for k, v in self.models.iteritems()}), axis=0)
        logsumexp_logliks.ix['uniform'] = self.logbf_thresh
        return logsumexp_logliks.idxmax()


def switchy_score(array):
    """Transform a 1D array of data scores to a vector of "switchy scores"

    Calculates std deviation and mean of sine- and cosine-transformed
    versions of the array. Better than sorting by just the mean which doesn't
    push the really lowly variant events to the ends.

    Parameters
    ----------
    array : numpy.array
        A 1-D numpy array or something that could be cast as such (like a list)

    Returns
    -------
    switchy_score : float
        The "switchy score" of the study_data which can then be compared to
        other splicing event study_data

    """
    array = np.array(array)
    variance = 1 - np.std(np.sin(array[~np.isnan(array)] * np.pi))
    mean_value = -np.mean(np.cos(array[~np.isnan(array)] * np.pi))
    return variance * mean_value


def get_switchy_score_order(x):
    """Apply switchy scores to a 2D array of data scores

    Parameters
    ----------
    x : numpy.array
        A 2-D numpy array in the shape [n_events, n_samples]

    Returns
    -------
    score_order : numpy.array
        A 1-D array of the ordered indices, in switchy score order
    """
    switchy_scores = np.apply_along_axis(switchy_score, axis=0, arr=x)
    return np.argsort(switchy_scores)
