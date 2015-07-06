"""
Calculate modalities of splicing events.
"""

from collections import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.misc import logsumexp


MODALITIES_NAMES = ['Psi~0', 'middle', 'Psi~1', 'bimodal',
                    'ambiguous']


class ModalityModel(object):
    """Object to model modalities from beta distributions"""

    def __init__(self, alphas, betas):
        if not isinstance(alphas, Iterable) and not isinstance(betas,
                                                               Iterable):
            alphas = [alphas]
            betas = [betas]

        self.alphas = np.array(alphas) if isinstance(alphas, Iterable) \
            else np.ones(len(betas)) * alphas
        self.betas = np.array(betas) if isinstance(betas, Iterable) \
            else np.ones(len(alphas)) * betas

        self.rvs = [stats.beta(a, b) for a, b in
                    zip(self.alphas, self.betas)]
        self.scores = np.ones(self.alphas.shape).astype(float)
        self.prob_parameters = self.scores/self.scores.sum()

    def __eq__(self, other):
        return np.all(self.alphas == other.alphas) \
            and np.all(self.betas == other.betas) \
            and np.all(self.prob_parameters == other.prob_parameters)

    def __ne__(self, other):
        return not self.__eq__(other)

    def logliks(self, x):
        x = x.copy()
        x[x == 0] = 0.001
        x[x == 1] = 0.999

        return np.array([np.log(prob) + rv.logpdf(x[np.isfinite(x)]).sum()
                         for prob, rv in
                         zip(self.prob_parameters, self.rvs)])

    def logsumexp_logliks(self, x):
        return logsumexp(self.logliks(x))


class ModalityEstimator(object):
    """Use Bayesian methods to estimate modalities of splicing events"""

    # colors = dict(
    # zip(['excluded', 'middle', 'included', 'bimodal', 'uniform'],
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
        self.middle_model = ModalityModel(self.parameters+3, self.parameters+3)
        self.bimodal_model = ModalityModel(1 / (self.parameters+3),
                                           1 / (self.parameters+3))

        self.one_param_models = {'Psi~1': self.inclusion_model,
                                 'Psi~0': self.exclusion_model}
        self.two_param_models = {'bimodal': self.bimodal_model,
                                 'middle': self.middle_model}
        self.models = {'Psi~1': self.inclusion_model,
                       'Psi~0': self.exclusion_model,
                       'bimodal': self.bimodal_model,
                       'middle': self.middle_model}

    def _loglik(self, event):
        """Calculate log-likelihoods of an event, given the modality models"""
        return dict((name, m.logliks(event))
                    for name, m in self.models.iteritems())

    def _logsumexp(self, logliks):
        """Calculate logsumexps of each modality's loglikelihood"""
        logsumexps = pd.Series(dict((name, logsumexp(loglik))
                                    for name, loglik in logliks.iteritems()))
        logsumexps['ambiguous'] = self.logbf_thresh
        return logsumexps

    def _guess_modality(self, logsumexps):
        """Guess the most likely modality.

        If no modalilites have logsumexp'd logliks greater than the log Bayes
        factor threshold, then they are assigned the 'uniform' modality,
        which is the null hypothesis
        """

        if all(logsumexps[self.one_param_models.keys()] > self.logbf_thresh):
            return logsumexps[self.one_param_models.keys()].idxmax()
        else:
            other_models = logsumexps.index.difference(
                self.one_param_models.keys())
            return logsumexps[other_models].idxmax()

    def fit_transform(self, data):
        """Get the modality assignments of each splicing event in the data

        Parameters
        ----------
        data : pandas.DataFrame
            A (n_samples, n_events) dataframe of splicing events' PSI scores.
            Must be psi scores which range from 0 to 1

        Returns
        -------
        modality_assignments : pandas.Series
            A (n_events,) series of the estimated modality for each splicing
            event

        Raises
        ------
        AssertionError
            If ``data`` does not fall only between 0 and 1.
        """
        assert np.all(data.values.flat[np.isfinite(data.values.flat)] <= 1)
        assert np.all(data.values.flat[np.isfinite(data.values.flat)] >= 0)

        # If all values are NA, don't calculate anything
        if (data.count() > 0).sum() == 0:
            return pd.Series(np.nan, index=data.columns)

        # Estimate Psi~0/Psi~1 first
        non_na_columns = data.count() > 0
        logsumexp_logliks1 = data.ix[:, non_na_columns].apply(
            lambda x: pd.Series(
                {k: v.logsumexp_logliks(x)
                 for k, v in self.one_param_models.iteritems()}), axis=0)
        logsumexp_logliks1.ix['ambiguous'] = self.logbf_thresh
        # na_columns1 = data.count() == 0
        # logsumexp_logliks1[na_columns1] = np.nan
        modality_assignments1 = logsumexp_logliks1.idxmax()

        # Take everything that was ambiguous for included/excluded and estimate
        # bimodal and middle
        series = modality_assignments1 == 'ambiguous'
        ind = series[series].index
        data2 = data.ix[:, ind]
        non_na_columns = data2.count() > 0
        logsumexp_logliks2 = data2.ix[:, non_na_columns].apply(
            lambda x: pd.Series(
                {k: v.logsumexp_logliks(x)
                 for k, v in self.two_param_models.iteritems()}), axis=0)
        logsumexp_logliks2.ix['ambiguous'] = self.logbf_thresh
        # na_columns2 = data.count() == 0
        # logsumexp_logliks2[na_columns2] = np.nan
        modality_assignments2 = logsumexp_logliks2.idxmax()

        # Combine the results
        modality_assignments = modality_assignments1
        modality_assignments[modality_assignments2.index] = \
            modality_assignments2.values

        # Add back the NA columns
        modality_assignments = modality_assignments.reindex(data.columns)
        return modality_assignments


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
