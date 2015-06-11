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

    def assign_modalities(self, log2_bayes_factors, reset_index=False):
        """Guess the most likely modality for each event

        For each event that has at least one non-NA value, if no modalilites
        have logsumexp'd logliks greater than the log Bayes factor threshold,
        then they are assigned the 'ambiguous' modality, because we cannot
        reject the null hypothesis that these did not come from the uniform
        distribution.

        Parameters
        ----------
        log2_bayes_factors : pandas.DataFrame
            A (4, n_events) dataframe with bayes factors for the Psi~1, Psi~0,
            bimodal, and middle modalities. If an event has no bayes factors
            for any of those modalities, it is ignored
        reset_index : bool
            If True, remove the first level of the index from the dataframe.
            Useful if you are using this function to apply to a grouped
            dataframe where the first level is something other than the
            modality, e.g. the celltype

        Returns
        -------
        modalities : pandas.Series
            A (n_events,) series with the most likely modality for each event

        """
        if reset_index:
            x = log2_bayes_factors.reset_index(level=0, drop=True)
        else:
            x = log2_bayes_factors
        not_na = (x.notnull() > 0).any()
        not_na_columns = not_na[not_na].index
        x.ix['ambiguous', not_na_columns] = self.logbf_thresh
        return x.idxmax()

    def _fit_transform_one_step(self, data, models):
        non_na = data.count() > 0
        non_na_columns = non_na[non_na].index
        data_non_na = data[non_na_columns]
        if data_non_na.empty:
            return pd.DataFrame()
        else:
            return data_non_na.apply(lambda x: pd.Series(
                {k: v.logsumexp_logliks(x)
                 for k, v in models.iteritems()}), axis=0)

    def fit_transform(self, data):
        """Get the modality assignments of each splicing event in the data

        Parameters
        ----------
        data : pandas.DataFrame
            A (n_samples, n_events) dataframe of splicing events' PSI scores.
            Must be psi scores which range from 0 to 1

        Returns
        -------
        log2_bayes_factors : pandas.DataFrame
            A (n_modalities, n_events) dataframe of the estimated log2
            bayes factor for each splicing event, for each modality

        Raises
        ------
        AssertionError
            If ``data`` does not fall only between 0 and 1.
        """
        assert np.all(data.values.flat[np.isfinite(data.values.flat)] <= 1)
        assert np.all(data.values.flat[np.isfinite(data.values.flat)] >= 0)

        # Estimate Psi~0/Psi~1 first (only one parameter change with each
        # paramterization)
        logbf_one_param = self._fit_transform_one_step(data,
                                                       self.one_param_models)

        # Take everything that was below the threshold for included/excluded
        # and estimate bimodal and middle (two parameters change in each
        # parameterization
        ind = (logbf_one_param < self.logbf_thresh).all()
        ambiguous_columns = ind[ind].index
        data2 = data.ix[:, ambiguous_columns]
        logbf_two_param = self._fit_transform_one_step(data2,
                                                       self.two_param_models)
        log2_bayes_factors = pd.concat([logbf_one_param, logbf_two_param],
                                       axis=0)

        # Make sure the returned dataframe has the same number of columns
        empty = data.count() == 0
        empty_columns = empty[empty].index
        empty_df = pd.DataFrame(np.nan, index=log2_bayes_factors.index,
                                columns=empty_columns)
        log2_bayes_factors = pd.concat([log2_bayes_factors, empty_df], axis=1)
        return log2_bayes_factors


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
