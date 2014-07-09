import collections

import numpy as np
import pandas as pd
from sklearn import cross_validation

from ..util import memoize
from .infotheory import jsd, binify


class Modalities(object):
    """

    """
    modalities_bins = np.array([[1, 0, 0],  # excluded
                                [0, 1, 0],  # middle
                                [0, 0, 1],  # included
                                [1, 0, 1],  # bimodal
                                [1, 1, 1]])  # uniform

    modalities_names = ['excluded', 'middle', 'included', 'bimodal', 'uniform']

    true_modalities = pd.DataFrame(modalities_bins.T, columns=modalities_names)

    def __init__(self, excluded_max=0.2, included_min=0.8):
        self.bins = (0, excluded_max, included_min, 1)

    # def __call__(self, *args, **kwargs):
    #     return self.assignments

    def _col_jsd_modalities(self, col):
        return self.true_modalities.apply(lambda x: jsd(x, col), axis=0)

    def sqrt_jsd_modalities(self, binned):
        """Return the square root of the JSD of each splicing event versus
            all the modalities. Use square root of JSD because it's a metric.
        """
        return np.sqrt(binned.apply(self._col_jsd_modalities, axis=0))

    def assignments(self, sqrt_jsd_modalities):
        modalities = self.true_modalities.columns[
            np.argmin(sqrt_jsd_modalities.values, axis=0)]
        return pd.Series(modalities, sqrt_jsd_modalities.columns)

    @memoize
    def fit_transform(self, data, do_not_memoize=False):
        """Given psi scores, estimate the modality of each

        Parameters
        ----------
        data : pandas.DataFrame
            A samples x features dataframe, where you want to find the
            splicing modality of each column (feature)
        do_not_memoize : bool
            Whether or not to memoize the results of the fit_transform on this
            data

        Returns
        -------
        assignments : pandas.Series
            Modality assignments of each column (feature)
        """
        binned = binify(data, self.bins)
        self.true_modalities.index = binned.index
        return self.assignments(self.sqrt_jsd_modalities(binned))

    def bootstrapped_fit_transform(self, data, n_iter=100, thresh=0.6,
                                   min_samples=10):
        """Resample each splicing event n_iter times to robustly estimate
        modalities.
        """
        bs = cross_validation.Bootstrap(data.shape[0], n_iter=n_iter)

        assignments = pd.DataFrame(columns=data.columns,
                                   index=range(n_iter))

        for i, (train_index, test_index) in enumerate(bs):
            # intex = train_index + test_index
            psi = data.ix[data.index[train_index], :]
            psi = psi.dropna(axis=1, thresh=min_samples)
            assignments.ix[i] = self.fit_transform(psi, do_not_memoize=True)

        counts = assignments.apply(lambda x: pd.Series(
            collections.Counter(x.dropna())) if x.isnull().sum() < x.shape[
            0] else np.nan)
        fractions = counts / counts.sum().astype(float)
        thresh_assignments = fractions[fractions >= thresh].apply(
            self._max_assignment, axis=0)
        thresh_assignments = thresh_assignments.fillna('unassigned')
        return thresh_assignments

    @staticmethod
    def _max_assignment(x):
        """Given a pandas.Series of modalities counts, return the maximum
        value. Necessary because just np.argmax will use np.nan as the max :(
        """
        if np.isfinite(x).sum() == 0:
            return np.nan
        else:
            return np.argmax(x)

    def counts(self, psi):
        assignments = self.fit_transform(psi)
        return assignments.groupby(assignments).size()


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
    float
        The "switchy score" of the study_data which can then be compared to other
        splicing event study_data

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
    numpy.array
        A 1-D array of the ordered indices, in switchy score order
    """
    switchy_scores = np.apply_along_axis(switchy_score, axis=0, arr=x)
    return np.argsort(switchy_scores)
