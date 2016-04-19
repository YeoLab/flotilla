from itertools import chain
from collections import Iterable
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseData
from ..util import timestamp
from ..visualize.splicing import lavalamp, hist_single_vs_pooled_diff, \
    lavalamp_pooled_inconsistent

FRACTION_DIFF_THRESH = 0.1


class SplicingData(BaseData):
    binned_reducer = None
    raw_reducer = None

    n_components = 2
    _binsize = 0.1

    included_label = '~1'
    excluded_label = '~0'

    def __init__(self, data,
                 feature_data=None, binsize=0.1, outliers=None,
                 feature_rename_col=None,
                 feature_ignore_subset_cols=None,
                 excluded_max=0.2, included_min=0.8,
                 pooled=None, predictor_config_manager=None,
                 technical_outliers=None, minimum_samples=0,
                 feature_expression_id_col=None):
        """Instantiate a object for percent spliced in (PSI) scores

        Parameters
        ----------
        data : pandas.DataFrame
            A [n_events, n_samples] dataframe of data events
        n_components : int
            Number of components to use in the reducer
        binsize : float
            Value between 0 and 1, the bin size for binning the study_data
            scores
        excluded_max : float
            Maximum value for the "excluded" bin of psi scores. Default 0.2.
        included_max : float
            Minimum value for the "included" bin of psi scores. Default 0.8.

        Notes
        -----
        'thresh' from BaseData is not used.
        """
        sys.stdout.write("{}\tInitializing splicing\n".format(timestamp()))

        super(SplicingData, self).__init__(
            data, feature_data=feature_data,
            feature_rename_col=feature_rename_col,
            feature_ignore_subset_cols=feature_ignore_subset_cols,
            outliers=outliers, pooled=pooled,
            technical_outliers=technical_outliers,
            predictor_config_manager=predictor_config_manager,
            minimum_samples=minimum_samples, data_type='splicing')
        sys.stdout.write(
            "{}\tDone initializing splicing\n".format(timestamp()))

        self.feature_expression_id_col = feature_expression_id_col \
            if feature_expression_id_col is not None \
            else self.feature_rename_col

        self.binsize = binsize
        self.excluded_max = excluded_max
        self.included_min = included_min

        self.bins = np.arange(0, 1 + self.binsize, self.binsize)

    def plot_feature(self, feature_id, sample_ids=None,
                     phenotype_groupby=None,
                     phenotype_order=None, color=None,
                     phenotype_to_color=None,
                     phenotype_to_marker=None,
                     violinplot_kws=None, col_wrap=4):
        violinplot_kws = {} if violinplot_kws is None else violinplot_kws
        violinplot_kws.setdefault('bw', 0.2)
        violinplot_kws.setdefault('ylim', (0, 1))
        violinplot_kws.setdefault('yticks', (0, 0.5, 1))
        violinplot_kws.setdefault('ylabel', '$\Psi$')
        super(SplicingData, self).plot_feature(feature_id, sample_ids,
                                               phenotype_groupby,
                                               phenotype_order, color,
                                               phenotype_to_color,
                                               phenotype_to_marker,
                                               col_wrap=col_wrap,
                                               violinplot_kws=violinplot_kws)

    def pooled_inconsistent(self, data, feature_ids=None,
                            fraction_diff_thresh=FRACTION_DIFF_THRESH):
        """Return splicing events which pooled samples are consistently
        different from the single cells.

        Parameters
        ----------
        singles_ids : list-like
            List of sample ids of single cells (in the main ".data" DataFrame)
        pooled_ids : list-like
            List of sample ids of pooled cells (in the other ".pooled"
            DataFrame)
        feature_ids : None or list-like
            List of feature ids. If None, use all
        fraction_diff_thresh : float


        Returns
        -------
        large_diff : pandas.DataFrame
            All splicing events which have a scaled difference larger than
            the fraction diff thresh
        """
        # singles = self._subset(self.data, singles_ids, feature_ids)
        singles, pooled, not_measured_in_pooled, diff_from_singles = \
            self._diff_from_singles(data, feature_ids, scaled=True)

        try:
            ind = diff_from_singles.abs() >= fraction_diff_thresh
            large_diff = diff_from_singles[ind].dropna(axis=1, how='all')
        except AttributeError:
            large_diff = None
        return singles, pooled, not_measured_in_pooled, large_diff

    # @memoize
    def _diff_from_singles(self, data,
                           feature_ids=None, scaled=True, dropna=True):
        """Calculate the difference between pooled and singles' psis

        Parameters
        ----------
        data : pandas.DataFrame
            A (n_samples, n_features) DataFrame
        feature_ids : list-like
            Subset of the features you want
        scaled : bool
            If True, then take the average difference between each pooled
            sample and all singles. If False, then get the summed difference
        dropna : bool
            If True, remove events which were not measured in the pooled
            samples

        Returns
        -------
        singles : pandas.DataFrame
            Subset of the data that's only the single-cell samples
        pooled : pandas.DataFrame
            Subset of the data that's only the pooled samples
        not_measured_in_pooled : list-like
            List of features not measured in the pooled samples
        diff_from_singles : pandas.DataFrame
            A (n_pooled, n_features) Dataframe of the summed (or scaled if
            scaled=True)
        """
        singles, pooled = self._subset_singles_and_pooled(
            feature_ids, data=data, require_min_samples=False)
        if pooled is None:
            not_measured_in_pooled = None
            diff_from_singles = None
            return singles, pooled, not_measured_in_pooled, diff_from_singles

        # Make sure "pooled" is always a dataframe
        if isinstance(pooled, pd.Series):
            pooled = pd.DataFrame([pooled.values], columns=pooled.index,
                                  index=[pooled.name])
        pooled = pooled.dropna(how='all', axis=1)
        not_measured_in_pooled = singles.columns.diff(pooled.columns)
        singles, pooled = singles.align(pooled, axis=1, join='inner')
        # import pdb; pdb.set_trace()

        diff_from_singles = pooled.apply(
            lambda x: (singles - x.values).abs().sum(), axis=1)

        if scaled:
            diff_from_singles = \
                diff_from_singles / singles.count().astype(float)
        if dropna:
            diff_from_singles = diff_from_singles.dropna(axis=1, how='all')
        return singles, pooled, not_measured_in_pooled, diff_from_singles

    def plot_lavalamp(self, phenotype_to_color, sample_ids=None,
                      feature_ids=None,
                      data=None, groupby=None, order=None):
        if data is None:
            data = self._subset(self.data, sample_ids, feature_ids,
                                require_min_samples=False)
        else:
            if feature_ids is not None and sample_ids is not None:
                raise ValueError('Can only specify `sample_ids` and '
                                 '`feature_ids` or `data`, but not both.')

        if groupby is None:
            groupby = pd.Series('all', index=self.singles.index)
        grouped = data.groupby(groupby)

        nrows = len(grouped.groups)
        figsize = 12, nrows * 4
        fig, axes = plt.subplots(nrows=len(grouped.groups), figsize=figsize,
                                 sharex=False)
        if not isinstance(axes, Iterable):
            axes = [axes]

        if order is None:
            order = grouped.groups.keys()

        for ax, name in zip(axes, order):
            try:
                color = phenotype_to_color[name]
            except KeyError:
                color = None
            samples = grouped.groups[name]
            psi = data.ix[samples]
            lavalamp(psi, color=color, ax=ax)
            ax.set_title(name)
        sns.despine()
        fig.tight_layout()

    def plot_lavalamp_pooled_inconsistent(
            self, data, feature_ids=None,
            fraction_diff_thresh=FRACTION_DIFF_THRESH, color=None):
        """

        Parameters
        ----------


        Returns
        -------


        Raises
        ------

        """
        singles, pooled, not_measured_in_pooled, pooled_inconsistent = \
            self.pooled_inconsistent(data, feature_ids,
                                     fraction_diff_thresh)
        percent = self._divide_inconsistent_and_pooled(pooled,
                                                       pooled_inconsistent)
        lavalamp_pooled_inconsistent(singles, pooled, pooled_inconsistent,
                                     color=color, percent=percent)

    def plot_hist_single_vs_pooled_diff(self, data, feature_ids=None,
                                        color=None, title='',
                                        hist_kws=None):
        """Plot histogram of distances between singles and pooled"""
        singles, pooled, not_measured_in_pooled, diff_from_singles = \
            self._diff_from_singles(data, feature_ids)
        singles, pooled, not_measured_in_pooled, diff_from_singles_scaled = \
            self._diff_from_singles(data, feature_ids, scaled=True)
        hist_single_vs_pooled_diff(diff_from_singles,
                                   diff_from_singles_scaled, color=color,
                                   title=title, hist_kws=hist_kws)

    @staticmethod
    def _divide_inconsistent_and_pooled(pooled, pooled_inconsistent):
        """The percent of events with pooled psi different from singles"""
        if pooled_inconsistent is None:
            return np.nan
        if pooled_inconsistent.shape[1] == 0:
            return 0.0
        try:
            return pooled_inconsistent.shape[1] / float(pooled.shape[1]) * 100
        except ZeroDivisionError:
            return 100.0

    def _calculate_linkage(self, sample_ids, feature_ids,
                           metric='euclidean', linkage_method='median',
                           bins=None, standardize=False):
        if bins is not None:
            data = self.binify(bins)
        else:
            data = self.data
        return super(SplicingData, self)._calculate_linkage(
            data, sample_ids=sample_ids, feature_ids=feature_ids,
            standardize=standardize, metric=metric,
            linkage_method=linkage_method)

    def _subset_and_standardize(self, data, sample_ids=None,
                                feature_ids=None,
                                standardize=True, return_means=False,
                                rename=False):
        """Grab a subset of the provided data and standardize/remove NAs

        Take only the sample ids and feature ids from this data, require
        at least some minimum samples. Standardization is performed by
        replacing ``NA``s with the value 0.5. Then, all values for
        that event are transformed with :math:`\arccos`/:math:`\cos^{-1}`/arc
        cosine so that all values range from :math:`-\pi` to :math:`+\pi` and
        are centered around :math:`0`. As much of single-cell alternative
        splicing data is near-0 or near-1, this spreads out the values near 0
        and 1, and squishes the values near 0.5.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe to subset
        sample_ids : list-like, optional (default=None)
            If None, all sample ids will be used, else only the sample ids
            specified
        feature_ids : list-like, optional (default=None)
            If None, all features will be used, else only the features
            specified
        standardize : bool, optional (default=True)
            If True, replaced NAs with 0.5 and perform an arccosine transform
            to 0-center the splicing data.
        return_means : bool, optional (default=False)
            If True, return a tuple of (subset, means), otherwise just return
            the subset
        rename : bool, optional (default=False)
            Whether or not to rename the feature ids using ``feature_renamer``

        Returns
        -------
        subset : pandas.DataFrame
            Subset of the dataframe with the requested samples and features,
            and standardized as described
        means : pandas.DataFrame
            (Only if return_means=True) Mean values of the features (columns).

        """
        subset = self._subset(self.data, sample_ids, feature_ids)
        subset = subset.dropna(how='all', axis=1).dropna(how='all', axis=0)

        # This is splicing data ranging from 0 to 1, so fill na with 0.5
        # and perform an arc-cosine transform to make the data range from
        # -pi to pi
        if standardize:
            subset = subset.fillna(subset.mean())
            subset = -2 * np.arccos(subset * 2 - 1) + np.pi
        means = subset.mean()

        if rename:
            means = means.rename_axis(self.feature_renamer)
            subset = subset.rename_axis(self.feature_renamer, 1)

        if return_means:
            return subset, means
        else:
            return subset

    def plot_two_features(self, feature1, feature2, groupby=None,
                          label_to_color=None, fillna=None, **kwargs):
        xlim = kwargs.pop('xlim', (0, 1))
        ylim = kwargs.pop('ylim', (0, 1))
        return super(SplicingData, self).plot_two_features(
            feature1, feature2, groupby=groupby, label_to_color=label_to_color,
            xlim=xlim, ylim=ylim, **kwargs)

    def plot_two_samples(self, sample1, sample2, fillna=None, **kwargs):
        xlim = kwargs.pop('xlim', (0, 1))
        ylim = kwargs.pop('ylim', (0, 1))
        return super(SplicingData, self).plot_two_samples(
            sample1, sample2, xlim=xlim, ylim=ylim, **kwargs)

    def plot_clustermap(self, sample_ids=None, feature_ids=None, data=None,
                        feature_colors=None, sample_id_to_color=None,
                        metric='euclidean', method='average',
                        scale_fig_by_data=True, **kwargs):
        kwargs.setdefault('cmap', 'RdYlBu_r')
        kwargs.setdefault('center', 0)
        return super(SplicingData, self).plot_clustermap(
            sample_ids=sample_ids, feature_ids=feature_ids, data=data,
            feature_colors=feature_colors,
            sample_id_to_color=sample_id_to_color,
            metric=metric, method=method, scale_fig_by_data=scale_fig_by_data,
            norm_features=False, **kwargs)

    def splicing_to_expression_id(self, feature_ids):
        """Get the gene ids corresponding to the splicing ids provided"""
        return list(chain(*self.feature_data[self.feature_expression_id_col][
            feature_ids].str.split(',').dropna().values))

    def expression_to_splicing_id(self, expression_ids):
        ind = self.feature_data.ensembl_id.map(
            lambda x: expression_ids.isin(
                x.split(',')).any() if isinstance(x, str) else False)

        return self.feature_data.index[ind]
