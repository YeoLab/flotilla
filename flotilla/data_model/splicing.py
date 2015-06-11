from collections import Iterable
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseData
from ..compute.splicing import ModalityEstimator
from ..visualize.splicing import ModalitiesViz
from ..util import memoize, timestamp
from ..visualize.splicing import lavalamp, hist_single_vs_pooled_diff, \
    lavalamp_pooled_inconsistent

FRACTION_DIFF_THRESH = 0.1


class SplicingData(BaseData):
    binned_reducer = None
    raw_reducer = None

    n_components = 2
    _binsize = 0.1

    included_label = '$\Psi~1$ >>'
    excluded_label = '$\Psi~0$ >>'

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

        self.modality_estimator = ModalityEstimator(step=1., vmax=10.)
        # self.modalities_calculator = Modalities(excluded_max=excluded_max,
        # included_min=included_min)
        self.modality_visualizer = ModalitiesViz()

    @memoize
    def modality_log2bf(self, sample_ids=None, feature_ids=None, data=None,
                        groupby=None, min_samples=20):
        """Get log2 bayes factor of how well events fit to each modality

        Scores are in units of log2 bayes factors

        Parameters
        ----------
        sample_ids : list of str, optional
            Which samples to use. If None, use all. Default None.
        feature_ids : list of str, optional
            Which features to use. If None, use all. Default None.
        data : pandas.DataFrame, optional
            If provided, use this dataframe instead of the sample_ids and
            feature_ids provided
        min_samples : int, optional
            Minimum number of samples to use per grouped celltype. Default 20

        Returns
        -------
        modality_assignments : pandas.Series
            The modality assignments of each feature given these samples
        """
        if data is None:
            data = self._subset(self.singles, sample_ids, feature_ids,
                                require_min_samples=False)
        else:
            if feature_ids is not None and sample_ids is not None:
                raise ValueError('Can only specify `sample_ids` and '
                                 '`feature_ids` or `data`, but not both.')
        if groupby is None:
            groupby = pd.Series('all', index=data.index)

        grouped = data.groupby(groupby)
        data = pd.concat([df.dropna(thresh=min_samples, axis=1)
                          for name, df in grouped])
        scores = data.groupby(groupby).apply(
            self.modality_estimator.fit_transform)
        return scores

    @memoize
    def modality_assignments(self, sample_ids=None, feature_ids=None,
                             data=None, groupby=None, min_samples=20):
        """Assign a modality to each splicing event in each group

        Parameters
        ----------
        sample_ids : list of str, optional
            Which samples to use. If None, use all. Default None.
        feature_ids : list of str, optional
            Which features to use. If None, use all. Default None.
        data : pandas.DataFrame, optional
            If provided, use this dataframe instead of the sample_ids and
            feature_ids provided
        min_samples : int, optional
            Minimum number of samples to use per grouped celltype. Default 10

        Returns
        -------
        modality_assignments : pandas.DataFrame
            The modality assignments of each feature, in each group of the
            groupby
        """
        scores = self.modality_log2bf(sample_ids=sample_ids,
                                      feature_ids=feature_ids, groupby=groupby,
                                      data=data, min_samples=min_samples)
        return scores.groupby(level=0, axis=0).apply(
            self.modality_estimator.assign_modalities, reset_index=True)


    @memoize
    def modality_counts(self, sample_ids=None, feature_ids=None, data=None,
                        groupby=None, min_samples=20):
        """Count the number of each modalities of these samples and features

        Parameters
        ----------
        sample_ids : list of str
            Which samples to use. If None, use all. Default None.
        feature_ids : list of str
            Which features to use. If None, use all. Default None.
        data : pandas.DataFrame, optional
            If provided, use this dataframe instead of the sample_ids and
            feature_ids provided
        min_samples : int, optional
            Minimum number of samples to use per grouped celltype. Default 10

        Returns
        -------
        modalities_counts : pandas.Series
            The number of events detected in each modality
        """
        assignments = self.modality_assignments(sample_ids, feature_ids, data,
                                                groupby, min_samples)
        counts = assignments.apply(lambda x: x.groupby(x).size(), axis=1)
        return counts

    def binify(self, data):
        return super(SplicingData, self).binify(data, self.bins)

    def plot_modalities_reduced(self, sample_ids=None, feature_ids=None,
                                data=None, ax=None, title=None,
                                min_samples=20):
        """Plot events modality assignments in NMF space

        This will calculate modalities on all samples provided, without
        grouping them by celltype. This is because each NMF axis can only show
        one set of sample ids' modalties.

        Parameters
        ----------
        sample_ids : list of str
            Which samples to use. If None, use all. Default None.
        feature_ids : list of str
            Which features to use. If None, use all. Default None.
        data : pandas.DataFrame, optional
            If provided, use this dataframe instead of the sample_ids and
            feature_ids provided
        min_samples : int, optional
            Minimum number of samples to use per grouped celltype. Default 10
        ax : matplotlib.axes.Axes object
            Axes to plot on. If none, gets current axes
        title : str
            Title of the reduced space plot
        """
        groupby = pd.Series('all', self.data.index)
        modality_assignments = self.modality_assignments(sample_ids,
                                                         feature_ids,
                                                         data, groupby,
                                                         min_samples)
        modality_assignments = pd.Series(modality_assignments.values[0],
                                         index=modality_assignments.columns)

        self.modality_visualizer.plot_reduced_space(
            self.binned_nmf_reduced(sample_ids, feature_ids),
            modality_assignments, ax=ax, title=title,
            xlabel=self._nmf_space_xlabel(groupby),
            ylabel=self._nmf_space_ylabel(groupby))

    def plot_modalities_bars(self, sample_ids=None, feature_ids=None,
                             data=None, groupby=None, phenotype_to_color=None,
                             percentages=False, ax=None, min_samples=20):
        """Make grouped barplots of the number of modalities per group

        Parameters
        ----------
        sample_ids : None or list of str
            Which samples to use. If None, use all
        feature_ids : None or list of str
            Which features to use. If None, use all
        color : None or matplotlib color
            Which color to use for plotting the lavalamps of these features
            and samples
        min_samples : int, optional
            Minimum number of samples to use per grouped celltype. Default 10
        """

        counts = self.modality_counts(
            sample_ids, feature_ids, data=data, groupby=groupby,
            min_samples=min_samples)

        # make sure this is always a dataframe
        if isinstance(counts, pd.Series):
            counts = pd.DataFrame([counts.values],
                                  index=counts.name,
                                  columns=counts.index)
        return self.modality_visualizer.bar(counts, phenotype_to_color,
                                            percentages=percentages, ax=ax)

    def plot_modalities_lavalamps(self, sample_ids=None, feature_ids=None,
                                  data=None, groupby=None,
                                  phenotype_to_color=None, min_samples=20):
        """Plot "lavalamp" scatterplot of each event

        Parameters
        ----------
        sample_ids : None or list of str
            Which samples to use. If None, use all
        feature_ids : None or list of str
            Which features to use. If None, use all
        color : None or matplotlib color
            Which color to use for plotting the lavalamps of these features
            and samples
        x_offset : numeric
            How much to offset the x-axis of each event. Useful if you want
            to plot the same event, but in several iterations with different
            celltypes or colors
        min_samples : int, optional
            Minimum number of samples to use per grouped celltype. Default 10
        """
        if groupby is None:
            groupby = pd.Series('all', index=self.data.index)

        assignments = self.modality_assignments(
            sample_ids, feature_ids, data=data, groupby=groupby,
            min_samples=min_samples)

        # make sure this is always a dataframe
        if isinstance(assignments, pd.Series):
            assignments = pd.DataFrame([assignments.values],
                                       index=assignments.name,
                                       columns=assignments.index)

        grouped = self.singles.groupby(groupby)
        nrows = assignments.groupby(
            level=0, axis=0).apply(
            lambda x: np.unique(x.values)).apply(lambda x: len(x)).sum()
        figsize = 10, nrows * 4
        fig, axes = plt.subplots(nrows=nrows, figsize=figsize)
        axes_iter = axes.flat

        yticks = [0, self.excluded_max, self.included_min, 1]
        for phenotype, modalities in assignments.iterrows():
            color = phenotype_to_color[phenotype]
            sample_ids = grouped.groups[phenotype]
            for modality, s in modalities.groupby(modalities):
                ax = axes_iter.next()
                psi = self.data.ix[sample_ids, s.index]
                # if modality == 'excluded': import pdb; pdb.set_trace()
                lavalamp(psi, color=color, ax=ax, yticks=yticks)
                ax.set_title('{} {}'.format(phenotype, modality))
        sns.despine()
        fig.tight_layout()

    def plot_event_modality_estimation(self, event_id, sample_ids=None,
                                       data=None,
                                       groupby=None, min_samples=20):
        """Plots the mathematical reasoning for an event's modality assignment

        Parameters
        ----------
        event_id : str
            Unique name of the splicing event
        sample_ids : list of str, optional
            Which sample ids to use
        data : pandas.DataFrame
            Which data to use, if e.g. you filtered splicing events on
            expression data
        groupby : mapping, optional
            A sample id to celltype mapping
        min_samples : int, optional
            Minimum number of samples to use per grouped celltype. Default 10
        """
        if data is None:
            data = self._subset(self.singles, sample_ids,
                                require_min_samples=False)
        else:
            if sample_ids is not None:
                raise ValueError(
                    'Can only specify `sample_ids` or `data`, but not both.')
        if groupby is None:
            groupby = pd.Series('all', index=data.index)

        grouped = data.groupby(groupby)
        if isinstance(min_samples, int):
            thresh = self._thresh_int
        elif isinstance(min_samples, float):
            thresh = self._thresh_float
        else:
            raise TypeError('Threshold for minimum samples for modality '
                            'detection can only be int or float, '
                            'not {}'.format(type(min_samples)))
        data = pd.concat([df.dropna(thresh=thresh(df, min_samples), axis=1)
                          for name, df in grouped])
        event = data[event_id]
        renamed = self.feature_renamer(event_id)
        logliks = self.modality_estimator._loglik(event)
        logsumexps = self.modality_estimator._logsumexp(logliks)
        self.modality_visualizer.event_estimation(event, logliks, logsumexps,
                                                  renamed=renamed)

    @memoize
    def _is_nmf_space_x_axis_excluded(self, phenotype_groupby):
        nmf_space_positions = self.nmf_space_positions(phenotype_groupby)

        # Get the correct included/excluded labeling for the x and y axes
        event, phenotype = nmf_space_positions.pc_1.argmax()
        top_pc1_samples = self.data.groupby(phenotype_groupby).groups[
            phenotype]

        data = self._subset(self.data, sample_ids=top_pc1_samples)
        binned = self.binify(data)
        return bool(binned[event][0])

    def _nmf_space_xlabel(self, phenotype_groupby):
        if self._is_nmf_space_x_axis_excluded(phenotype_groupby):
            return self.excluded_label
        else:
            return self.included_label

    def _nmf_space_ylabel(self, phenotype_groupby):
        if self._is_nmf_space_x_axis_excluded(phenotype_groupby):
            return self.included_label
        else:
            return self.excluded_label

    def plot_feature(self, feature_id, sample_ids=None,
                     phenotype_groupby=None,
                     phenotype_order=None, color=None,
                     phenotype_to_color=None,
                     phenotype_to_marker=None, nmf_xlabel=None,
                     nmf_ylabel=None,
                     nmf_space=False, fig=None, axesgrid=None, n=20):
        if nmf_space:
            nmf_xlabel = self._nmf_space_xlabel(phenotype_groupby)
            nmf_ylabel = self._nmf_space_ylabel(phenotype_groupby)
        else:
            nmf_ylabel = None
            nmf_xlabel = None

        super(SplicingData, self).plot_feature(feature_id, sample_ids,
                                               phenotype_groupby,
                                               phenotype_order, color,
                                               phenotype_to_color,
                                               phenotype_to_marker, nmf_xlabel,
                                               nmf_ylabel, nmf_space=nmf_space,
                                               fig=fig, axesgrid=axesgrid,
                                               n=n)

    @memoize
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

    @memoize
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
            subset = subset.fillna(0.5)
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


# class SpliceJunctionData(SplicingData):
# """Class for splice junction information from SJ.out.tab files from STAR
#
#     Attributes
#     ----------
#
#
#     Methods
#     -------
#
#     """
#
#     def __init__(self, df, phenotype_data):
#         """Constructor for SpliceJunctionData
#
#         Parameters
#         ----------
#         data, experiment_design_data
#
#         Returns
#         -------
#
#
#         Raises
#         ------
#
#         """
#         super(SpliceJunctionData).__init__()
#         pass
#
#
# class DownsampledSplicingData(BaseData):
#     binned_reducer = None
#     raw_reducer = None
#
#     n_components = 2
#     _binsize = 0.1
#     _var_cut = 0.2
#
#     def __init__(self, df, sample_descriptors):
#         """Instantiate an object of downsampled splicing data
#
#         Parameters
#         ----------
#         df : pandas.DataFrame
#             A "tall" dataframe of all miso summary events, with the usual
#             MISO summary columns, and these are required: 'splice_type',
#             'probability', 'iteration.' Where "probability" indicates the
#             randomly sampling probability from the bam file used to generate
#             these reads, and "iteration" indicates the integer iteration
#             performed, e.g. if multiple resamplings were performed.
#         experiment_design_data: pandas.DataFrame
#
#         Notes
#         -----
#         Warning: this data is usually HUGE (we're taking like 10GB raw .tsv
#         files) so make sure you have the available memory for dealing with
#         these.
#
#         """
#         super(DownsampledSplicingData, self).__init__(sample_descriptors)
#
#         self.sample_descriptors, splicing = \
#             self.sample_descriptors.align(df, join='inner', axis=0)
#
#         self.df = df
#
#     @property
#     def shared_events(self):
#         """
#         Parameters
#         ----------
#
#         Returns
#         -------
#         event_count_df : pandas.DataFrame
#             Splicing events on the rows, splice types and probability as
#             column MultiIndex. Values are the number of iterations which
#             share this splicing event at that probability and splice type.
#         """
#
#         if not hasattr(self, '_shared_events'):
#             shared_events = {}
#
#             for (splice_type, probability), df in self.df.groupby(
#                     ['splice_type', 'probability']):
#                 event_count = collections.Counter(df.event_name)
#                 shared_events[(splice_type, probability)] = pd.Series(
#                     event_count)
#
#             self._shared_events = pd.DataFrame(shared_events)
#             self._shared_events.columns = pd.MultiIndex.from_tuples(
#                 self._shared_events_df.columns.tolist())
#         else:
#             return self._shared_events
#
#     def shared_events_barplot(self, figure_dir='./'):
#         """PLot a "histogram" via colored bars of the number of events shared
#         by different iterations at a particular sampling probability
#
#         Parameters
#         ----------
#         figure_dir : str
#             Where to save the pdf figures created
#         """
#         figure_dir = figure_dir.rstrip('/')
#         colors = purples + ['#262626']
#
#         for splice_type, df in self.shared_events.groupby(level=0, axis=1):
#             print splice_type, df.dropna(how='all').shape
#
#             fig, ax = plt.subplots(figsize=(16, 4))
#
#             count_values = np.unique(df.values)
#             count_values = count_values[np.isfinite(count_values)]
#
#             height_so_far = np.zeros(df.shape[1])
#             left = np.arange(df.shape[1])
#
#             for count, color in zip(count_values, colors):
#                 height = df[df == count].count()
#                 ax.bar(left, height, bottom=height_so_far, color=color,
#                        label=str(int(count)))
#                 height_so_far += height
#             ymax = max(height_so_far)
#             ax.set_ylim(0, ymax)
#
#             legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
#                                title='Iterations sharing event')
#             ax.set_title(splice_type)
#             ax.set_xlabel('Percent downsampled')
#             ax.set_ylabel('number of events')
#             sns.despine()
#             fig.tight_layout()
#             filename = '{}/downsampled_shared_events_{}.pdf'.format(
#                 figure_dir, splice_type)
#             fig.savefig(filename, bbox_extra_artists=(legend,),
#                         bbox_inches='tight', format="pdf")
#
#     def shared_events_percentage(self, min_iter_shared=5, figure_dir='./'):
#         """Plot the percentage of all events detected at that iteration,
#         shared by at least 'min_iter_shared'
#
#         Parameters
#         ----------
#         min_iter_shared : int
#             Minimum number of iterations sharing an event
#         figure_dir : str
#             Where to save the pdf figures created
#         """
#         figure_dir = figure_dir.rstrip('/')
#         sns.set(style='whitegrid', context='talk')
#
#         for splice_type, df in self.shared_events.groupby(level=0, axis=1):
#             df = df.dropna()
#
#             fig, ax = plt.subplots(figsize=(16, 4))
#
#             left = np.arange(df.shape[1])
#             num_greater_than = df[df >= min_iter_shared].count()
#             percent_greater_than = num_greater_than / df.shape[0]
#
#             ax.plot(left, percent_greater_than,
#                     label='Shared with at least {} iter'.format(
#                         min_iter_shared))
#
#             ax.set_xticks(np.arange(0, 101, 10))
#
#             legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
#                                title='Iterations sharing event')
#
#             ax.set_title(splice_type)
#             ax.set_xlabel('Percent downsampled')
#             ax.set_ylabel('Percent of events')
#             sns.despine()
#             fig.tight_layout()
#             fig.savefig(
#                 '{}/downsampled_shared_events_{}_min_iter_shared{}.pdf'
#                 .format(figure_dir, splice_type, min_iter_shared),
#                 bbox_extra_artists=(legend,), bbox_inches='tight',
#                 format="pdf")
