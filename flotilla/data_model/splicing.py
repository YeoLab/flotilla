import collections
import sys
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseData
from ..compute.infotheory import binify
from ..compute.splicing import Modalities
from ..visualize.decomposition import NMFViz, PCAViz
from ..visualize.color import purples
from ..visualize.predict import ClassifierViz
from ..visualize.splicing import ModalitiesViz
from ..util import cached_property, memoize
from ..visualize.color import red
from ..visualize.splicing import lavalamp, hist_single_vs_pooled_diff, \
    lavalamp_pooled_inconsistent, psi_violinplot


FRACTION_DIFF_THRESH = 0.1


class SplicingData(BaseData):
    binned_reducer = None
    raw_reducer = None

    n_components = 2
    _binsize = 0.1
    _var_cut = 0.2

    _last_reducer_accessed = None

    def __init__(self, data,
                 metadata=None, binsize=0.1,
                 var_cut=_var_cut, outliers=None,
                 feature_rename_col=None, excluded_max=0.2, included_min=0.8,
                 pooled=None):
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
        reducer : sklearn.decomposition object
            An scikit-learn class that reduces the dimensionality of study_data
            somehow. Must accept the parameter n_components, have the
            functions fit, transform, and have the attribute components_
        excluded_max : float
            Maximum value for the "excluded" bin of psi scores. Default 0.2.
        included_max : float
            Minimum value for the "included" bin of psi scores. Default 0.8.
        """
        super(SplicingData, self).__init__(
            data, metadata,
            feature_rename_col=feature_rename_col,
            outliers=outliers, pooled=pooled)

        self.binsize = binsize
        self.bins = np.arange(0, 1 + self.binsize, self.binsize)
        psi_variant = pd.Index(
            [i for i, j in (data.var().dropna() > var_cut).iteritems() if j])
        self.feature_subsets['variant'] = psi_variant

        self.modalities_calculator = Modalities(excluded_max=excluded_max,
                                                included_min=included_min)
        self.modalities_visualizer = ModalitiesViz()

        try:
            for modality in set(self.modalities()):
                self.feature_data[
                    'modality_' + modality] = self.modalities() == modality
        except TypeError:
            # Unless there is no feature_data
            pass

    @memoize
    def modalities(self, sample_ids=None, feature_ids=None,
                   bootstrapped=False, bootstrapped_kws=None):
        """Assigned modalities for these samples and features.

        Parameters
        ----------
        sample_ids : list of str
            Which samples to use. If None, use all. Default None.
        feature_ids : list of str
            Which features to use. If None, use all. Default None.
        bootstrapped : bool
            Whether or not to use bootstrapping, i.e. resample each splicing
            event several times to get a better estimate of its true modality.
        bootstrappped_kws : dict
            Valid arguments to _bootstrapped_fit_transform. If None, default is
            dict(n_iter=100, thresh=0.6, min_samples=10)

        Returns
        -------
        modality_assignments : pandas.Series
            The modality assignments of each feature given these samples
        """
        data = self._subset(self.data, sample_ids, feature_ids)
        return self.modalities_calculator.fit_transform(data, bootstrapped,
                                                        bootstrapped_kws)

    @memoize
    def modalities_counts(self, sample_ids=None, feature_ids=None,
                          bootstrapped=False, bootstrapped_kws=False):
        """Count the number of each modalities of these samples and features

        Parameters
        ----------
        sample_ids : list of str
            Which samples to use. If None, use all. Default None.
        feature_ids : list of str
            Which features to use. If None, use all. Default None.
        bootstrapped : bool
            Whether or not to use bootstrapping, i.e. resample each splicing
            event several times to get a better estimate of its true modality.
            Default False.
        bootstrappped_kws : dict
            Valid arguments to _bootstrapped_fit_transform. If None, default is
            dict(n_iter=100, thresh=0.6, min_samples=10)

        Returns
        -------
        modalities_counts : pandas.Series
            The number of events detected in each modality
        """
        data = self._subset(self.data, sample_ids, feature_ids)
        return self.modalities_calculator.counts(data, bootstrapped,
                                                 bootstrapped_kws)

    def binify(self, data):
        return binify(data, self.bins)

    @cached_property()
    def nmf(self):
        data = self._subset(self.data)
        return NMFViz(self.binify(data).T, n_components=2)

    @memoize
    def binned_reduced(self, sample_ids=None, feature_ids=None):
        """

        """
        data = self._subset(self.data, sample_ids, feature_ids)
        binned = self.binify(data)
        reduced = self.nmf.transform(binned.T)
        return reduced

    def reduce(self, sample_ids=None, feature_ids=None,
               featurewise=False, reducer=PCAViz,
               standardize=True, title='',
               reducer_kwargs=None, bins=None):
        """make and cache a reduced dimensionality representation of data

        Default is PCAViz because
        """
        if bins is not None:
            data = self.binify(bins)
        else:
            data = self.data

        reducer_kwargs = {} if reducer_kwargs is None else reducer_kwargs
        reducer_kwargs['title'] = title
        subset, means = self._subset_and_standardize(data,
                                                     sample_ids, feature_ids,
                                                     standardize,
                                                     return_means=True)

        # compute reduction
        if featurewise:
            subset = subset.T
        reducer_object = reducer(subset, **reducer_kwargs)

        # always the mean of input features. i.e. featurewise doesn't change
        # this.
        reducer_object.means = means
        return reducer_object

    @memoize
    def classify(self, trait, sample_ids=None, feature_ids=None,
                 standardize=True, predictor=ClassifierViz,
                 predictor_kwargs=None, predictor_scoring_fun=None,
                 score_coefficient=None,
                 score_cutoff_fun=None, plotting_kwargs=None):
        """Make and memoize a predictor on a categorical trait (associated
        with samples) subset of genes

        Parameters
        ----------
        trait : pandas.Series
            samples x categorical feature
        sample_ids : None or list of strings
            If None, all sample ids will be used, else only the sample ids
            specified
        feature_ids : None or list of strings
            If None, all features will be used, else only the features
            specified
        standardize : bool
            Whether or not to "whiten" (make all variables uncorrelated) and
            mean-center and make unit-variance all the data via sklearn
            .preprocessing.StandardScaler
        predictor : flotilla.visualize.predict classifier
            Must inherit from flotilla.visualize.PredictorBaseViz. Default is
            flotilla.visualize.predict.ClassifierViz
        predictor_kwargs : dict or None
            Additional 'keyword arguments' to supply to the predictor class
        predictor_scoring_fun : function
            Function to get the feature scores for a scikit-learn classifier.
            This can be different for different classifiers, e.g. for a
            classifier named "x" it could be x.scores_, for other it's
            x.feature_importances_. Default: lambda x: x.feature_importances_
        score_cutoff_fun : function
            Function to cut off insignificant scores
            Default: lambda scores: np.mean(x) + 2 * np.std(x)

        Returns
        -------
        predictor : flotilla.compute.predict.PredictorBaseViz
            A ready-to-plot object containing the predictions
        """
        subset = self._subset_and_standardize(self.data,
                                              sample_ids,
                                              feature_ids,
                                              standardize)
        if plotting_kwargs is None:
            plotting_kwargs = {}

        classifier = predictor(subset, trait=trait,
                               predictor_kwargs=predictor_kwargs,
                               predictor_scoring_fun=predictor_scoring_fun,
                               score_cutoff_fun=score_cutoff_fun,
                               score_coefficient=score_coefficient,
                               **plotting_kwargs)
        # classifier.set_reducer_plotting_args(classifier.reduction_kwargs)
        return classifier

    def plot_modalities_reduced(self, sample_ids=None, feature_ids=None,
                                ax=None, title=None,
                                bootstrapped=False, bootstrapped_kws=None):
        """Plot modality assignments in NMF space (option for lavalamp?)

        Parameters
        ----------
        bootstrapped : bool
            Whether or not to use bootstrapping, i.e. resample each splicing
            event several times to get a better estimate of its true modality.
            Default False.
        bootstrappped_kws : dict
            Valid arguments to _bootstrapped_fit_transform. If None, default is
            dict(n_iter=100, thresh=0.6, min_samples=10)


        Returns
        -------


        Raises
        ------
        """
        modalities_assignments = self.modalities(
            sample_ids, feature_ids, bootstrapped=bootstrapped,
            bootstrapped_kws=bootstrapped_kws)
        self.modalities_visualizer.plot_reduced_space(
            self.binned_reduced(sample_ids, feature_ids),
            modalities_assignments, ax=ax, title=title)

    def plot_modalities_bar(self, sample_ids=None, feature_ids=None, ax=None,
                            i=0, normed=True, legend=True,
                            bootstrapped=False, bootstrapped_kws=None):
        """Plot stacked bar graph of each modality

        Parameters
        ----------
        bootstrapped : bool
            Whether or not to use bootstrapping, i.e. resample each splicing
            event several times to get a better estimate of its true modality.
            Default False.
        bootstrappped_kws : dict
            Valid arguments to _bootstrapped_fit_transform. If None, default is
            dict(n_iter=100, thresh=0.6, min_samples=10)


        Returns
        -------


        Raises
        ------
        """
        modalities_counts = self.modalities_counts(
            sample_ids, feature_ids, bootstrapped=bootstrapped,
            bootstrapped_kws=bootstrapped_kws)
        self.modalities_visualizer.bar(modalities_counts, ax, i, normed,
                                       legend)
        modalities_fractions = \
            modalities_counts / modalities_counts.sum().astype(float)
        sys.stdout.write(str(modalities_fractions) + '\n')

    def plot_modalities_lavalamps(self, sample_ids=None, feature_ids=None,
                                  color=None, x_offset=0,
                                  use_these_modalities=True,
                                  bootstrapped=False, bootstrapped_kws=None,
                                  ax=None):
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
        axes : None or list of matplotlib.axes.Axes objects
            Which axes to plot these on
        use_these_modalities : bool
            If True, then use these sample ids to calculate modalities.
            Otherwise, use the modalities assigned using ALL samples and
            features
        bootstrapped : bool
            Whether or not to use bootstrapping, i.e. resample each splicing
            event several times to get a better estimate of its true modality.
            Default False.
        bootstrappped_kws : dict
            Valid arguments to _bootstrapped_fit_transform. If None, default is
            dict(n_iter=100, thresh=0.6, min_samples=10)
        """

        if use_these_modalities:
            modalities_assignments = self.modalities(
                sample_ids, feature_ids, bootstrapped=bootstrapped,
                bootstrapped_kws=bootstrapped_kws)
        else:
            modalities_assignments = self.modalities(
                bootstrapped=bootstrapped, bootstrapped_kws=bootstrapped_kws)
        modalities_names = modalities_assignments.unique()
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
        import matplotlib.pyplot as plt

        gs_x = len(modalities_names)
        gs_y = 15

        if ax is None:
            fig, ax = plt.subplots(1, 1,
                                   figsize=(18, 3 * len(modalities_names)))
            gs = GridSpec(gs_x, gs_y)

        else:
            gs = GridSpecFromSubplotSpec(gs_x, gs_y, ax.get_subplotspec())
            fig = plt.gcf()

        lavalamp_axes = [plt.subplot(gs[i, :12]) for i in
                         xrange(len(modalities_names))]
        pie_axis = plt.subplot(gs[:, 12:])
        pie_axis.set_aspect('equal')
        pie_axis.axis('off')
        if color is None:
            color = pd.Series(red, index=modalities_assignments.index)

        modalities_grouped = modalities_assignments.groupby(
            modalities_assignments)
        modality_count = {}
        for ax, (modality, s) in itertools.izip(lavalamp_axes,
                                                modalities_grouped):
            modality_count[modality] = len(s)
            psi = self.data[s.index]
            lavalamp(psi, color=color, ax=ax, x_offset=x_offset)
            ax.set_title(modality)
        pie_axis.pie(map(int, modality_count.values()),
                     labels=modality_count.keys(), autopct='%1.1f%%')

    def plot_event(self, feature_id, sample_ids=None, phenotype_groupby=None,
                   phenotype_order=None, ax=None, color=None):
        """
        Plot the violinplot of a splicing event (should also show NMF movement)
        """
        if ax is None:
            ax = plt.gca()

        singles, pooled = self._subset_singles_and_pooled(
            self.data, self.pooled, sample_ids, [feature_id])
        title = self.feature_renamer(feature_id)
        title = '{} {}'.format(title, ':'.join(feature_id.split(':')[:2]))

        psi_violinplot(singles, groupby=phenotype_groupby, color=color,
                       pooled_psi=pooled, order=phenotype_order,
                       title=title)
        # psi = self.data.ix[sample_ids, feature_id].dropna()

        # import pdb; pdb.set_trace()

        # Add a tiny amount of uniform random noise in case all the values
        # are equal
        # psi += np.random.uniform(0, 0.01, psi.shape)
        # sns.violinplot(psi, groupby=phenotype_groupby, ax=ax, bw=0.2,
        #                inner='points', order=phenotype_order)
        # sns.despine()
        # ax.set_ylim(0, 1)
        # ax.set_yticks((0, 0.5, 1))
        # ax.set_ylabel('PSI ($\Psi$) scores')
        #
        # pooled_grouped = self.pooled.groupby(phenotype_groupby, axis=0)
        #
        # for i, (celltype, df) in enumerate(pooled_grouped):
        #     ys = df.ix[:, feature_id]
        #     xs = np.ones(ys.shape) * i
        #     for x, y in zip(xs, ys):
        #         ax.scatter(x, y, marker='o', color='k', s=100)
        #         ax.annotate('pooled', (x, y), textcoords='offset points',
        #                     xytext=(10, 5), fontsize=14)

    @memoize
    def pooled_inconsistent(self, sample_ids, feature_ids=None,
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
        diff_from_singles = self._diff_from_singles(sample_ids,
                                                    feature_ids, scaled=True)

        large_diff = \
            diff_from_singles[diff_from_singles.abs()
                              >= fraction_diff_thresh].dropna(axis=1,
                                                              how='all')
        return large_diff

    @memoize
    def _diff_from_singles(self, sample_ids,
                           feature_ids=None, scaled=True, dropna=True):
        singles, pooled = self._subset_singles_and_pooled(
            self.data, self.pooled, sample_ids, feature_ids)

        diff_from_singles = pooled.apply(
            lambda x: (singles - x.values).abs().sum(), axis=1)

        if scaled:
            diff_from_singles = \
                diff_from_singles / singles.count().astype(float)
        if dropna:
            diff_from_singles = diff_from_singles.dropna(axis=1, how='all')
        return diff_from_singles

    def plot_lavalamp_pooled_inconsistent(
            self, sample_ids, feature_ids=None,
            fraction_diff_thresh=FRACTION_DIFF_THRESH, color=None):
        pooled_inconsistent = self.pooled_inconsistent(sample_ids,
                                                       feature_ids,
                                                       fraction_diff_thresh)
        singles, pooled = self._subset_singles_and_pooled(
            self.data, self.pooled, sample_ids, feature_ids)
        percent = self.percent_pooled_inconsistent(sample_ids, feature_ids,
                                                   fraction_diff_thresh)
        lavalamp_pooled_inconsistent(singles, pooled, pooled_inconsistent,
                                     color=color, percent=percent)

    def plot_hist_single_vs_pooled_diff(self, sample_ids,
                                        feature_ids=None,
                                        color=None, title='', hist_kws=None):
        diff_from_singles = self._diff_from_singles(sample_ids,
                                                    feature_ids)
        diff_from_singles_scaled = self._diff_from_singles(sample_ids,
                                                           feature_ids,
                                                           scaled=True)
        hist_single_vs_pooled_diff(diff_from_singles,
                                   diff_from_singles_scaled, color=color,
                                   title=title, hist_kws=hist_kws)

    @memoize
    def percent_pooled_inconsistent(self, sample_ids,
                                    feature_ids=None,
                                    fraction_diff_thresh=FRACTION_DIFF_THRESH):
        """The percent of splicing events which are

        """
        singles, pooled = self._subset_singles_and_pooled(
            self.data, self.pooled, sample_ids, feature_ids)
        large_diff = self.pooled_inconsistent(sample_ids, feature_ids,
                                              fraction_diff_thresh)
        return large_diff.shape[1] / float(pooled.shape[1]) * 100


class SpliceJunctionData(SplicingData):
    """Class to hold splice junction information from SJ.out.tab files from
    STAR

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, df, phenotype_data):
        """Constructor for SpliceJunctionData

        Parameters
        ----------
        data, experiment_design_data

        Returns
        -------


        Raises
        ------

        """
        super(SpliceJunctionData).__init__()
        pass


class DownsampledSplicingData(BaseData):
    binned_reducer = None
    raw_reducer = None

    n_components = 2
    _binsize = 0.1
    _var_cut = 0.2

    def __init__(self, df, sample_descriptors):
        """Instantiate an object of downsampled splicing data

        Parameters
        ----------
        data : pandas.DataFrame
            A "tall" dataframe of all miso summary events, with the usual
            MISO summary columns, and these are required: 'splice_type',
            'probability', 'iteration.' Where "probability" indicates the
            randomly sampling probability from the bam file used to generate
            these reads, and "iteration" indicates the integer iteration
            performed, e.g. if multiple resamplings were performed.
        experiment_design_data: pandas.DataFrame

        Notes
        -----
        Warning: this data is usually HUGE (we're taking like 10GB raw .tsv
        files) so make sure you have the available memory for dealing with
        these.

        """
        super(DownsampledSplicingData, self).__init__(sample_descriptors)

        self.sample_descriptors, splicing = \
            self.sample_descriptors.align(df, join='inner', axis=0)

        self.df = df

    @property
    def shared_events(self):
        """
        Parameters
        ----------

        Returns
        -------
        event_count_df : pandas.DataFrame
            Splicing events on the rows, splice types and probability as
            column MultiIndex. Values are the number of iterations which
            share this splicing event at that probability and splice type.
        """

        if not hasattr(self, '_shared_events'):
            shared_events = {}

            for (splice_type, probability), df in self.df.groupby(
                    ['splice_type', 'probability']):
                event_count = collections.Counter(df.event_name)
                shared_events[(splice_type, probability)] = pd.Series(
                    event_count)

            self._shared_events = pd.DataFrame(shared_events)
            self._shared_events.columns = pd.MultiIndex.from_tuples(
                self._shared_events_df.columns.tolist())
        else:
            return self._shared_events

    def shared_events_barplot(self, figure_dir='./'):
        """PLot a "histogram" via colored bars of the number of events shared
        by different iterations at a particular sampling probability

        Parameters
        ----------
        figure_dir : str
            Where to save the pdf figures created
        """
        figure_dir = figure_dir.rstrip('/')
        colors = purples + ['#262626']

        for splice_type, df in self.shared_events.groupby(level=0, axis=1):
            print splice_type, df.dropna(how='all').shape

            fig, ax = plt.subplots(figsize=(16, 4))

            count_values = np.unique(df.values)
            count_values = count_values[np.isfinite(count_values)]

            height_so_far = np.zeros(df.shape[1])
            left = np.arange(df.shape[1])

            for count, color in zip(count_values, colors):
                height = df[df == count].count()
                ax.bar(left, height, bottom=height_so_far, color=color,
                       label=str(int(count)))
                height_so_far += height
            ymax = max(height_so_far)
            ax.set_ylim(0, ymax)

            legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                               title='Iterations sharing event')
            ax.set_title(splice_type)
            ax.set_xlabel('Percent downsampled')
            ax.set_ylabel('number of events')
            sns.despine()
            fig.tight_layout()
            filename = '{}/downsampled_shared_events_{}.pdf'.format(
                figure_dir, splice_type)
            fig.savefig(filename, bbox_extra_artists=(legend,),
                        bbox_inches='tight')

    def shared_events_percentage(self, min_iter_shared=5, figure_dir='./'):
        """Plot the percentage of all events detected at that iteration,
        shared by at least 'min_iter_shared'

        Parameters
        ----------
        min_iter_shared : int
            Minimum number of iterations sharing an event
        figure_dir : str
            Where to save the pdf figures created
        """
        figure_dir = figure_dir.rstrip('/')
        sns.set(style='whitegrid', context='talk')

        for splice_type, df in self.shared_events.groupby(level=0, axis=1):
            df = df.dropna()

            fig, ax = plt.subplots(figsize=(16, 4))

            left = np.arange(df.shape[1])
            num_greater_than = df[df >= min_iter_shared].count()
            percent_greater_than = num_greater_than / df.shape[0]

            ax.plot(left, percent_greater_than,
                    label='Shared with at least {} iter'.format(
                        min_iter_shared))

            ax.set_xticks(np.arange(0, 101, 10))

            legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                               title='Iterations sharing event')

            ax.set_title(splice_type)
            ax.set_xlabel('Percent downsampled')
            ax.set_ylabel('Percent of events')
            sns.despine()
            fig.tight_layout()
            fig.savefig('{}/downsampled_shared_events_{}_min_iter_shared{}.pdf'
                        .format(figure_dir, splice_type, min_iter_shared),
                        bbox_extra_artists=(legend,), bbox_inches='tight')
