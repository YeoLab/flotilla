import collections
import sys
import itertools

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import seaborn as sns

from .base import BaseData
from ..compute.infotheory import binify
from ..compute.splicing import Modalities

from ..visualize.decomposition import NMFViz, PCAViz
from ..visualize.color import purples
from ..visualize.predict import ClassifierViz
from ..visualize.splicing import ModalitiesViz
from ..util import cached_property, memoize
from ..visualize.color import red, grey
from ..visualize.splicing import lavalamp


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
                 feature_rename_col=None, excluded_max=0.2, included_min=0.8):
        """Instantiate a object for percent spliced in (PSI) scores

        Parameters
        ----------
        data : pandas.DataFrame
            A [n_events, n_samples] dataframe of data events
        n_components : int
            Number of components to use in the reducer
        binsize : float
            Value between 0 and 1, the bin size for binning the study_data scores
        reducer : sklearn.decomposition object
            An scikit-learn class that reduces the dimensionality of study_data
            somehow. Must accept the parameter n_components, have the
            functions fit, transform, and have the attribute components_
        excluded_max : float
            Maximum value for the "excluded" bin of psi scores. Default 0.2.
        included_max : float
            Minimum value for the "included" bin of psi scores. Default 0.8.
        """
        super(SplicingData, self).__init__(data, metadata,
                                           feature_rename_col=feature_rename_col,
                                           outliers=outliers)
        self.binsize = binsize
        self.bins = np.arange(0, 1 + self.binsize, self.binsize)
        psi_variant = pd.Index(
            [i for i, j in (data.var().dropna() > var_cut).iteritems() if j])
        self.feature_sets['variant'] = psi_variant

        self.modalities_calculator = Modalities(excluded_max=excluded_max,
                                                included_min=included_min)
        self.modalities_visualizer = ModalitiesViz()

    @memoize
    def modalities(self, sample_ids=None, feature_ids=None):
        """Assigned modalities for these samples and features.

        Parameters
        ----------
        sample_ids : list of str
            Which samples to use. If None, use all. Default None.
        feature_ids : list of str
            Which features to use. If None, use all. Default None.

        Returns
        -------
        modality_assignments : pandas.Series
            The modality assignments of each feature given these samples
        """
        data = self._subset(self.data, sample_ids, feature_ids)
        return self.modalities_calculator.fit_transform(data)

    @memoize
    def modalities_counts(self, sample_ids=None, feature_ids=None):
        """Count the number of each modalities of these samples and features

        Parameters
        ----------
        sample_ids : list of str
            Which samples to use. If None, use all. Default None.
        feature_ids : list of str
            Which features to use. If None, use all. Default None.

        Returns
        -------
        modalities_counts : pandas.Series
            The number of events detected in each modality
        """
        data = self._subset(self.data, sample_ids, feature_ids)
        return self.modalities_calculator.counts(data)

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
        # redc = NMFViz(binned.T, n_components=2)

        reduced = self.nmf.transform(binned.T)

        # # Make sure x-axis (component 0) is excluded, which is the first
        # # element of a column in the binned dataframe
        # x0 = reduced.ix[reduced.pc_1 == 0]
        # if binned.ix[:, x0.index[0]][0] < 1:
        #     reduced = pd.concat([reduced.pc_2, reduced.pc_1],
        #                         keys=reduced.columns, axis=1)
        return reduced

    @memoize
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
        # feature_renamer = self.feature_renamer()

        subset, means = self._subset_and_standardize(data,
                                                     sample_ids, feature_ids,
                                                     standardize)

        # compute reduction
        if featurewise:
            subset = subset.T
        reducer_object = reducer(subset, **reducer_kwargs)
        reducer_object.means = means  #always the mean of input features... i
        # .e.
        # featurewise doesn't change this.

        #add mean gene_expression
        return reducer_object

    @memoize
    def classify(self, trait, sample_ids=None, feature_ids=None,
                 standardize=True, predictor=ClassifierViz,
                 predictor_kwargs=None, predictor_scoring_fun=None,
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
        subset, means = self._subset_and_standardize(self.data,
                                                     sample_ids,
                                                     feature_ids,
                                                     standardize)

        classifier = predictor(subset, trait=trait,
                               predictor_kwargs=predictor_kwargs,
                               predictor_scoring_fun=predictor_scoring_fun,
                               score_cutoff_fun=score_cutoff_fun,
                               **plotting_kwargs)
        # classifier.set_reducer_plotting_args(classifier.reduction_kwargs)
        return classifier

    def plot_modalities_reduced(self, sample_ids=None, feature_ids=None,
                                ax=None, title=None):
        """Plot modality assignments in NMF space (option for lavalamp?)
        """
        modalities_assignments = self.modalities(sample_ids, feature_ids)
        self.modalities_visualizer.plot_reduced_space(
            self.binned_reduced(sample_ids, feature_ids),
            modalities_assignments, ax=ax, title=title)

    def plot_modalities_bar(self, sample_ids=None, feature_ids=None, ax=None,
                            i=0, normed=True, legend=True):
        modalities_counts = self.modalities_counts(sample_ids, feature_ids)
        self.modalities_visualizer.bar(modalities_counts, ax, i, normed,
                                       legend)
        modalities_fractions = modalities_counts / modalities_counts.sum().astype(
            float)
        sys.stdout.write(str(modalities_fractions) + '\n')

    def plot_modalities_lavalamps(self, sample_ids=None, feature_ids=None,
                                 color=None, **kwargs):
        """Plot modality assignments in NMF space (option for lavalamp?)
        """
        modalities_assignments = self.modalities(sample_ids, feature_ids)
        modalities_names = self.modalities_calculator.modalities_names

        f, axes = plt.subplots(len(modalities_names), 1, figsize=(18, 3*len(modalities_names)))
        axes = itertools.chain(axes)

        if color is None:
            color = pd.Series(red, index=modalities_assignments.index)
        else:
            color = color.ix[self.data.index]
            color = color.fillna(rgb2hex(grey))

        for modality in modalities_names:
            ax = axes.next()
            modal_psis = self.data[modalities_assignments[modalities_assignments == modality].index]
            lavalamp(modal_psis, color=color, ax=ax, **kwargs)
            ax.set_title(modality)


    def plot_event(self, feature_id, sample_groupby, sample_colors):
        pass

        # def plot_shared_events(self):


class SpliceJunctionData(SplicingData):
    """Class to hold splice junction information from SJ.out.tab files from STAR

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
                # print splice_type, probability, data.shape, \
                #     data.event_name.unique().shape[0],
                # n_iter = data.iteration.unique().shape[0]
                event_count = collections.Counter(df.event_name)
                # print sum(1 for k, v in event_count.iteritems() if v == n_iter)
                shared_events[(splice_type, probability)] = pd.Series(
                    event_count)

            self._shared_events = pd.DataFrame(shared_events)
            self._shared_events.columns = pd.MultiIndex.from_tuples(
                self._shared_events_df.columns.tolist())
        else:
            return self._shared_events

    def shared_events_barplot(self, figure_dir='./'):
        """PLot a "histogram" via colored bars of the number of events shared by
        different iterations at a particular sampling probability

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
            fig.savefig('{}/downsampled_shared_events_{}.pdf'.format(figure_dir,
                                                                     splice_type),
                        bbox_extra_artists=(legend,), bbox_inches='tight')

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