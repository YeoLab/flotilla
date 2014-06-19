import collections
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import BaseData
from ..visualize.decomposition import NMFViz, PCAViz
from ..visualize.color import purples
from ..visualize.predict import PredictorViz
from ..compute.generic import binify, dropna_mean
from ..external import link_to_list
from ..util import memoize

class SplicingData(BaseData):
    binned_reducer = None
    raw_reducer = None

    n_components = 2
    _binsize=0.1
    _var_cut = 0.2

    def __init__(self, data,
                 feature_data=None, binsize=_binsize,
                 var_cut = _var_cut,
                 drop_outliers=True,
                 load_cargo=False,
                 **kwargs
                 ):
        """Instantiate a object for study_data scores with binned and reduced study_data

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

        """
        super(SplicingData, self).__init__(data,
                                           **kwargs)
        if drop_outliers:
            self.data = self.drop_outliers(data)
        # self.phenotype_data, data = self.phenotype_data.align(data, join='inner', axis=0)

        # self.data = data
        self.binsize = binsize
        psi_variant = pd.Index([i for i,j in (data.var().dropna() > var_cut).iteritems() if j])
        self._set_naming_fun(self.feature_rename)
        self.feature_sets['variant'] = pd.Series(psi_variant, index=psi_variant)
        self.feature_sets['all_genes'] =  pd.Series(data.index, index=data.index)
        self.feature_data = feature_data
        self._set_plot_colors()
        self._set_plot_markers()

    def feature_rename(self, x):
        "this is for miso psi IDs..."
        short = ":".join(x.split("@")[1].split(":")[:2])
        try:
            dd = self.event_metadata.set_index('event_name')
            return dd['gene_symbol'].ix[x] + " " + short
        except Exception as e:
            #print e
            return short

    @memoize
    def binify(self, bins):
        return binify(self.data, bins)

    # def set_binsize(self, binsize):
    #     self.binsize = binsize
    #
    # def get_binned_data(self):
    #     try:
    #         assert hasattr(self, 'binned') #binned has been set
    #         assert self._binsize == self.binsize #binsize hasn't changed
    #     except:
    #         #only bin once, until binsize is updated
    #         bins = np.arange(0, 1+self.binsize, self.binsize)
    #         self.binned = binify(self.data, bins)
    #         self._binsize = self.binsize
    #     return self.binned

    def get_binned_reduced(self, reducer=NMFViz):
        binned = self.get_binned_data()
        redc = reducer(binned)
        self.binned_reduced = redc.reduced_space
        return self.binned_reduced

    _last_reducer_accessed = None

    @memoize
    def reduce(self, list_name, group_id, reducer=NMFViz,
                    featurewise=False, reducer_args=None, standardize=True):
        """make and cache a reduced dimensionality representation of data """

        if reducer_args is None:
            reducer_args = self._default_reducer_args

        min_samples = self.get_min_samples()
        if list_name not in self.feature_sets:
            self.feature_sets[list_name] = link_to_list(list_name)

        event_list = self.feature_sets[list_name]
        #some samples, somefeatures

        if group_id.startswith("~"):
            #print 'not', group_id.lstrip("~")
            sample_ind = ~pd.Series(self.phenotype_data[group_id.lstrip("~")], dtype='bool')
        else:
            sample_ind = pd.Series(self.phenotype_data[group_id], dtype='bool')

        subset = self.data.ix[sample_ind, event_list]
        frequent = pd.Index([i for i,j in (subset.count() > min_samples).iteritems() if j])
        subset = subset[frequent]
        #fill na with mean for each event
        means = subset.apply(dropna_mean, axis=0)
        mf_subset = subset.fillna(means,).fillna(0)
        #whiten, mean-center
        naming_fun=self.get_feature_renamer()
        #whiten, mean-center

        if standardize:
            data = StandardScaler().fit_transform(mf_subset)
        else:
            data = mf_subset

        ss = pd.DataFrame(data, index = mf_subset.index,
                          columns = mf_subset.columns).rename_axis(naming_fun, 1)

        if featurewise:
            ss = ss.T

        rdc_obj = reducer(ss, **reducer_args)

        rdc_obj.means = means.rename_axis(naming_fun) #always the mean of input features... i.e. featurewise doesn't change this.

        return rdc_obj

    @memoize
    def classify(self, list_name, group_id, categorical_trait,
                       standardize=True, classifier=PredictorViz,
                       ):
        """
        make and cache a classifier on a categorical trait (associated with samples) subset of genes
         """

        min_samples=self.get_min_samples()
        if list_name not in self.feature_sets:
            self.feature_sets[list_name] = link_to_list(list_name)

        event_list = self.feature_sets[list_name]

        if group_id.startswith("~"):
            #print 'not', group_id.lstrip("~")
            sample_ind = ~pd.Series(self.phenotype_data[group_id.lstrip("~")], dtype='bool')
        else:
            sample_ind = pd.Series(self.phenotype_data[group_id], dtype='bool')
        sample_ind = sample_ind[sample_ind].index
        subset = self.data.ix[sample_ind, event_list]
        frequent = pd.Index([i for i, j in (subset.count() > min_samples).iteritems() if j])
        subset = subset[frequent]
        #fill na with mean for each event
        means = subset.apply(dropna_mean, axis=0)
        mf_subset = subset.fillna(means, ).fillna(0)

        #whiten, mean-center
        if standardize:
            data = StandardScaler().fit_transform(mf_subset)
        else:
            data = mf_subset
        naming_fun = self.get_feature_renamer()
        ss = pd.DataFrame(data, index = mf_subset.index,
                          columns = mf_subset.columns).rename_axis(naming_fun, 1)
        clf = classifier(ss, self.phenotype_data,
                        categorical_traits=[categorical_trait],)
        clf.set_reducer_plotting_args(self._default_reducer_args)
        return clf

    def load_cargo(self):
        raise NotImplementedError

    def _get(self, splicing_data_filename):
        return {'splicing_df': self.load(*splicing_data_filename)}



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
        data, phenotype_data

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
    _binsize=0.1
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
        phenotype_data: pandas.DataFrame

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

        # figure_dir = '/home/obotvinnik/Dropbox/figures2/singlecell/splicing/what_is_noise'

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

        # figure_dir = '/home/obotvinnik/Dropbox/figures2/singlecell/splicing/what_is_noise'

        for splice_type, df in self.shared_events.groupby(level=0, axis=1):
            df = df.dropna()

            fig, ax = plt.subplots(figsize=(16, 4))


            left = np.arange(df.shape[1])
            num_greater_than = df[df >= min_iter_shared].count()
            percent_greater_than = num_greater_than / df.shape[0]

            ax.plot(left, percent_greater_than,
                    label='Shared with at least {} iter'.format(min_iter_shared))

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