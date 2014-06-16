import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from _Data import BaseData
from .._submaraine_viz import NMF_viz, PCA_viz, PredictorViz
from .._frigate_compute import binify, dropna_mean
from .._skiff_external_sources import link_to_list
import brewer2mpl
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='ticks', context='talk')

PURPLES = brewer2mpl.get_map('Purples', 'sequential', 9).mpl_colors

import collections

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
        df : pandas.DataFrame
            A "tall" dataframe of all miso summary events, with the usual
            MISO summary columns, and these are required: 'splice_type',
            'probability', 'iteration.' Where "probability" indicates the
            randomly sampling probability from the bam file used to generate
            these reads, and "iteration" indicates the integer iteration
            performed, e.g. if multiple resamplings were performed.
        sample_metadata: pandas.DataFrame

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

        if not hasattr(self, '_shared_events_df'):
            shared_events_dict = {}

            for (splice_type, probability), df in self.df.groupby(
                    ['splice_type', 'probability']):
                # print splice_type, probability, df.shape, \
                #     df.event_name.unique().shape[0],
                # n_iter = df.iteration.unique().shape[0]
                event_count = collections.Counter(df.event_name)
                # print sum(1 for k, v in event_count.iteritems() if v == n_iter)
                shared_events_dict[(splice_type, probability)] = pd.Series(
                    event_count)

            self._shared_events_df = pd.DataFrame(shared_events_dict)
            self._shared_events_df.columns = pd.MultiIndex.from_tuples(
                self._shared_events_df.columns.tolist())
        else:
            return self._shared_events_df

    def shared_events_barplot(self, figure_dir='./'):
        """PLot a "histogram" via colored bars of the number of events shared by
        different iterations at a particular sampling probability

        Parameters
        ----------
        figure_dir : str
            Where to save the pdf figures created
        """
        figure_dir = figure_dir.rstrip('/')
        colors = PURPLES + ['#262626']

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