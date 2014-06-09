import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from _Data import Data
from .._submaraine_viz import NMF_viz, PCA_viz, PredictorViz
from .._frigate_compute import binify, dropna_mean
from .._skiff_external_sources import link_to_list


import collections

class DownsampledSplicingData(Data):
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

    def _event_count_df(self):
        """
        Parameters
        ----------


        Returns
        -------


        Raises
        ------
        """


        event_count_dict = {}

        for (splice_type, probability), df in summary.groupby(
                ['splice_type', 'probability']):
            print splice_type, probability, df.shape, \
                df.event_name.unique().shape[0],
            n_iter = df.iteration.unique().shape[0]
            event_count = collections.Counter(df.event_name)
            print sum(1 for k, v in event_count.iteritems() if v == n_iter)
            event_count_dict[(splice_type, probability)] = pd.Series(
                event_count)

        event_count_df = pd.DataFrame(event_count_dict)
        event_count_df.columns = pd.MultiIndex.from_tuples(
            event_count_df.columns.tolist())
