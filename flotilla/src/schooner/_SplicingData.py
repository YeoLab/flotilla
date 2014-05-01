import numpy as np
from collections import defaultdict


from _Data import Data
from ..submarine import NMF_viz, PCA_viz
from ..frigate import binify, dropna_mean
import pandas as pd
from ...project.project_params import min_cells, _default_group_id
from sklearn.preprocessing import StandardScaler
from ..skiff import link_to_list

class SplicingData(Data):
    _default_reducer_args = Data._default_reducer_args
    _default_group_id = _default_group_id
    binned_reducer = None
    raw_reducer = None
    binsize=0.1,
    n_components = 2
    _binsize=None
    var_cut = 0.2

    ###
    #
    # TODO: make these databases, so you don't have to re-calculate

    event_lists = {}
    samplewise_reduction = defaultdict(dict)
    featurewise_reduction = defaultdict(dict)

    #
    ###

    def __init__(self, psi, sample_descriptors,
                 event_descriptors, binsize = binsize,
                 var_cut = var_cut
                 ):
        """Instantiate a object for data scores with binned and reduced data

        Parameters
        ----------
        df : pandas.DataFrame
            A [n_events, n_samples] dataframe of splicing events
        n_components : int
            Number of components to use in the reducer
        binsize : float
            Value between 0 and 1, the bin size for binning the data scores
        reducer : sklearn.decomposition object
            An scikit-learn class that reduces the dimensionality of data
            somehow. Must accept the parameter n_components, have the
            functions fit, transform, and have the attribute components_

        """
        self.psi = psi
        self.binsize = binsize
        psi_variant = pd.Index([i for i,j in (psi.var().dropna() > var_cut).iteritems() if j])
        self.event_lists['variant'] = psi_variant
        self.event_lists['default'] = self.event_lists['variant']
        self.sample_descriptors = sample_descriptors
        self.event_descriptors = event_descriptors

    def set_binsize(self, binsize):
        self.binsize = binsize

    def get_binned_data(self):
        try:
            assert hasattr(self, 'binned') #binned has been set
            assert self._binsize == self.binsize #binsize hasn't changed
        except:
            #only bin once, until binsize is updated
            self.binned = binify(self.psi, binsize=self.binsize)
            self._binsize = self.binsize
        return self.binned

    def get_binned_reduced(self, reducer=NMF_viz):
        binned = self.get_binned_data()
        redc = reducer(binned)
        self.binned_reduced = redc.reduced_space
        return self.binned_reduced

    def get_reduced(self, event_list='default', group_id=_default_group_id, min_cells=min_cells, reducer=PCA_viz, featurewise=False,
                         reducer_args=_default_reducer_args):
        if featurewise:
            rdc_dict = self.featurewise_reduction
        else:
            rdc_dict = self.samplewise_reduction
        try:
            return rdc_dict[event_list][group_id]
        except:

            if event_list not in self.event_lists:
                self.event_lists[event_list] = link_to_list(event_list)

            event_list = self.event_lists[event_list]
            #some samples, somefeatures
            subset = self.psi.ix[self.sample_descriptors[group_id], event_list]
            frequent = pd.Index([i for i,j in (subset.count() > min_cells).iteritems() if j])
            subset = subset[frequent]
            #fill na with mean for each event
            means = subset.apply(dropna_mean, axis=0)
            mf_subset = subset.fillna(means,).fillna(0)
            #whiten, mean-center
            naming_fun=self.get_naming_fun()
            ss = pd.DataFrame(StandardScaler().fit_transform(mf_subset), index=mf_subset.index,
                              columns=mf_subset.columns).rename_axis(naming_fun, 1)

            #compute pca
            #compute pca
            if featurewise:
                ss = ss.T
            rdc_obj = reducer(ss, **reducer_args)

            rdc_obj.means = means

            #add mean gene_expression
            rdc_dict[event_list][group_id] = rdc_obj
        return rdc_dict[event_list][group_id]


