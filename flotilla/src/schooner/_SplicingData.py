import numpy as np
from _Data import Data
from ..frigate import binify, NMF, PCA
import pandas as pd
from ...project.project_params import min_cells

from ..skiff import link_to_list
class SplicingData(Data):

    binned_reducer = None
    raw_reducer = None
    binsize=0.1,
    n_components = 2
    event_lists = {}
    samplewise_reduction = {}
    featurewise_reduction = {}
    def __init__(self, psi, binsize = binsize, n_components=n_components,
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
        self.n_components = n_components
        psi_variant = pd.Index([i for i,j in (psi.var().dropna() > var_cut).iteritems() if j])
        self.event_lists['variant'] = psi_variant
        self.event_lists['default'] = event_lists['variant']

    def binify(self, binsize=binsize):

        self.binned = binify(self.df, binsize=self.binsize)

    def reduce(self, reducer=NMF,  n_components=2):
        """
        Reduces dimensionality of the binned df score data
        """
        reducer = reducer if reducer else self._reducer
        assert reducer is not None

        rdc = reducer(n_components=self.n_components)

        self.reduced_binned = rdc.transform(self.binned)
        if hasattr(self.reducer, 'explained_variance_ratio_'):
            self.plot_explained_variance(self.reducer,
                                         '{} on binned data'.format(self.reducer))
        return self

    def reduce_binned(self, reducer=None):
        reducer = reducer(n_components=self.n_components) if reducer else self._reducer
        reducer.fit(self.binned)
        self.binned_reduced = self.reduce(self.binned, self._reducer)




    def get_featurewise_reduced_dims(self, event_list, letter, min_cells=min_cells, reducer=PCA):
        try:
            return self.featurewise_reduction[event_list][letter]
        except:

            if event_list not in event_lists:
                event_lists[event_list] = link_to_list(event_list)

            event_list = event_lists[event_list]
            sparse_subset = self.psi.ix[descriptors[letter+"_cell"], event_list]
            frequent = pd.Index([i for i,j in (sparse_subset.count() > min_cells).iteritems() if j])
            sparse_subset = sparse_subset[frequent]
            #fill na with mean for each event
            means = sparse_subset.apply(dropna_mean, axis=0)
            mf_subset = sparse_subset.fillna(means,).fillna(0)
            #whiten, mean-center
            ss = pd.DataFrame(StandardScaler().fit_transform(mf_subset), index=mf_subset.index,
                              columns=mf_subset.columns).rename_axis(go.geneNames, 1)
            #compute pca
            pca_obj = PCA_viz(ss.T, whiten=False)
            pca_obj.means = means

            #add mean gene_expression
            featurewise_reduction[event_list][letter] = pca_obj
        return featurewise_reduction[event_list][letter]



def get_cellwise_event_pca(event_list, letter):
    try:
        return cellwise_psi_pca[event_list][letter]
    except:

        if event_list not in event_lists:
            event_lists[event_list] = link_to_list(event_list)
        other = event_lists[event_list]

        variant = pd.Index([i for i,j in (psi.var().dropna() > var_cut).iteritems() if j])
        event_list = variant
        sparse_subset = psi.ix[descriptors[letter+"_cell"], event_list]
        frequent = pd.Index([i for i,j in (sparse_subset.count() > min_cells).iteritems() if j])
        sparse_subset = sparse_subset[frequent]
        #fill na with mean for each gene
        means = sparse_subset.apply(dropna_mean, axis=0)
        mf_subset = sparse_subset.fillna(means,).fillna(0)
        #whiten, mean-center
        ss = pd.DataFrame(StandardScaler().fit_transform(mf_subset), index=mf_subset.index,
                          columns=mf_subset.columns).rename_axis(go.geneNames, 1)
        #compute pca
        pca_obj = PCA_viz(ss, whiten=False)
        pca_obj.means = means

        #add mean gene_expression
        cellwise_psi_pca[event_list][letter] = pca_obj
    return cellwise_psi_pca[event_list][letter]
event_lists = dict() # don't have any of these right now.



var_cut = 0.2

