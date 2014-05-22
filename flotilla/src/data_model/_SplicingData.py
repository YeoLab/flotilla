import pandas as pd
from sklearn.preprocessing import StandardScaler

from _Data import Data
from ..viz import NMF_viz, PCA_viz
from ..compute import binify, dropna_mean
from ..external import link_to_list


class SplicingData(Data):
    _default_reducer_args = Data._default_reducer_args
    binned_reducer = None
    raw_reducer = None
    binsize=0.1,
    n_components = 2
    _binsize=None
    var_cut = 0.2
    lists = {}
    samplewise_reduction = {}
    featurewise_reduction = {}

    def __init__(self, splicing, sample_descriptors,
                 event_descriptors, binsize = binsize,
                 var_cut = var_cut
                 ):
        """Instantiate a object for study_data scores with binned and reduced study_data

        Parameters
        ----------
        df : pandas.DataFrame
            A [n_events, n_samples] dataframe of splicing events
        n_components : int
            Number of components to use in the reducer
        binsize : float
            Value between 0 and 1, the bin size for binning the study_data scores
        reducer : sklearn.decomposition object
            An scikit-learn class that reduces the dimensionality of study_data
            somehow. Must accept the parameter n_components, have the
            functions fit, transform, and have the attribute components_

        """
        self.psi = splicing
        self.binsize = binsize
        psi_variant = pd.Index([i for i,j in (splicing.var().dropna() > var_cut).iteritems() if j])

        self.lists['variant'] = pd.Series(psi_variant, index=psi_variant)
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

    _last_reducer_accessed = None

    def make_reduced(self, list_name, group_id,  min_cells=min_cells, reducer=PCA_viz,
                    featurewise=False, reducer_args=_default_reducer_args):
        """make and cache a reduced dimensionality representation of data """


        if list_name not in self.lists:
            self.lists[list_name] = link_to_list(list_name)

        event_list = self.lists[list_name]
        #some samples, somefeatures

        if group_id.startswith("~"):
            #print 'not', group_id.lstrip("~")
            sample_ind = ~pd.Series(self.sample_descriptors[group_id.lstrip("~")], dtype='bool')
        else:
            sample_ind = pd.Series(self.sample_descriptors[group_id], dtype='bool')

        subset = self.psi.ix[sample_ind, event_list]
        frequent = pd.Index([i for i,j in (subset.count() > min_cells).iteritems() if j])
        subset = subset[frequent]
        #fill na with mean for each event
        means = subset.apply(dropna_mean, axis=0)
        mf_subset = subset.fillna(means,).fillna(0)
        #whiten, mean-center
        naming_fun=self.get_naming_fun()
        ss = pd.DataFrame(StandardScaler().fit_transform(mf_subset), index=mf_subset.index,
                          columns=mf_subset.columns).rename_axis(naming_fun, 1)

        if featurewise:
            ss = ss.T

        rdc_obj = reducer(ss, **reducer_args)

        rdc_obj.means = means.rename_axis(naming_fun) #always the mean of input features... i.e. featurewise doesn't change this.

        return rdc_obj


