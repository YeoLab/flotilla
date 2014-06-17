import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from _BaseData import BaseData
from ..visualize.decomposition import NMFViz, PCAViz
from ..visualize.predict import PredictorViz
from ..compute.generic import binify, dropna_mean
from ..external import link_to_list


class SplicingData(BaseData):
    binned_reducer = None
    raw_reducer = None

    n_components = 2
    _binsize=0.1
    _var_cut = 0.2


    def __init__(self, splicing, sample_metadata,
                 event_metadata, binsize=_binsize,
                 var_cut = _var_cut,
                 drop_outliers=True,
                 load_cargo=False,
                 **kwargs
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
        super(SplicingData, self).__init__(sample_metadata, **kwargs)
        if drop_outliers:
            splicing = self.drop_outliers(splicing)
        self.sample_metadata, splicing = self.sample_metadata.align(splicing, join='inner', axis=0)

        self.df = splicing
        self.binsize = binsize
        psi_variant = pd.Index([i for i,j in (splicing.var().dropna() > var_cut).iteritems() if j])
        self.set_naming_fun(self.namer)
        self.lists['variant'] = pd.Series(psi_variant, index=psi_variant)
        self.lists['all_genes'] =  pd.Series(splicing.index, index=splicing.index)
        self.event_metadata = event_metadata
        self._set_plot_colors()
        self._set_plot_markers()

    def namer(self, x):
        "this is for miso psi IDs..."
        shrt = ":".join(x.split("@")[1].split(":")[:2])
        try:
            dd = self.event_metadata.set_index('event_name')
            return dd['gene_symbol'].ix[x] + " " + shrt
        except Exception as e:
            #print e
            return shrt

    def set_binsize(self, binsize):
        self.binsize = binsize

    def get_binned_data(self):
        try:
            assert hasattr(self, 'binned') #binned has been set
            assert self._binsize == self.binsize #binsize hasn't changed
        except:
            #only bin once, until binsize is updated
            bins = np.arange(0, 1+self.binsize, self.binsize)
            self.binned = binify(self.df, bins)
            self._binsize = self.binsize
        return self.binned

    def get_binned_reduced(self, reducer=NMFViz):
        binned = self.get_binned_data()
        redc = reducer(binned)
        self.binned_reduced = redc.reduced_space
        return self.binned_reduced

    _last_reducer_accessed = None

    def make_reduced(self, list_name, group_id, reducer=PCAViz,
                    featurewise=False, reducer_args=None, standardize=True):
        """make and cache a reduced dimensionality representation of data """

        if reducer_args is None:
            reducer_args = self._default_reducer_args

        min_samples = self.get_min_samples()
        if list_name not in self.lists:
            self.lists[list_name] = link_to_list(list_name)

        event_list = self.lists[list_name]
        #some samples, somefeatures

        if group_id.startswith("~"):
            #print 'not', group_id.lstrip("~")
            sample_ind = ~pd.Series(self.sample_metadata[group_id.lstrip("~")], dtype='bool')
        else:
            sample_ind = pd.Series(self.sample_metadata[group_id], dtype='bool')

        subset = self.df.ix[sample_ind, event_list]
        frequent = pd.Index([i for i,j in (subset.count() > min_samples).iteritems() if j])
        subset = subset[frequent]
        #fill na with mean for each event
        means = subset.apply(dropna_mean, axis=0)
        mf_subset = subset.fillna(means,).fillna(0)
        #whiten, mean-center
        naming_fun=self.get_naming_fun()
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

    def make_classifier(self, list_name, group_id, categorical_trait,
                       standardize=True, classifier=PredictorViz,
                       ):
        """
        make and cache a classifier on a categorical trait (associated with samples) subset of genes
         """

        min_samples=self.get_min_samples()
        if list_name not in self.lists:
            self.lists[list_name] = link_to_list(list_name)

        event_list = self.lists[list_name]

        if group_id.startswith("~"):
            #print 'not', group_id.lstrip("~")
            sample_ind = ~pd.Series(self.sample_metadata[group_id.lstrip("~")], dtype='bool')
        else:
            sample_ind = pd.Series(self.sample_metadata[group_id], dtype='bool')
        sample_ind = sample_ind[sample_ind].index
        subset = self.df.ix[sample_ind, event_list]
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
        naming_fun = self.get_naming_fun()
        ss = pd.DataFrame(data, index = mf_subset.index,
                          columns = mf_subset.columns).rename_axis(naming_fun, 1)
        clf = classifier(ss, self.sample_metadata,
                        categorical_traits=[categorical_trait],)
        clf.set_reducer_plotting_args(self._default_reducer_args)
        return clf

    def load_cargo(self):
        raise NotImplementedError

    def _get(self, splicing_data_filename):
        return {'splicing_df': self.load(*splicing_data_filename)}




