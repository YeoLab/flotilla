from _Data import Data
import scipy
from scipy import sparse
import pandas as pd
from collections import defaultdict

import seaborn
from sklearn.preprocessing import StandardScaler

from flotilla.src.submarine import PCA_viz

from ..frigate import dropna_mean
from ..skiff import link_to_list
from ...project.project_params import min_cells, _default_group_id, _default_group_ids

seaborn.set_context('paper')


class ExpressionData(Data):
    _default_reducer_args = Data._default_reducer_args
    _default_group_id = _default_group_id
    _default_group_ids = _default_group_ids
    samplewise_reduction = {}
    featurewise_reduction = {}
    lists = {}
    var_cut=0.5
    expr_cut=0.1
    def __init__(self, rpkm, sample_descriptors,
                 gene_descriptors = None,
                 var_cut=var_cut, expr_cut=expr_cut, load_cargo=True, rename=True,
    ):

        self.rpkm = rpkm
        self.sparse_rpkm = rpkm[rpkm > expr_cut]
        rpkm_variant = pd.Index([i for i, j in (rpkm.var().dropna() > var_cut).iteritems() if j])
        self.lists['variant'] = pd.Series(rpkm_variant, index=rpkm_variant)


        self.sample_descriptors = sample_descriptors
        self.gene_descriptors = gene_descriptors
        if load_cargo:
            from ..cargo import gene_lists, go
            self.lists.update(gene_lists)
            self._default_list = 'confident_rbps'
            if rename:
                self.set_naming_fun(lambda x: go.geneNames(x))
        naming_fun = self.get_naming_fun()
        self.lists.update({'all_genes':pd.Series(map(naming_fun, self.rpkm.columns),
                                                           index = self.rpkm.columns)})
        self._default_reducer_args.update({'colors_dict':self.sample_descriptors.color})
        self._default_reducer_args.update({'markers_dict':self.sample_descriptors.marker})
        self._default_reducer_args.update({'show_vectors':False})

    def make_reduced(self, list_name, group_id, featurewise=False,
                    min_cells=min_cells,
                    reducer=PCA_viz,
                    standardize=True,
                    **reducer_args):

        input_reducer_args = reducer_args.copy()
        reducer_args = self._default_reducer_args.copy()
        reducer_args.update(input_reducer_args)
        reducer_args['title'] = list_name + " : " + group_id
        if list_name not in self.lists:
            self.lists[list_name] = link_to_list(list_name)

        gene_list = self.lists[list_name]
        if group_id.startswith("~"):
            #print 'not', group_id.lstrip("~")
            sample_ind = ~pd.Series(self.sample_descriptors[group_id.lstrip("~")], dtype='bool')
        else:
            sample_ind = pd.Series(self.sample_descriptors[group_id], dtype='bool')
        sample_ind = sample_ind[sample_ind].index
        subset = self.sparse_rpkm.ix[sample_ind, gene_list.index]
        frequent = pd.Index([i for i, j in (subset.count() > min_cells).iteritems() if j])
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

        #compute pca
        if featurewise:
            ss = ss.T
        rdc_obj = reducer(ss, **reducer_args)
        rdc_obj.means = means.rename_axis(naming_fun) #always the mean of input features... i.e. featurewise doesn't change this.


        #add mean gene_expression
        return rdc_obj



