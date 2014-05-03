import pandas as pd
import seaborn
from sklearn.preprocessing import StandardScaler

from _Data import Data
from .._Submaraine_viz import PCA_viz
from .._Frigate_compute import dropna_mean
from .._Skiff_external_sources import link_to_list


seaborn.set_context('paper')


class ExpressionData(Data):
    _default_reducer_args = Data._default_reducer_args

    samplewise_reduction = {}
    featurewise_reduction = {}
    lists = {}
    var_cut=0.5
    expr_cut=0.1
    #=study.expression,
    #                                         sample_descriptors= study.sample_info,
    #                                         gene_descriptors=study.expression_info,
    def __init__(self, expression_df, sample_descriptors,
                 gene_descriptors = None,
                 var_cut=var_cut, expr_cut=expr_cut, load_cargo=True, rename=True,
                 ):
        self.sample_descriptors = sample_descriptors
        self.gene_descriptors = gene_descriptors
        self.expression_df = expression_df
        self.sparse_df = expression_df[expression_df > expr_cut]
        rpkm_variant = pd.Index([i for i, j in (expression_df.var().dropna() > var_cut).iteritems() if j])
        self.lists['variant'] = pd.Series(rpkm_variant, index=rpkm_variant)

        naming_fun = self.get_naming_fun()
        self.lists.update({'all_genes':pd.Series(map(naming_fun, self.expression_df.columns),
                                                           index = self.expression_df.columns)})
        self._default_reducer_args.update({'colors_dict':self.sample_descriptors.color})
        self._default_reducer_args.update({'markers_dict':self.sample_descriptors.marker})
        self._default_reducer_args.update({'show_vectors':False})


    def make_reduced(self, list_name, group_id, featurewise=False,
                    reducer=PCA_viz,
                    standardize=True,
                    **reducer_args):
        """make and cache a reduced dimensionality representation of data """

        min_samples=self.get_min_samples()
        input_reducer_args = reducer_args.copy()
        reducer_args = self._default_reducer_args.copy()
        reducer_args.update(input_reducer_args)
        reducer_args['title'] = list_name + " : " + group_id
        naming_fun = self.get_naming_fun()

        if list_name not in self.lists:
            this_list = link_to_list(list_name)
            self.lists[list_name] = pd.Series(map(naming_fun, this_list), index =this_list)


        gene_list = self.lists[list_name]
        if group_id.startswith("~"):
            #print 'not', group_id.lstrip("~")
            sample_ind = ~pd.Series(self.sample_descriptors[group_id.lstrip("~")], dtype='bool')
        else:
            sample_ind = pd.Series(self.sample_descriptors[group_id], dtype='bool')
        sample_ind = sample_ind[sample_ind].index
        subset = self.sparse_df.ix[sample_ind, gene_list.index]
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

        ss = pd.DataFrame(data, index = mf_subset.index,
                          columns = mf_subset.columns).rename_axis(naming_fun, 1)

        #compute pca
        if featurewise:
            ss = ss.T
        rdc_obj = reducer(ss, **reducer_args)
        rdc_obj.means = means.rename_axis(naming_fun) #always the mean of input features... i.e. featurewise doesn't change this.


        #add mean gene_expression
        return rdc_obj



