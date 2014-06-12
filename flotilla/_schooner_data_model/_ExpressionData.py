import pandas as pd
import seaborn
from sklearn.preprocessing import StandardScaler

from _Data import Data, cargo
from .._submaraine_viz import PCA_viz, PredictorViz
from .._frigate_compute import dropna_mean, Predictor
from .._skiff_external_sources import link_to_list

seaborn.set_context('paper')


class ExpressionData(Data):


    _var_cut=0.5
    _expr_cut=0.1


    def __init__(self, expression_df, sample_metadata,
                 gene_metadata= None,
                 var_cut=_var_cut, expr_cut=_expr_cut,
                 drop_outliers=True, load_cargo=False,
                 **kwargs
                 ):

        super(ExpressionData, self).__init__(sample_metadata, **kwargs)
        if drop_outliers:
            expression_df = self.drop_outliers(expression_df)

        a = self.sample_metadata.align(expression_df, join='inner', axis=0)
        self.sample_metadata, expression_df = a

        self.gene_metadata = gene_metadata
        self.df = expression_df

        self.sparse_df = expression_df[expression_df > expr_cut]
        rpkm_variant = pd.Index([i for i, j in (expression_df.var().dropna() > var_cut).iteritems() if j])
        self.lists['variant'] = pd.Series(rpkm_variant, index=rpkm_variant)

        naming_fun = self.get_naming_fun()
        self.lists.update({'all_genes':pd.Series(map(naming_fun, self.df.columns),
                                                           index = self.df.columns)})
        self.set_reducer_colors()
        self.set_reducer_markers()
        if load_cargo:
            self.load_cargo()


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
            sample_ind = ~pd.Series(self.sample_metadata[group_id.lstrip("~")], dtype='bool')
        else:
            sample_ind = pd.Series(self.sample_metadata[group_id], dtype='bool')

        sample_ind = sample_ind[sample_ind].index
        subset = self.sparse_df.ix[sample_ind]
        subset = subset.T.ix[gene_list.index].T
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

    def make_classifier(self, gene_list_name, group_id, categorical_trait,
                       standardize=True, predictor=PredictorViz,
                       ):
        """
        make and cache a classifier on a categorical trait (associated with samples) subset of genes
         """

        min_samples=self.get_min_samples()
        naming_fun = self.get_naming_fun()

        if gene_list_name not in self.lists:
            this_list = link_to_list(gene_list_name)
            self.lists[gene_list_name] = pd.Series(map(naming_fun, this_list), index =this_list)

        gene_list = self.lists[gene_list_name]

        if group_id.startswith("~"):
            #print 'not', group_id.lstrip("~")
            sample_ind = ~pd.Series(self.sample_metadata[group_id.lstrip("~")], dtype='bool')
        else:
            sample_ind = pd.Series(self.sample_metadata[group_id], dtype='bool')
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
        clf = predictor(ss, self.sample_metadata,
                        categorical_traits=[categorical_trait],)
        clf.set_reducer_plotting_args(self._default_reducer_args)
        return clf

    def load_cargo(self, rename=True, **kwargs):
        try:
            species = self.species
            self.cargo = cargo.get_species_cargo(self.species)
            self.go = self.cargo.get_go(species)
            self.lists.update(self.cargo.gene_lists)

            if rename:
                self.set_naming_fun(lambda x: self.go.geneNames(x))
        except:
            raise


