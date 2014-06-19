"""
ExpressionData
--------------
A container for gene expression data and related feature data, e.g. gene
symbols of ensembl IDs and GO terms.
"""

import pandas as pd
import seaborn
from sklearn.preprocessing import StandardScaler

from .base import BaseData
from ..visualize.decomposition import PCAViz
from ..visualize.predict import PredictorViz
from ..compute.generic import dropna_mean
from ..compute.predict import Predictor
from ..external import link_to_list
from ..util import memoize

seaborn.set_context('paper')


class ExpressionData(BaseData):

    _var_cut = 0.5
    _expr_cut = 0.1



    def __init__(self, data,
                 feature_data= None,
                 var_cut=_var_cut, expr_cut=_expr_cut,
                 drop_outliers=True, load_cargo=False,
                 **kwargs):

        super(ExpressionData, self).__init__(data)
        if drop_outliers:
            self.data = self.drop_outliers(data)

        # self.phenotype_data, data = \
        #     self.phenotype_data.align(data, join='inner', axis=0)

        self.feature_data = feature_data
        # self.data = data

        self.sparse_data = data[data > expr_cut]
        rpkm_variant = pd.Index([i for i, j in (data.var().dropna() > var_cut).iteritems() if j])
        self.feature_sets['variant'] = pd.Series(rpkm_variant, index=rpkm_variant)

        feature_renamer = self.get_feature_renamer()
        self.feature_sets.update({'all_genes': pd.Series(
            self.data.columns.map(feature_renamer), index=self.data.columns)})
        self._set_plot_colors()
        self._set_plot_markers()
        if load_cargo:
            self.load_cargo()

    @memoize
    def reduce(self, list_name, group_id, featurewise=False,
                    reducer=PCAViz,
                    standardize=True,
                    **reducer_args):
        """make and cache a reduced dimensionality representation of data """

        min_samples=self.get_min_samples()
        input_reducer_args = reducer_args.copy()
        reducer_args = self._default_reducer_kwargs.copy()
        reducer_args.update(input_reducer_args)
        reducer_args['title'] = list_name + " : " + group_id
        feature_renamer = self.get_feature_renamer()

        if list_name not in self.feature_sets:
            this_list = link_to_list(list_name)
            self.feature_sets[list_name] = pd.Series(map(feature_renamer, this_list),
                                              index=this_list)

        gene_list = self.feature_sets[list_name]

        if group_id.startswith("~"):
            #print 'not', group_id.lstrip("~")
            sample_ind = ~pd.Series(self.phenotype_data[group_id.lstrip("~")],
                                    dtype='bool')
        else:
            sample_ind = pd.Series(self.phenotype_data[group_id], dtype='bool')

        sample_ind = sample_ind[sample_ind].index
        subset = self.sparse_data.ix[sample_ind]
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

        ss = pd.DataFrame(data, index=mf_subset.index,
                          columns=mf_subset.columns).rename_axis(feature_renamer, 1)

        #compute pca
        if featurewise:
            ss = ss.T
        rdc_obj = reducer(ss, **reducer_args)
        rdc_obj.means = means.rename_axis(feature_renamer) #always the mean of input features... i.e. featurewise doesn't change this.

        #add mean gene_expression
        return rdc_obj

    @memoize
    def classify(self, gene_list_name, group_id, categorical_trait,
                       standardize=True, predictor=PredictorViz):
        """
        make and cache a classifier on a categorical trait (associated with samples) subset of genes
         """

        min_samples=self.get_min_samples()
        feature_renamer = self.get_feature_renamer()

        if gene_list_name not in self.feature_sets:
            this_list = link_to_list(gene_list_name)
            self.feature_sets[gene_list_name] = pd.Series(map(feature_renamer, this_list), index =this_list)

        gene_list = self.feature_sets[gene_list_name]

        if group_id.startswith("~"):
            #print 'not', group_id.lstrip("~")
            sample_ind = ~pd.Series(self.phenotype_data[group_id.lstrip("~")], dtype='bool')
        else:
            sample_ind = pd.Series(self.phenotype_data[group_id], dtype='bool')
        sample_ind = sample_ind[sample_ind].index
        subset = self.sparse_data.ix[sample_ind, gene_list.index]
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
                          columns = mf_subset.columns).rename_axis(feature_renamer, 1)
        clf = predictor(ss, self.phenotype_data,
                        categorical_traits=[categorical_trait],)
        clf.set_reducer_plotting_args(self._default_reducer_kwargs)
        return clf

    def load_cargo(self, rename=True, **kwargs):
        try:
            species = self.species
            # self.cargo = cargo.get_species_cargo(self.species)
            self.go = self.cargo.get_go(species)
            self.feature_sets.update(self.cargo.gene_lists)

            if rename:
                self._set_feature_renamer(lambda x: self.go.geneNames(x))
        except:
            raise

    #
    # def _get(self, expression_data_filename):
    #     return {'expression_df': self.load(*expression_data_filename)}

    def twoway(self, sample1, sample2, **kwargs):
        from ..visualize.expression import TwoWayScatterViz

        pCut = kwargs['p_value_cutoff']
        this_name = "_".join([sample1, sample2, str(pCut)])
        if this_name in self.localZ_dict:
            vz = self.localZ_dict[this_name]
        else:
            df = self.data
            df.rename_axis(self.get_feature_renamer(), 1)
            vz = TwoWayScatterViz(sample1, sample2, df, **kwargs)
            self.localZ_dict[this_name] = vz

        return vz

    def plot_twoway(self, sample1, sample2, **kwargs):
        vz = self.twoway(sample1, sample2, **kwargs)
        vz()
        return vz


class SpikeInData(ExpressionData):
    """Class for Spikein data and associated functions
    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, df, phenotype_data):
        """Constructor for

        Parameters
        ----------
        data, phenotype_data

        Returns
        -------


        Raises
        ------

        """
        pass

    def spikeins_violinplot(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        fig, axes = plt.subplots(nrows=5, figsize=(16, 20), sharex=True,
                                 sharey=True)
        ercc_concentrations = ercc_controls_analysis.mix1_molecules_per_ul.copy()
        ercc_concentrations.sort()

        for ax, (celltype, celltype_df) in zip(axes.flat,
                                               tpm.ix[spikeins].groupby(
                                                       sample_id_to_celltype_,
                                                       axis=1)):
            print celltype
            #     fig, ax = plt.subplots(figsize=(16, 4))
            x_so_far = 0
            #     ax.set_yscale('log')
            xticklabels = []
            for spikein_type, spikein_df in celltype_df.groupby(
                    spikein_to_type):
                #         print spikein_df.shape
                df = spikein_df.T + np.random.uniform(0, 0.01,
                                                      size=spikein_df.T.shape)
                df = np.log2(df)
                if spikein_type == 'ERCC':
                    df = df[ercc_concentrations.index]
                xticklabels.extend(df.columns.tolist())
                color = 'husl' if spikein_type == 'ERCC' else 'Greys_d'
                sns.violinplot(df, ax=ax,
                               positions=np.arange(df.shape[1]) + x_so_far,
                               linewidth=0, inner='none', color=color)

                x_so_far += df.shape[1]

            ax.set_title(celltype)
            ax.set_xticks(np.arange(x_so_far))
            ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
            ax.set_ylabel('$\\log_2$ TPM')

            xmin, xmax = -0.5, x_so_far - 0.5

            ax.hlines(0, xmin, xmax)
            ax.set_xlim(xmin, xmax)
            sns.despine()
            # fig.savefig('/projects/ps-yeolab/obotvinnik/mn_diff_singlecell/figures/spikeins.pdf')
            # ! cp /projects/ps-yeolab/obotvinnik/mn_diff_singlecell/figures/spikeins.pdf ~/Dropbox/figures2/singlecell/spikeins.pdf

        def samples_violinplot():
            fig, axes = plt.subplots(nrows=3, figsize=(16, 6))

            for ax, (spikein_type, df) in zip(axes, tpm.groupby(spikein_to_type,
                                                                axis=0)):
                print spikein_type, df.shape

                if df.shape[0] > 1:
                    sns.violinplot(np.log2(df + 1), ax=ax, linewidth=0.1)
                    ax.set_xticks([])
                    ax.set_xlabel('')

                else:
                    x = np.arange(df.shape[1])
                    ax.bar(np.arange(df.shape[1]), np.log2(df.ix[spikein_type]),
                           color=green)
                    ax.set_xticks(x + 0.4)
                    ax.set_xticklabels(df.columns, rotation=60)
                    sns.despine()

                ax.set_title(spikein_type)
                ax.set_xlim(0, tpm.shape[1])
                ax.set_ylabel('$\\log_2$ TPM')
            sns.despine()