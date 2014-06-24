"""
ExpressionData
--------------
A container for gene expression data and related feature data, e.g. gene
symbols of ensembl IDs and GO terms.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BaseData
from ..visualize.decomposition import PCAViz
from ..visualize.predict import PredictorViz
from ..util import memoize


class ExpressionData(BaseData):
    _expression_thresh = 0.1


    def __init__(self, data,
                 feature_data=None, expression_thresh=_expression_thresh,
                 feature_rename_col=None, outliers=None):
        """
        Parameters
        ----------


        Returns
        -------



        Raises
        ------
        """

        super(ExpressionData, self).__init__(data, feature_data,
                                             feature_rename_col=feature_rename_col)

        self._var_cut = data.var().dropna().mean() + 2 * data.var() \
            .dropna().std()
        rpkm_variant = pd.Index(
            [i for i, j in (self.data.var().dropna() > self._var_cut)
            .iteritems()
             if j])
        self.feature_sets['variant'] = pd.Series(rpkm_variant,
                                                 index=rpkm_variant)


        # self.experiment_design_data, data = \
        #     self.experiment_design_data.align(data, join='inner', axis=0)

        self.feature_data = feature_data
        # self.data = data

        self.sparse_data = self.data[self.data > expression_thresh]

        self.feature_sets.update({'all_genes': pd.Series(
            self.data.columns.map(self.feature_renamer), index=self.data
            .columns)})
        self.default_feature_sets.extend(self.feature_sets.keys())

        # self._set_plot_colors()
        # self._set_plot_markers()
        # if load_cargo:
        #     self.load_cargo()

    def _subset_and_standardize(self, data, sample_ids=None,
                                feature_ids=None,
                                standardize=True):
        """Take only the sample ids and feature ids from this data, require
        at least some minimum samples, and standardize data using
        scikit-learn. Will also fill na values with the mean of the feature
        (column)

        Parameters
        ----------
        data : pandas.DataFrame
            The data you want to standardize
        sample_ids : None or list of strings
            If None, all sample ids will be used, else only the sample ids
            specified
        feature_ids : None or list of strings
            If None, all features will be used, else only the features
            specified
        standardize : bool
            Whether or not to "whiten" (make all variables uncorrelated) and
            mean-center via sklearn.preprocessing.StandardScaler

        Returns
        -------
        subset : pandas.DataFrame
            Subset of the dataframe with the requested samples and features,
            and standardized as described
        means : pandas.DataFrame
            Mean values of the features (columns). Ignores NAs.

        """
        if feature_ids is None:
            feature_ids = self.data.columns
        if sample_ids is None:
            sample_ids = self.data.index

        subset = data.ix[sample_ids]
        subset = subset.T.ix[feature_ids].T
        subset = subset.ix[:, subset.count() > self.min_samples]
        #fill na with mean for each event
        means = subset.mean().rename_axis(self.feature_renamer)
        subset = subset.fillna(means).fillna(0)
        subset = subset.rename_axis(self.feature_renamer, 1)

        # whiten, mean-center
        if standardize:
            data = StandardScaler().fit_transform(subset)
        else:
            data = subset

        # "data" is a matrix so need to transform it back into a convenient
        # dataframe
        subset = pd.DataFrame(data, index=subset.index,
                              columns=subset.columns)
        return subset, means

    @memoize
    def reduce(self, sample_ids=None, feature_ids=None,
               featurewise=False,
               reducer=PCAViz,
               standardize=True,
               title='',
               reducer_kwargs=None):
        """Make and memoize a reduced dimensionality representation of data

        Parameters
        ----------
        sample_ids : None or list of strings
            If None, all sample ids will be used, else only the sample ids
            specified
        feature_ids : None or list of strings
            If None, all features will be used, else only the features
            specified
        featurewise : bool
            Whether or not to use the features as the "samples", e.g. if you
            want to reduce the features in to "sample-space" instead of
            reducing the samples into "feature-space"
        standardize : bool
            Whether or not to "whiten" (make all variables uncorrelated) and
            mean-center via sklearn.preprocessing.StandardScaler
        title : str
            Title of the plot
        reducer_kwargs : dict
            Any additional arguments to send to the reducer

        Returns
        -------
        reducer_object : flotilla.compute.reduce.ReducerViz
            A ready-to-plot object containing the reduced space
        """

        # min_samples = self.min_samples
        # input_reducer_args = reducer_args.copy()
        # reducer_kwargs = self._default_reducer_kwargs.copy()
        # reducer_kwargs.update(input_reducer_args)
        reducer_kwargs = {} if reducer_kwargs is None else reducer_kwargs
        reducer_kwargs['title'] = title
        # feature_renamer = self.feature_renamer()

        subset, means = self._subset_and_standardize(self.sparse_data,
                                                     sample_ids, feature_ids,
                                                     standardize)

        # compute reduction
        if featurewise:
            subset = subset.T
        reducer_object = reducer(subset, **reducer_kwargs)
        reducer_object.means = means  #always the mean of input features... i
        # .e.
        # featurewise doesn't change this.

        #add mean gene_expression
        return reducer_object

    @memoize
    def classify(self, sample_ids=None, feature_ids=None,
                 categorical_trait=None,
                 standardize=True, predictor=PredictorViz):
        """Make and memoize a classifier on a categorical trait (associated
        with samples) subset of genes

        Parameters
        ----------
        sample_ids : None or list of strings
            If None, all sample ids will be used, else only the sample ids
            specified
        feature_ids : None or list of strings
            If None, all features will be used, else only the features
            specified
        categorical_trait : ???
            ???
        standardize : bool
            Whether or not to "whiten" (make all variables uncorrelated) and
            mean-center via sklearn.preprocessing.StandardScaler

        Returns
        -------
        classifier : flotilla.compute.predict.PredictorViz
            A ready-to-plot object containing the predictions
        """
        subset, means = self._subset_and_standardize(self.sparse_data,
                                                     sample_ids,
                                                     feature_ids,
                                                     standardize)

        classifier = predictor(subset,
                               categorical_traits=[categorical_trait], )
        classifier.set_reducer_plotting_args(self._default_reducer_kwargs)
        return classifier

    # def load_cargo(self, rename=True, **kwargs):
    #     try:
    #         species = self.species
    #         # self.cargo = cargo.get_species_cargo(self.species)
    #         self.go = self.cargo.get_go(species)
    #         self.feature_sets.update(self.cargo.gene_lists)
    #
    #         if rename:
    #             self._set_feature_renamer(lambda x: self.go.geneNames(x))
    #     except:
    #         raise

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
            df.rename_axis(self.feature_renamer(), 1)
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

    def __init__(self, data, feature_data):
        """Constructor for

        Parameters
        ----------
        data, experiment_design_data

        Returns
        -------


        Raises
        ------

        """
        super(ExpressionData, self).__init__(data, feature_data,
                                             feature_rename_col=feature_rename_col)


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