"""
Data types related to gene expression, e.g. from RNA-Seq or microarrays.
Included SpikeIn data.
"""
import sys

import numpy as np

from .base import BaseData
from ..visualize.decomposition import PCAViz
from ..visualize.predict import ClassifierViz
from ..util import memoize


class ExpressionData(BaseData):
    _expression_thresh = 0.1

    def __init__(self, data,
                 metadata=None, expression_thresh=_expression_thresh,
                 feature_rename_col=None, outliers=None, log_base=None,
                 pooled=None, predictor_config_manager=None):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            Expression matrix. samples (rows) x features (columns
        metadata : pandas.DataFrame


        Returns
        -------



        """
        sys.stderr.write("initializing expression\n")

        super(ExpressionData, self).__init__(
            data, metadata,
            feature_rename_col=feature_rename_col,
            outliers=outliers, pooled=pooled,
            predictor_config_manager=predictor_config_manager)
        self.data_type = 'expression'

        sys.stderr.write("done initializing expression\n")

        if log_base is not None:
            self.log_data = np.log(self.data + .1) / np.log(log_base)
        else:
            self.log_data = self.data

        self.original_data = self.data
        self.feature_data = metadata

        # This may not be totally kosher.... but original data is in
        # self.original_data
        self.expression_thresh = expression_thresh
        self.data = self.log_data[self.log_data > self.expression_thresh]
        self.default_feature_sets.extend(self.feature_subsets.keys())

    def reduce(self, sample_ids=None, feature_ids=None,
               featurewise=False,
               reducer=PCAViz,
               standardize=True,
               title='',
               reducer_kwargs=None,
               color=None,
               groupby=None, label_to_color=None, label_to_marker=None,
               order=None, x_pc='pc_1', y_pc='pc_1'):
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
        return super(ExpressionData, self).reduce(
            self.data, sample_ids=sample_ids, feature_ids=feature_ids,
            featurewise=featurewise, reducer=reducer, standardize=standardize,
            title=title, reducer_kwargs=reducer_kwargs, groupby=groupby,
            label_to_color=label_to_color, label_to_marker=label_to_marker,
            order=order, color=color, x_pc=x_pc, y_pc=y_pc)

    @memoize
    def classify(self, trait, sample_ids, feature_ids,
                 standardize=True,
                 data_name='expression',
                 predictor_name='ExtraTreesClassifier',
                 predictor_obj=None,
                 predictor_scoring_fun=None,
                 score_cutoff_fun=None,
                 n_features_dependent_parameters=None,
                 constant_parameters=None,
                 plotting_kwargs=None):
        #Should all this be exposed to the user???

        """Make and memoize a predictor on a categorical trait (associated
        with samples) subset of genes

        Parameters
        ----------
        trait : pandas.Series
            samples x categorical feature
        sample_ids : None or list of strings
            If None, all sample ids will be used, else only the sample ids
            specified
        feature_ids : None or list of strings
            If None, all features will be used, else only the features
            specified
        standardize : bool
            Whether or not to "whiten" (make all variables uncorrelated) and
            mean-center and make unit-variance all the data via sklearn
            .preprocessing.StandardScaler
        predictor : flotilla.visualize.predict classifier
            Must inherit from flotilla.visualize.PredictorBaseViz. Default is
            flotilla.visualize.predict.ClassifierViz
        predictor_kwargs : dict or None
            Additional 'keyword arguments' to supply to the predictor class
        predictor_scoring_fun : function
            Function to get the feature scores for a scikit-learn classifier.
            This can be different for different classifiers, e.g. for a
            classifier named "x" it could be x.scores_, for other it's
            x.feature_importances_. Default: lambda x: x.feature_importances_
        score_cutoff_fun : function
            Function to cut off insignificant scores
            Default: lambda scores: np.mean(x) + 2 * np.std(x)

        Returns
        -------
        predictor : flotilla.compute.predict.PredictorBaseViz
            A ready-to-plot object containing the predictions
        """
        subset = self._subset_and_standardize(self.log_data,
                                              sample_ids,
                                              feature_ids,
                                              standardize)
        if plotting_kwargs is None:
            plotting_kwargs = {}

        classifier = ClassifierViz(data_name, trait.name,
                                   predictor_name=predictor_name,
                                   X_data=subset,
                                   trait=trait,
                                   predictor_obj=predictor_obj,
                                   predictor_scoring_fun=predictor_scoring_fun,
                                   score_cutoff_fun=score_cutoff_fun,
                                   n_features_dependent_parameters=n_features_dependent_parameters,
                                   constant_parameters=constant_parameters,
                                   predictor_dataset_manager=self.predictor_dataset_manager,
                                   **plotting_kwargs)
        return classifier

    # def load_cargo(self, rename=True, **kwargs):
    #     try:
    #         species = self.species
    #         # self.cargo = cargo.get_species_cargo(self.species)
    #         self.go = self.cargo.get_go(species)
    #         self.feature_subsets.update(self.cargo.gene_lists)
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

    def _calculate_linkage(self, sample_ids, feature_ids, metric='euclidean',
                           linkage_method='average', standardize=True):
        return super(ExpressionData, self)._calculate_linkage(
            self.data, sample_ids=sample_ids, feature_ids=feature_ids,
            standardize=standardize, metric=metric,
            linkage_method=linkage_method)


class SpikeInData(ExpressionData):
    """Class for Spikein data and associated functions
    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, data, feature_data=None, predictor_config_manager=None):
        """Constructor for

        Parameters
        ----------
        data, experiment_design_data

        Returns
        -------


        Raises
        ------

        """
        super(SpikeInData, self).__init__(data, feature_data,
                                          predictor_config_manager=predictor_config_manager)

        # def spikeins_violinplot(self):
        #     import matplotlib.pyplot as plt
        #     import seaborn as sns
        #     import numpy as np
        #
        #     fig, axes = plt.subplots(nrows=5, figsize=(16, 20), sharex=True,
        #                              sharey=True)
        #     ercc_concentrations = \
        #         ercc_controls_analysis.mix1_molecules_per_ul.copy()
        #     ercc_concentrations.sort()
        #
        #     for ax, (celltype, celltype_df) in \
        #             zip(axes.flat, tpm.ix[spikeins].groupby(
        #                     sample_id_to_celltype_, axis=1)):
        #         print celltype
        #         #     fig, ax = plt.subplots(figsize=(16, 4))
        #         x_so_far = 0
        #         #     ax.set_yscale('log')
        #         xticklabels = []
        #         for spikein_type, spikein_df in celltype_df.groupby(
        #                 spikein_to_type):
        #             #         print spikein_df.shape
        #             df = spikein_df.T + np.random.uniform(0, 0.01,
        #                                                   size=spikein_df.T.shape)
        #             df = np.log2(df)
        #             if spikein_type == 'ERCC':
        #                 df = df[ercc_concentrations.index]
        #             xticklabels.extend(df.columns.tolist())
        #             color = 'husl' if spikein_type == 'ERCC' else 'Greys_d'
        #             sns.violinplot(df, ax=ax,
        #                            positions=np.arange(df.shape[1])+x_so_far,
        #                            linewidth=0, inner='none', color=color)
        #
        #             x_so_far += df.shape[1]
        #
        #         ax.set_title(celltype)
        #         ax.set_xticks(np.arange(x_so_far))
        #         ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
        #         ax.set_ylabel('$\\log_2$ TPM')
        #
        #         xmin, xmax = -0.5, x_so_far - 0.5
        #
        #         ax.hlines(0, xmin, xmax)
        #         ax.set_xlim(xmin, xmax)
        #         sns.despine()
        #
        #     def samples_violinplot():
        #         fig, axes = plt.subplots(nrows=3, figsize=(16, 6))
        #
        #         for ax, (spikein_type, df) in zip(axes,
        #                                           tpm.groupby(spikein_to_type,
        #                                                       axis=0)):
        #             print spikein_type, df.shape
        #             if df.shape[0] > 1:
        #                 sns.violinplot(np.log2(df + 1), ax=ax, linewidth=0.1)
        #                 ax.set_xticks([])
        #                 ax.set_xlabel('')
        #
        #             else:
        #                 x = np.arange(df.shape[1])
        #                 ax.bar(np.arange(df.shape[1]),
        #                        np.log2(df.ix[spikein_type]),
        #                        color=green)
        #                 ax.set_xticks(x + 0.4)
        #                 ax.set_xticklabels(df.columns, rotation=60)
        #                 sns.despine()
        #
        #             ax.set_title(spikein_type)
        #             ax.set_xlim(0, tpm.shape[1])
        #             ax.set_ylabel('$\\log_2$ TPM')
        #         sns.despine()
