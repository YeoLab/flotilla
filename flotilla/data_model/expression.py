"""
Data types related to gene expression, e.g. from RNA-Seq or microarrays.
Included SpikeIn data.
"""
import sys

import numpy as np

from .base import BaseData
from ..util import memoize


class ExpressionData(BaseData):
    _expression_thresh = 0.1

    def __init__(self, data,
                 metadata=None, expression_thresh=_expression_thresh,
                 feature_rename_col=None, outliers=None, log_base=None,
                 pooled=None,
                 technical_outliers=None, predictor_config_manager=None):
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
            predictor_config_manager=predictor_config_manager,
            technical_outliers=technical_outliers)
        self.data_type = 'expression'
        self.expression_thresh = expression_thresh
        self.original_data = self.data
        self.data = self.data[self.data >= self.expression_thresh]

        sys.stderr.write("done initializing expression\n")

        self.log_base = log_base

        if self.log_base is not None:
            self.log_data = np.log(self.data + .1) / np.log(self.log_base)
        else:
            self.log_data = self.data

        self.data = self.log_data
        self.feature_data = metadata

        # This may not be totally kosher.... but original data is in
        # self.original_data
        if outliers is not None:
            self.outliers = self.outliers[
                self.outliers > self.expression_thresh]
        if pooled is not None:
            self.pooled = self.pooled[self.pooled > self.expression_thresh]
        self.default_feature_sets.extend(self.feature_subsets.keys())

    # def reduce(self, sample_ids=None, feature_ids=None,
    #            featurewise=False, reducer=PCAViz, standardize=True,
    #            title='', reducer_kwargs=None, color=None, groupby=None,
    #            label_to_color=None, label_to_marker=None,
    #            order=None, x_pc='pc_1', y_pc='pc_1'):
    #     """Make and memoize a reduced dimensionality representation of data
    #
    #     Parameters
    #     ----------
    #     sample_ids : None or list of strings
    #         If None, all sample ids will be used, else only the sample ids
    #         specified
    #     feature_ids : None or list of strings
    #         If None, all features will be used, else only the features
    #         specified
    #     featurewise : bool
    #         Whether or not to use the features as the "samples", e.g. if you
    #         want to reduce the features in to "sample-space" instead of
    #         reducing the samples into "feature-space"
    #     standardize : bool
    #         Whether or not to "whiten" (make all variables uncorrelated) and
    #         mean-center via sklearn.preprocessing.StandardScaler
    #     title : str
    #         Title of the plot
    #     reducer_kwargs : dict
    #         Any additional arguments to send to the reducer
    #
    #     Returns
    #     -------
    #     reducer_object : flotilla.compute.reduce.ReducerViz
    #         A ready-to-plot object containing the reduced space
    #     """
    #     return super(ExpressionData, self).reduce(
    #         self.data, sample_ids=sample_ids, feature_ids=feature_ids,
    #         featurewise=featurewise, reducer=reducer, standardize=standardize,
    #         title=title, reducer_kwargs=reducer_kwargs, groupby=groupby,
    #         label_to_color=label_to_color, label_to_marker=label_to_marker,
    #         order=order, color=color, x_pc=x_pc, y_pc=y_pc)

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

    @memoize
    def binify(self, data):
        data = self._subset(data, require_min_samples=False)
        data = (data - data.min()) / (data.max() - data.min())
        # vmax = data.abs().max().max()
        # vmin = -vmax
        # bins = np.linspace(vmin, vmax, 10)
        bins = np.arange(0, 1.1, .1)
        # print 'bins:', bins
        return super(ExpressionData, self).binify(data, bins)


class SpikeInData(ExpressionData):
    """Class for Spikein data and associated functions
    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, data, feature_data=None,
                 predictor_config_manager=None,
                 technical_outliers=None):
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
                                          technical_outliers=technical_outliers,
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
