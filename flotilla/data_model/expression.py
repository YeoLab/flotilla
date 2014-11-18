"""
Data types related to gene expression, e.g. from RNA-Seq or microarrays.
Included SpikeIn data.
"""
import sys

import numpy as np

from .base import BaseData
from ..util import memoize, timestamp

EXPRESSION_THRESH = -np.inf


class ExpressionData(BaseData):
    def __init__(self, data,
                 feature_data=None, thresh=EXPRESSION_THRESH,
                 feature_rename_col=None, feature_ignore_subset_cols=None,
                 outliers=None, log_base=None,
                 pooled=None, plus_one=False, minimum_samples=0,
                 technical_outliers=None, predictor_config_manager=None):
        """Object for holding and operating on expression data


        """
        sys.stdout.write("{}\tInitializing expression\n".format(timestamp()))

        super(ExpressionData, self).__init__(
            data, feature_data=feature_data,
            feature_rename_col=feature_rename_col,
            feature_ignore_subset_cols=feature_ignore_subset_cols,
            thresh=thresh,
            outliers=outliers, pooled=pooled, minimum_samples=minimum_samples,
            predictor_config_manager=predictor_config_manager,
            technical_outliers=technical_outliers)
        self.data_type = 'expression'
        self.thresh = thresh
        self.plus_one = plus_one

        if plus_one:
            self.data += 1
            self.thresh += 1
        # self.original_data = self.data
        # import pdb; pdb.set_trace()
        # self.data = self._threshold(data, thresh)
        self.log_base = log_base

        if self.log_base is not None:
            self.data = np.log(self.data) / np.log(self.log_base)

        self.feature_data = feature_data

        sys.stdout.write("{}\tDone initializing expression\n".format(
            timestamp()))

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

        # def plot_two_samples(self, sample1, sample2, **kwargs):
        # thresholded = kwargs.pop('thresholded', True)
        # super(ExpressionData, self).plot_two_samples(sample1, sample2,
        #                                                  thresholded=thresholded,
        #                                                  **kwargs)


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
        # import matplotlib.pyplot as plt
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
