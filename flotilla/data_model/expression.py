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

    def __init__(self, data, sample_metadata=None,
                 feature_metadata=None, expression_thresh=_expression_thresh,
                 feature_rename_col=None, outliers=None, log_base=None,
                 pooled=None,
                 technical_outliers=None, predictor_config_manager=None):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            Expression matrix. samples (rows) x features (columns
        feature_metadata : pandas.DataFrame


        Returns
        -------



        """
        sys.stderr.write("initializing expression\n")

        super(ExpressionData, self).__init__(
            data, feature_metadata=feature_metadata,
            sample_metadata=sample_metadata,
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
        self.feature_data = feature_metadata

        # This may not be totally kosher.... but original data is in
        # self.original_data
        if outliers is not None:
            self.outliers = self.outliers[
                self.outliers > self.expression_thresh]
        if pooled is not None:
            self.pooled = self.pooled[self.pooled > self.expression_thresh]
        self.default_feature_sets.extend(self.feature_subsets.keys())

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

