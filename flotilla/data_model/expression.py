"""
Data types related to gene expression, e.g. from RNA-Seq or microarrays.
"""
import sys

import numpy as np

from .base import BaseData
from ..util import timestamp

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
            technical_outliers=technical_outliers, data_type='expression')
        self.thresh_original = thresh
        self.plus_one = plus_one

        if plus_one:
            self.data += 1
            self.thresh = self.thresh_original + 1
        # self.original_data = self.data
        # import pdb; pdb.set_trace()
        # self.data = self._threshold(data, thresh)
        self.log_base = log_base

        if self.log_base is not None:
            self.data = np.divide(np.log(self.data), np.log(self.log_base))

        self.feature_data = feature_data

        sys.stdout.write("{}\tDone initializing expression\n".format(
            timestamp()))

    def _calculate_linkage(self, sample_ids, feature_ids, metric='euclidean',
                           linkage_method='average', standardize=True):
        return super(ExpressionData, self)._calculate_linkage(
            self.data, sample_ids=sample_ids, feature_ids=feature_ids,
            standardize=standardize, metric=metric,
            linkage_method=linkage_method)
