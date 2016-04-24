from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import pytest
from sklearn.decomposition import PCA


@pytest.fixture(params=[None, 2])
def n_components(request):
    return request.param


class TestDataFramePCA():

    def test_init(self, df_norm, n_components):

        from flotilla.compute.decomposition import DataFramePCA

        test_pca = DataFramePCA(df_norm, n_components=n_components)

        true_pca = PCA(n_components=n_components)
        true_pca.fit(df_norm.values)
        pc_names = ['pc_{}'.format(i + 1) for i in
                    range(true_pca.components_.shape[0])]
        true_pca.components_ = pd.DataFrame(true_pca.components_,
                                            index=pc_names,
                                            columns=df_norm.columns)
        true_pca.explained_variance_ = pd.Series(
            true_pca.explained_variance_, index=pc_names)
        true_pca.explained_variance_ratio_ = pd.Series(
            true_pca.explained_variance_ratio_, index=pc_names)
        true_pca.reduced_space = true_pca.transform(df_norm.values)
        true_pca.reduced_space = pd.DataFrame(true_pca.reduced_space,
                                              index=df_norm.index,
                                              columns=pc_names)

        npt.assert_array_equal(test_pca.X, df_norm.values)
        pdt.assert_frame_equal(test_pca.components_,
                               true_pca.components_)
        pdt.assert_series_equal(test_pca.explained_variance_,
                                true_pca.explained_variance_)
        pdt.assert_series_equal(test_pca.explained_variance_ratio_,
                                true_pca.explained_variance_ratio_)
        pdt.assert_frame_equal(test_pca.reduced_space,
                               true_pca.reduced_space)


# class TestDataFrameICA():
#     pass
#
#
# class TestDataFrameTSNE():
#     pass
