"""
This tests whether the SplicingData object was created correctly. No
computation or visualization tests yet.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pytest


@pytest.fixture
def n():
    return 10


class TestSplicingData:

    @pytest.fixture(params=['groupby_real', 'groupby_none'])
    def groupby_params(self, request, groupby):
        if request.param == 'groupby_real':
            return groupby
        elif request.param == 'groupby_none':
            return None

    @pytest.fixture(params=[True, False])
    def percentages(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def rename(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def return_means(self, request):
        return request.param

    def test__subset_and_standardize(self, splicing):
        test_subset = splicing._subset_and_standardize(splicing.data)

        true_subset = splicing._subset(splicing.data)
        true_subset = true_subset.dropna(how='all', axis=1).dropna(how='all',
                                                                   axis=0)

        true_subset = true_subset.fillna(true_subset.mean())
        true_subset = -2 * np.arccos(true_subset*2-1) + np.pi

        pdt.assert_frame_equal(test_subset, true_subset)

    def test__subset_and_standardize_rename_means(self, splicing,
                                                  rename):
        test_subset, test_means = splicing._subset_and_standardize(
            splicing.data, return_means=True, rename=rename)

        true_subset = splicing._subset(splicing.data)
        true_subset = true_subset.dropna(how='all', axis=1).dropna(how='all',
                                                                   axis=0)

        true_subset = true_subset.fillna(true_subset.mean())
        true_subset = -2 * np.arccos(true_subset*2-1) + np.pi

        true_means = true_subset.mean()

        if rename:
            true_means = true_means.rename_axis(splicing.feature_renamer)
            true_subset = true_subset.rename_axis(
                splicing.feature_renamer, 1)

        pdt.assert_frame_equal(test_subset, true_subset)
        pdt.assert_series_equal(test_means, true_means)

    def test_plot_feature(self, splicing):
        splicing.plot_feature(splicing.data.columns[0])
        plt.close('all')

    def test_plot_lavalamp(self, splicing, group_to_color):
        splicing.plot_lavalamp(group_to_color)
        plt.close('all')

    def test_plot_two_features(self, splicing, groupby,
                               group_to_color):
        ind = splicing.data.count() > 10

        features = splicing.data.columns[ind]
        feature1 = features[0]
        feature2 = features[1]
        splicing.plot_two_features(feature1, feature2, groupby=groupby,
                                   label_to_color=group_to_color)
        plt.close('all')

    def test_plot_two_samples(self, splicing):
        samples = splicing.data.index[splicing.data.T.count() > 10]
        sample1 = samples[0]
        sample2 = samples[1]
        splicing.plot_two_samples(sample1, sample2)
        plt.close('all')
