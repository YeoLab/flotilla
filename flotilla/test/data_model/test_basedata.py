import numpy.testing as npt
import pandas.util.testing as pdt


class TestBaseData:
    def test_basedata_init(self, shalek2013_data):
        from flotilla.data_model.base import BaseData
        base_data = BaseData(shalek2013_data.expression)
        pdt.assert_frame_equal(base_data.data, shalek2013_data.expression)

    def test_subset(self, base_data, sample_ids, feature_ids):
        import pandas as pd

        subset = base_data._subset(base_data.data, sample_ids=sample_ids,
                                   feature_ids=feature_ids)

        data = base_data.data
        if feature_ids is None:
            feature_ids = data.columns
        else:
            feature_ids = pd.Index(set(feature_ids).intersection(data.columns))
        if sample_ids is None:
            sample_ids = data.index
        else:
            sample_ids = pd.Index(set(sample_ids).intersection(data.index))

        true_subset = data.ix[sample_ids, feature_ids]

        pdt.assert_frame_equal(subset, true_subset)

    def test__subset_and_standardize(self, base_data, standardize, feature_ids,
                                     sample_ids):
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        base_data.subset, base_data.means = \
            base_data._subset_and_standardize(base_data.data,
                                              sample_ids=sample_ids,
                                              feature_ids=feature_ids,
                                              return_means=True,
                                              standardize=standardize)

        subset = base_data._subset(base_data.data, sample_ids=sample_ids,
                                   feature_ids=feature_ids)
        means = subset.mean().rename_axis(base_data.feature_renamer)
        subset = subset.fillna(means).fillna(0)
        subset = subset.rename_axis(base_data.feature_renamer, 1)

        if standardize:
            data = StandardScaler().fit_transform(subset)
        else:
            data = subset

        subset_standardized = pd.DataFrame(data, index=subset.index,
                                           columns=subset.columns)

        pdt.assert_frame_equal(subset_standardized, base_data.subset)
        pdt.assert_series_equal(means, base_data.means)


    def test_reduce(self, shalek2013_data, featurewise):
        # TODO: parameterize and test with featurewise and subsets
        from flotilla.compute.decomposition import DataFramePCA
        from flotilla.data_model.base import BaseData
        expression = BaseData(shalek2013_data.expression)
        test_reduced = expression.reduce(featurewise=featurewise)

        subset, means = expression._subset_and_standardize(
            expression.data, return_means=True, standardize=True)

        if featurewise:
            subset = subset.T

        true_reduced = DataFramePCA(subset)
        true_reduced.means = means

        pdt.assert_frame_equal(test_reduced.X, subset)
        npt.assert_array_equal(test_reduced.reduced_space,
                               true_reduced.reduced_space)
        pdt.assert_series_equal(test_reduced.means,
                                true_reduced.means)
