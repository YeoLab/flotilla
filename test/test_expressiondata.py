import numpy.testing as npt
import pandas.util.testing as pdt

from flotilla.data_model import ExpressionData
from flotilla.visualize.decomposition import PCAViz


class TestExpressionData:
    def test_init(self, example_data):
        #TODO: parameterize and test with dropping outliers
        expression = ExpressionData(example_data.expression)
        pdt.assert_frame_equal(expression.data, example_data.expression)

        sparse_data = expression.data[
            expression.data > ExpressionData._expression_thresh]
        pdt.assert_frame_equal(expression.sparse_data, sparse_data)


    def test_subset(self):
        pass

    def test__subset_and_standardize(self, expression):
        # TODO: parameterize and test with feature/sample subsets and
        # with/without standardization
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        expression.subset, expression.means = \
            expression._subset_and_standardize(expression.sparse_data)

        columns = expression.sparse_data.count() > expression.min_samples
        subset = expression.sparse_data.ix[:, columns]
        means = subset.mean().rename_axis(expression.feature_renamer)
        subset = subset.fillna(means).fillna(0)
        subset = subset.rename_axis(expression.feature_renamer, 1)

        data = StandardScaler().fit_transform(subset)

        subset_standardized = pd.DataFrame(data, index=subset.index,
                                           columns=subset.columns)

        pdt.assert_frame_equal(subset_standardized, expression.subset)
        pdt.assert_series_equal(means, expression.means)


    def test_reduce(self, example_data):
        #TODO: parameterize and test with featurewise and subsets
        expression = ExpressionData(example_data.expression)
        expression.reduced = expression.reduce()

        subset, means = expression._subset_and_standardize(
            expression.sparse_data)
        reducer_kwargs = {'title': ""}
        reduced = PCAViz(subset, **reducer_kwargs)
        reduced.means = means

        pdt.assert_frame_equal(expression.reduced.df, subset)
        npt.assert_array_equal(expression.reduced.reduced_space,
                               reduced.reduced_space)
        pdt.assert_series_equal(expression.reduced.means,
                                reduced.means)
