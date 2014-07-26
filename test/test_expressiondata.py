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

    def test_reduce(self, example_data):
        #TODO: parameterize and test with featurewise and subsets
        expression = ExpressionData(example_data.expression)
        expression.reduced = expression.reduce()

        subset, means = expression._subset_and_standardize(
            expression.sparse_data, return_means=True)
        reducer_kwargs = {'title': ""}
        reduced = PCAViz(subset, **reducer_kwargs)
        reduced.means = means

        pdt.assert_frame_equal(expression.reduced.df, subset)
        npt.assert_array_equal(expression.reduced.reduced_space,
                               reduced.reduced_space)
        pdt.assert_series_equal(expression.reduced.means,
                                reduced.means)
