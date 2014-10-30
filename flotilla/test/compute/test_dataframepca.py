import numpy.testing as npt
import pandas.util.testing as pdt

from flotilla.data_model import ExpressionData
from flotilla.compute.decomposition import DataFramePCA


class TestDataFramePCA():
    def test_init(self, example_data):
        # TODO: parameterize and test with featurewise and subsets
        expression = ExpressionData(example_data.expression)
        test_reduced = expression.reduce()

        subset, means = expression._subset_and_standardize(
            expression.data, return_means=True)
        true_reduced = DataFramePCA(subset)
        true_reduced.means = means

        pdt.assert_frame_equal(test_reduced.X, subset)
        npt.assert_array_equal(test_reduced.reduced_space,
                               true_reduced.reduced_space)
        pdt.assert_series_equal(test_reduced.means,
                                true_reduced.means)