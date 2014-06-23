__author__ = 'olga'

import pandas.util.testing as pdt
import pytest

from flotilla.data_model import ExpressionData


@pytest.fixture
def expression(example_data):
    return ExpressionData(example_data.expression)


class TestExpressionData:
    def test_init(self, example_data):
        #TODO: parameterize and test with dropping outliers
        expression = ExpressionData(example_data.expression)
        pdt.assert_frame_equal(expression.data, example_data.expression)

        sparse_data = expression.data[
            expression.data > ExpressionData._expr_cut]
        pdt.assert_frame_equal(expression.sparse_data, sparse_data)


    def test_go_enrichment(self):
        pass

    def test__subset_and_standardize(self, expression):
        # TODO: parameterize and test with feature/sample subsets and
        # with/without standardization
        expression.subset, expression.means = \
            expression._subset_and_standardize(expression.sparse_data)

        columns = expression.sparse_data.count() > expression.min_samples
        subset = expression.sparse_data.ix[:, columns]
        means = subset.mean().rename_axis(expression.feature_renamer)
        subset = subset.fillna(means).fillna(0)
        subset = subset.rename_axis(expression.feature_renamer, 1)

        pdt.assert_frame_equal(subset, expression.subset)
        pdt.assert_series_equal(means, expression.means)


    def test_reduce(self, example_data):
        expression = ExpressionData(example_data.expression)
        expression_reduced = expression.reduce()

        data = example_data.expression.dropna(axis=0)
        means = data.fillna(data.mean())
        data = data.fillna()