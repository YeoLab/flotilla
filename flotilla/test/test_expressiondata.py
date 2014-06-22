__author__ = 'olga'

import pandas.util.testing as pdt

from flotilla.data_model import ExpressionData


def test_init(example_data):
    expression_data = ExpressionData(example_data.expression)
    pdt.assert_frame_equal(expression_data.data, example_data.expression)


def test_go_enrichment():
    pass


def test_reduce(example_data):
    expression = ExpressionData(example_data.expression)
    expression_reduced = expression.reduce()

    data = example_data.expression.dropna(axis=0)
    means = data.fillna(data.mean())
    data = data.fillna()