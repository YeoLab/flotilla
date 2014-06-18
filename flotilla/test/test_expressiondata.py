__author__ = 'olga'

from flotilla.data_model import ExpressionData
import pandas.util.testing as pdt

def test_init(example_data):
    expression_data = ExpressionData(example_data.metadata, example_data
                                 .expression)
    pdt.assert_frame_equal(expression_data.phenotype_data,
                           example_data.metadata)
    pdt.assert_frame_equal(expression_data.data, example_data.expression)