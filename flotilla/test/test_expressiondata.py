__author__ = 'olga'

from flotilla.data_model import ExpressionData
import pandas.util.testing as pdt

def test_init(example_data):
    expression_data = ExpressionData(example_data.phenotype_data,
                                     example_data.expression)
    pdt.assert_frame_equal(expression_data.phenotype_data,
                           example_data.phenotype_data)
    pdt.assert_frame_equal(expression_data.data, example_data.expression)


def test_go_enrichment():
    pass
