__author__ = 'olga'

from flotilla.data_model.base import BaseData
import pandas.util.testing as pdt

def test_init(example_data):
    base_data = BaseData(example_data.metadata, example_data.expression)
    pdt.assert_frame_equal(base_data.phenotype_data, example_data.metadata)
    pdt.assert_frame_equal(base_data.data, example_data.expression)