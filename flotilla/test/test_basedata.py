__author__ = 'olga'

from ..data_model._BaseData import BaseData

def test_init(example_data):
    base_data = BaseData(example_data.metadata, example_data.expression)
    assert base_data.metadata == example_data.metadata
    assert base_data._data == example_data.expression