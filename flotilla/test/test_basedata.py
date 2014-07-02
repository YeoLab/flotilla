from flotilla.data_model.base import BaseData
import pandas.util.testing as pdt

def test_basedata_init(example_data):
    base_data = BaseData(example_data.expression)
    pdt.assert_frame_equal(base_data.data, example_data.expression)