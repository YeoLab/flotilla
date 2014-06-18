"""
This tests whether the SplicingData object was created correctly. No
computation or visualization tests yet.
"""

from flotilla.data_model import SplicingData
import pandas.util.testing as pdt

def test_init(example_data):
    splicing_data = SplicingData(example_data.phenotype_data,
                                 example_data.splicing)
    pdt.assert_frame_equal(splicing_data.phenotype_data,
                           example_data.phenotype_data)
    pdt.assert_frame_equal(splicing_data.data, example_data.expression)
