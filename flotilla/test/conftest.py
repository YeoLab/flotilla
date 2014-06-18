__author__ = 'olga'

"""
This file will be auto-imported for every testing session, so you can use
these objects and functions across test files.
"""

import pytest
import pandas as pd

class ExampleData(object):
    __slots__ = ('metadata', 'expression', 'splicing', 'data')
    def __init__(self, sample_metadata, expression, splicing):
        self.sample_metadata = sample_metadata
        self.expression = expression
        self.splicing = splicing
        self.data = (sample_metadata, expression, splicing)


@pytest.fixture(scope='module')
def example_data():
    expression = pd.read_table('data/expression.tsv', index_col=0)
    splicing = pd.read_table('data/splicing.tsv', index_col=0)
    metadata = pd.read_table('data/metadata.tsv', index_col=0)
    return ExampleData(metadata, expression, splicing)