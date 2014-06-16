__author__ = 'olga'

import pytest
import pandas as pd

class ExampleData(object):
    __slots__ = ('metadata', 'expression', 'splicing', 'data')
    def __init__(self, metadata, expression, splicing):
        self.metadata = metadata
        self.expression = expression
        self.splicing = splicing
        self.data = (metadata, expression, splicing)


@pytest.fixture(scope='module')
def example_data():
    expression = pd.read_table('data/expression.tsv', index_col=0)
    splicing = pd.read_table('data/splicing.tsv', index_col=0)
    metadata = pd.read_table('data/metadata.tsv', index_col=0)
    return ExampleData(metadata, expression, splicing)