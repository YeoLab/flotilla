from ..schooner import Study
import pandas as pd

@pytest.fixture(scope='module')
def example_data():
    expression = pd.read_table('data/expression.tsv', index_col=0)
    splicing = pd.read_table('data/splicing.tsv', index_col=0)
    metadata = pd.read_table('data/metadata.tsv', index_col=0)
    return expression, splicing, metadata


def test_explicit_init(example_data):
    expression, splicing, metadata = example_data
    study = Study(metadata=metadata,
                  expression=expression,
                  splicing=splicing)

def test_args_init(example_data):
    study = Study(*example_data)
