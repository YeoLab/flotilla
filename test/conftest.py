"""
This file will be auto-imported for every testing session, so you can use
these objects and functions across test files.
"""
import os
import subprocess

import matplotlib as mpl
import pytest
import pandas as pd



# Tell matplotlib to not make any window popups
mpl.use('Agg')

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


class ExampleData(object):
    __slots__ = ('metadata', 'expression', 'splicing', 'data')

    def __init__(self, metadata, expression, splicing):
        self.metadata = metadata
        self.expression = expression
        self.splicing = splicing
        self.data = (metadata, expression, splicing)


@pytest.fixture(scope='module')
def data_dir():
    return '{}/data'.format(CURRENT_DIR.rstrip('/'))


@pytest.fixture(scope='module')
def example_data(data_dir):
    expression = pd.read_table('{}/expression.tsv'.format(data_dir),
                               index_col=0)
    splicing = pd.read_table('{}/splicing.tsv'.format(data_dir), index_col=0)
    metadata = pd.read_csv('{}/metadata.csv'.format(data_dir), index_col=0)
    return ExampleData(metadata, expression, splicing)


@pytest.fixture(scope='module')
def example_study(example_data):
    from flotilla.data_model import Study

    return Study(sample_metadata=example_data.metadata,
                 expression_data=example_data.expression,
                 splicing_data=example_data.splicing)


@pytest.fixture(scope='module')
def example_datapackage_path():
    return os.path.join(CURRENT_DIR, 'data/datapackage.json')


@pytest.fixture(scope='module')
def expression(example_data):
    from flotilla.data_model import ExpressionData

    return ExpressionData(example_data.expression)


@pytest.fixture(scope='module')
def study(example_datapackage_path):
    import flotilla

    return flotilla.embark(example_datapackage_path)


@pytest.fixture
def genelist_path():
    return '{}/test_gene_list.txt'.format(data_dir())


@pytest.fixture
def genelist_dropbox_link():
    return 'https://www.dropbox.com/s/qddybszcses6pi6/DE_genes.male%20adult%20%2019.txt?dl=0'


@pytest.fixture(params=['local', 'dropbox'])
def genelist_link(request, genelist_path, genelist_dropbox_link):
    if request.param == 'local':
        return genelist_path()
    elif request.param == 'dropbox':
        return genelist_dropbox_link()


@pytest.fixture(params=[None, 'transcription_factor',
                        genelist_dropbox_link,
                        genelist_path], scope='module')
def feature_subset(request):
    if request.param is None:
        return request.param
    elif isinstance(request.param, str):
        # If this is a name of a feature subset
        return request.param
    else:
        # Otherwise, this is a link to a list
        from flotilla.external import link_to_list

        try:
            link_to_list(request.param())
        except subprocess.CalledProcessError:
            # Downloading the dropbox link failed, aka not connected to the
            # internet, so just test "None" again
            return None
