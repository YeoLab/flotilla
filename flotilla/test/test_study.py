"""
This tests whether the Study object was created correctly. No
computation or visualization tests yet.
"""
import pandas.util.testing as pdt
from flotilla.data_model import ExpressionData, SplicingData, Study

def test_expression_splicing_init(example_data):
    study = Study(phenotype_data=example_data.sample_metadata,
                  expression_data=example_data.expression,
                  splicing_data=example_data.splicing)
    expression = ExpressionData(phenotype_data=example_data.sample_metadata,
                                data=example_data.expression)
    splicing = SplicingData(phenotype_data=example_data.sample_metadata,
                            data=example_data.splicing)
    assert study.expression == expression
    assert study.splicing == splicing

def test_write_package(tmpdir):
    from flotilla.data_model import StudyFactory

    new_study = StudyFactory()
    new_study.sample_metadata = None
    new_study.event_metadata = None
    new_study.expression_metadata = None
    new_study.expression_df = None
    new_study.splicing_df = None
    new_study.event_metadata = None
    new_study.write_package('test_package', 'test_package', install=False,
                            where=tmpdir)
