
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