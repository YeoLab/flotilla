from flotilla.data_model import ExpressionData, SplicingData, Study

def test_expression_splicing_init(example_data):
    study = Study(sample_metadata=example_data.sample_metadata,
                  expression=example_data.expression,
                  splicing=example_data.splicing)
    expression = ExpressionData(sample_metadata=example_data.sample_metadata,
                                data=example_data.expression)
    splicing = SplicingData(sample_metadata=example_data.sample_metadata,
                            data=example_data.splicing)
    assert study.expression == expression
    assert study.splicing == splicing