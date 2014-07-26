"""
This tests whether the Study object was created correctly. No
computation or visualization tests yet.
"""
import pandas.util.testing as pdt


def test_study_init(example_data):
    from flotilla.data_model import ExpressionData, SplicingData, Study

    study = Study(sample_metadata=example_data.metadata,
                  expression_data=example_data.expression,
                  splicing_data=example_data.splicing)
    expression = ExpressionData(data=example_data.expression)
    splicing = SplicingData(data=example_data.splicing)

    pdt.assert_frame_equal(study.metadata.data,
                           example_data.experiment_design_data)
    pdt.assert_frame_equal(study.expression.data, expression.data)
    pdt.assert_frame_equal(study.splicing.data, splicing.data)
    # There's more to test for correct initialization but this is barebones
    # for now

# def test_write_package(tmpdir):
#     from flotilla.data_model import StudyFactory
#
#     new_study = StudyFactory()
#     new_study.experiment_design_data = None
#     new_study.event_metadata = None
#     new_study.expression_metadata = None
#     new_study.expression_df = None
#     new_study.splicing_df = None
#     new_study.event_metadata = None
#     new_study.write_package('test_package', where=tmpdir, install=False)
