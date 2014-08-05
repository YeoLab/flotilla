"""
This tests whether the Study object was created correctly. No
computation or visualization tests yet.
"""
import pandas.util.testing as pdt
import pytest


class TestStudy(object):
    @pytest.fixture
    def toy_study(self, example_data):
        from flotilla import Study

        return Study(sample_metadata=example_data.metadata,
                     expression_data=example_data.expression,
                     splicing_data=example_data.splicing)

    @pytest.fixture
    def study(self, example_url):
        import flotilla

        return flotilla.embark(example_url)

    def test_toy_init(self, toy_study, example_data):
        from flotilla.data_model import ExpressionData, SplicingData

        expression = ExpressionData(data=example_data.expression)
        splicing = SplicingData(data=example_data.splicing)

        pdt.assert_frame_equal(toy_study.metadata.data,
                               example_data.metadata)
        pdt.assert_frame_equal(toy_study.expression.data, expression.data)
        pdt.assert_frame_equal(toy_study.splicing.data, splicing.data)
        # There's more to test for correct initialization but this is barebones
        # for now

    def test_real_init(self, study):
        pass

    def test_plot_pca(self, study):
        study.plot_pca()

    def test_plot_graph(self, study):
        study.plot_graph()

    def test_plot_classifier(self, study):
        study.plot_classifier('P_cell')

        # def test_plot_pca(self, study):
        #     study.interactive_pca()
        #
        # def test_plot_graph(self, study):
        #     study.interactive_graph()
        #
        # def test_plot_classifier(self, study):
        #     study.interactive_classifier()


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
