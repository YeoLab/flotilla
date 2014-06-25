import pandas.util.testing as pdt
import pytest


class TestPredictorBase:
    @pytest.fixture
    def predictorbase(self, example_data, trait):
        from flotilla.compute.predict import PredictorBase

        return PredictorBase(example_data.expression, trait)

    @pytest.fixture
    def trait(self, example_data):
        return example_data.experiment_design_data.celltype

    def test_init(self, example_data, predictorbase, trait):
        assert predictorbase.predictor is None
        pdt.assert_frame_equal(predictorbase.X, example_data.expression)
        pdt.assert_series_equal(predictorbase.trait_data, trait)
        assert predictorbase.has_been_fit_yet == False
        assert predictorbase.has_been_scored_yet == False

        # Nested is necessary
        with pytest.raises(ValueError) as excinfo:
            def test_fit():
                predictorbase.fit()

            test_fit()

        with pytest.raises(ValueError) as excinfo:
            def test_score():
                predictorbase.score()

            test_score()
