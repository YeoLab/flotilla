import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import pytest
from sklearn.ensemble import ExtraTreesRegressor


class TestPredictorBase:
    @pytest.fixture

    def trait(self, example_data):
        return example_data.experiment_design_data.celltype

    @pytest.fixture
    def predictorbase(self, example_data, trait):
        from flotilla.compute.predict import PredictorBase

        return PredictorBase(example_data.expression, trait)

    def test_init(self, example_data, predictorbase, trait):
        assert predictorbase.predictor is None
        assert predictorbase.predictor_class is None
        pdt.assert_frame_equal(predictorbase.X, example_data.expression)
        pdt.assert_series_equal(predictorbase.trait_data, trait)
        assert not predictorbase.has_been_fit_yet
        assert not predictorbase.has_been_scored_yet
        assert predictorbase.default_predictor_kwargs == predictorbase.predictor_kwargs
        assert predictorbase.default_score_cutoff_fun == predictorbase.score_cutoff_fun
        assert predictorbase.default_predictor_scoring_fun == predictorbase.predictor_scoring_fun

        # Nested is necessary
        with pytest.raises(ValueError) as excinfo:
            def test_fit():
                predictorbase.fit()
            test_fit()

        with pytest.raises(ValueError) as excinfo:
            def test_score():
                predictorbase.score()
            test_score()

    def test_default_score_cutoff_fun(self, predictorbase):
        x = np.arange(10)
        cutoff = np.mean(x) + 2 * np.std(x)

        npt.assert_approx_equal(cutoff,
                                predictorbase.default_score_cutoff_fun(x))

    def test_predictor_scoring_fun(self, predictorbase):
        class X:
            def feature_importances_(self):
                pass

        x = X()
        assert predictorbase.default_predictor_scoring_fun(
            x) == x.feature_importances_


class TestRegressor:
    @pytest.fixture
    def trait(self, example_data):
        return pd.Series(np.arange(example_data.expression.shape[0]),
                         name='dummy',
                         index=example_data.expression.index)


    @pytest.fixture
    def predictor_kwargs(self):
        return {'random_state': 2014}

    @pytest.fixture
    def regressor(self, example_data, trait, predictor_kwargs):
        import flotilla

        return flotilla.compute.predict.Regressor(example_data.expression,
                                                  trait,
                                                  predictor_kwargs=predictor_kwargs)

    def test_init(self, regressor, trait):
        assert isinstance(regressor.predictor_class, ExtraTreesRegressor)
        pdt.assert_series_equal(regressor.y, trait)

    def test_fit(self, regressor, example_data, trait):
        regressor.fit()

        true_regressor = ExtraTreesRegressor(**regressor.predictor_kwargs)
        true_regressor.fit(example_data.expression, trait)

        npt.assert_array_equal(regressor.predictor.scores_,
                               true_regressor.scores_)
        assert regressor.has_been_fit_yet

    def test_score(self, regressor, example_data, trait):
        regressor.fit()
        regressor.score()

        true_regressor = ExtraTreesRegressor(**regressor.predictor_kwargs)
        true_regressor.fit(example_data.expression, trait)
        scores = regressor.predictor_scoring_fun(true_regressor)
        true_regressor.scores_ = pd.Series(scores,
                                           index=example_data.expression.columns)
        true_regressor.score_cutoff_ = regressor.score_cutoff_fun(
            true_regressor.scores_)
        true_regressor.important_features = true_regressor.scores_ > \
                                            true_regressor.score_cutoff_
        true_regressor.n_good_features = np.sum(true_regressor
                                                .important_features)
        true_regressor.subset_ = example_data.expression.T[
            true_regressor.important_features].T

        pdt.assert_series_equal(true_regressor.scores_,
                                regressor.predictor.scores_)
        npt.assert_equal(true_regressor.score_cutoff_,
                         regressor.predictor.score_cutoff_)
        pdt.assert_series_equal(true_regressor.important_features,
                                regressor.important_features)
        assert true_regressor.n_good_features == regressor.predictor \
            .n_good_features_
        pdt.assert_frame_equal(true_regressor.subset_,
                               regressor.predictor.subset_)
        assert regressor.has_been_scored_yet


class TestClassifier:
    pass
