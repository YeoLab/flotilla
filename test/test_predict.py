import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import pytest
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder


@pytest.fixture
def study(example_data):
    from flotilla.data_model import Study

    return Study(sample_metadata=example_data.metadata,
                 expression_data=example_data.expression,
                 splicing_data=example_data.splicing)


@pytest.fixture
def reduced(study):
    return study.expression.reduce().df


class TestPredictorBase:
    @pytest.fixture
    def trait(self, study):
        return study.experiment_design.data.celltype

    @pytest.fixture
    def predictorbase(self, reduced, trait):
        from flotilla.compute.predict import PredictorBase

        return PredictorBase(reduced, trait)

    def test_init(self, reduced, predictorbase, trait):
        X, trait = reduced.align(trait, axis=0, join='inner')

        assert predictorbase.predictor is None
        assert predictorbase.predictor_class is None
        pdt.assert_frame_equal(predictorbase.X, X)
        pdt.assert_series_equal(predictorbase.trait, trait)
        assert not predictorbase.has_been_fit
        assert not predictorbase.has_been_scored
        assert predictorbase.default_predictor_kwargs == predictorbase.predictor_kwargs
        assert predictorbase.default_score_cutoff_fun == predictorbase.score_cutoff_fun
        assert predictorbase.default_predictor_scoring_fun == predictorbase.predictor_scoring_fun

    def test_fit(self, predictorbase):
        with pytest.raises(ValueError):
            predictorbase.fit()

    def test_score(self, predictorbase):
        with pytest.raises(ValueError):
            predictorbase.score()

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
        assert predictorbase.default_predictor_scoring_fun(x) \
               == x.feature_importances_


class TestRegressor:
    @pytest.fixture
    def y(self, study):
        return pd.Series(np.arange(study.expression.data.shape[0]),
                         name='dummy',
                         index=study.expression.data.index)

    @pytest.fixture
    def predictor_kwargs(self):
        return {'random_state': 2014}

    @pytest.fixture
    def regressor(self, reduced, y, predictor_kwargs):
        import flotilla

        return flotilla.compute.predict.Regressor(reduced, y,
                                                  predictor_kwargs=predictor_kwargs)

    def test_init(self, regressor, y):
        assert regressor.predictor_class == ExtraTreesRegressor
        pdt.assert_series_equal(regressor.y, y)

    def test_fit(self, regressor, reduced, y):
        regressor.fit()
        regressor.predictor.scores_ = regressor.predictor_scoring_fun(regressor
                                                                      .predictor)

        true_regressor = ExtraTreesRegressor(**regressor.predictor_kwargs)
        true_regressor.fit(reduced, y)
        true_regressor.scores_ = regressor.predictor_scoring_fun(true_regressor)

        npt.assert_array_equal(regressor.predictor.scores_,
                               true_regressor.scores_)
        assert regressor.has_been_fit

    def test_score(self, regressor, reduced, y):
        regressor.fit()
        regressor.score()

        true_regressor = ExtraTreesRegressor(**regressor.predictor_kwargs)
        true_regressor.fit(reduced, y)
        scores = regressor.predictor_scoring_fun(true_regressor)
        true_regressor.scores_ = pd.Series(scores, index=reduced.columns)
        true_regressor.score_cutoff_ = regressor.score_cutoff_fun(
            true_regressor.scores_)
        true_regressor.important_features = true_regressor.scores_ > \
                                            true_regressor.score_cutoff_
        true_regressor.n_good_features = np.sum(true_regressor
                                                .important_features)
        true_regressor.subset_ = reduced.T[
            true_regressor.important_features].T

        pdt.assert_series_equal(true_regressor.scores_,
                                regressor.predictor.scores_)
        npt.assert_equal(true_regressor.score_cutoff_,
                         regressor.predictor.score_cutoff_)
        pdt.assert_series_equal(true_regressor.important_features,
                                regressor.important_features)
        assert true_regressor.n_good_features \
               == regressor.predictor.n_good_features_
        pdt.assert_frame_equal(true_regressor.subset_,
                               regressor.predictor.subset_)
        assert regressor.has_been_scored


class TestClassifier:
    pass

    @pytest.fixture
    def trait(self, study):
        return study.experiment_design.data.celltype

    @pytest.fixture
    def y(self, trait):
        traitset = \
            trait.groupby(trait).describe().index.levels[0]
        le = LabelEncoder().fit(traitset)
        return pd.Series(data=le.transform(trait.values),
                         index=trait.index,
                         name=trait.name)

    @pytest.fixture
    def reduced_y_aligned(self, y, reduced):
        reduced, y = reduced.align(y, axis=0, join='inner')
        return reduced, y

    @pytest.fixture
    def classifier(self, reduced, trait):
        from flotilla.compute.predict import Classifier

        return Classifier(reduced, trait)

    def test_init(self, classifier, reduced_y_aligned):
        reduced, y = reduced_y_aligned
        assert classifier.predictor_class == ExtraTreesClassifier
        pdt.assert_series_equal(classifier.y, y)

    def test_fit(self, reduced_y_aligned, classifier):
        reduced, y = reduced_y_aligned

        classifier.fit()
        classifier.predictor.scores_ = classifier.predictor_scoring_fun(
            classifier
            .predictor)

        true_classifier = ExtraTreesClassifier(**classifier.predictor_kwargs)
        true_classifier.fit(reduced, y)
        true_classifier.scores_ = classifier.predictor_scoring_fun(
            true_classifier)

        npt.assert_array_equal(classifier.predictor.scores_,
                               true_classifier.scores_)
        assert classifier.has_been_fit

    def test_score(self, classifier, reduced_y_aligned):
        reduced, y = reduced_y_aligned

        classifier.fit()
        classifier.score()

        true_classifier = ExtraTreesClassifier(**classifier.predictor_kwargs)
        true_classifier.fit(reduced, y)
        scores = classifier.predictor_scoring_fun(true_classifier)
        true_classifier.scores_ = pd.Series(scores,
                                            index=reduced.columns)
        true_classifier.score_cutoff_ = classifier.score_cutoff_fun(
            true_classifier.scores_)
        true_classifier.important_features = true_classifier.scores_ > \
                                             true_classifier.score_cutoff_
        true_classifier.n_good_features = np.sum(true_classifier
                                                 .important_features)
        true_classifier.subset_ = reduced.T[
            true_classifier.important_features].T

        pdt.assert_series_equal(true_classifier.scores_,
                                classifier.predictor.scores_)
        npt.assert_equal(true_classifier.score_cutoff_,
                         classifier.predictor.score_cutoff_)
        pdt.assert_series_equal(true_classifier.important_features,
                                classifier.important_features)
        assert true_classifier.n_good_features \
               == classifier.predictor.n_good_features_
        pdt.assert_frame_equal(true_classifier.subset_,
                               classifier.predictor.subset_)
        assert classifier.has_been_scored