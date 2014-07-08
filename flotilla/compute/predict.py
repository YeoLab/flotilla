"""
Compute predictors on data, i.e. regressors or classifiers
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder


class PredictorBase(object):
    default_predictor_kwargs = {'n_estimators': 100,
                                'bootstrap': True,
                                'max_features': 'auto',
                                'random_state': 0,
                                'oob_score': True,
                                'n_jobs': 2,
                                'verbose': True}

    def __init__(self, data, trait, predictor=None, name="Predictor",
                predictor_kwargs=None, predictor_scoring_fun=None,
                score_cutoff_fun=None):
        """Initializer for scikit-learn predictors (classifiers and regressors)

        Initalizes everything except the "y" (response variable). This must
        be initialized by the base class.

        Parameters
        ----------
        data : pandas.DataFrame
            Samples x features data to train the predictor on
        trait : pandas.Series
            Samples x trait (single) data that you want to tell the
            difference between
        predictor : scikit-learn classifier
            Which regression to use. Default ExtraTreesClassifier
        name : str
            Titles for plots and things... Default "ExtraTreesRegressor"
        predictor_kwargs : dict
            Extra arguments to the predictor. Default:
            {'n_estimators': 100,
             'bootstrap': True,
             'max_features': 'auto',
             'random_state': 0,
             'oob_score': True,
             'n_jobs': 2,
             'verbose': True}
        predictor_scoring_fun : function
            Function to get the feature scores for a scikit-learn classifier.
            This can be different for different classifiers, e.g. for a
            classifier named "clf" it could be clf.scores_, for other it's
            clf.feature_importances_. Default: lambda clf: clf.feature_importances_
        score_cutoff_fun : function
            Function to cut off insignificant scores (scores as returned by
            predictor_scoring_fun)
            Default: lambda scores: np.mean(scores) + 2 * np.std(scores)
        """
        self.predictor_class = predictor

        self.has_been_fit = False
        self.has_been_scored = False
        self.name = name
        self.X = data
        self.important_features = {}

        # Set the keyword argument to the default if it's not already specified
        self.predictor_kwargs = {} if predictor_kwargs is None else predictor_kwargs
        for k, v in self.default_predictor_kwargs.items():
            self.predictor_kwargs.setdefault(k, v)

        self.predictor_scoring_fun = self.default_predictor_scoring_fun \
            if predictor_scoring_fun is None else predictor_scoring_fun
        self.score_cutoff_fun = self.default_score_cutoff_fun \
            if score_cutoff_fun is None else score_cutoff_fun


        # traits from source, in case they're needed later
        self.trait = trait
        self.trait_name = self.trait.name
        sys.stdout.write("Initializing predictor for "
                         "{}\n".format(self.trait_name))

        self.X, self.trait = self.X.align(self.trait, axis=0,
                                          join='inner')

        # This will be set after calling fit()
        self.predictor = None

    @staticmethod
    def default_predictor_scoring_fun(x):
        return x.feature_importances_

    @staticmethod
    def default_score_cutoff_fun(x):
        return np.mean(x) + 2 * np.std(x)

    def fit(self):
        """Fit predictor to the data
        """
        try:
            self.predictor = self.predictor_class(**self.predictor_kwargs)
        except TypeError:
            raise ValueError('There is no predictor assigned to this '
                             'PredictorBase class yet')
        sys.stdout.write("Fitting a predictor for trait {}... please wait.\n"
                         .format(self.trait_name))
        self.predictor.fit(self.X, self.y)
        sys.stdout.write("\tFinished.\n")
        self.has_been_fit = True

    def score(self):
        """Collect scores from predictor
        """
        sys.stdout.write("Scoring predictor: {} for trait: {}... please "
                         "wait\n".format(self.name, self.trait_name))
        try:
            scores = self.predictor_scoring_fun(self.predictor)
        except AttributeError:
            raise ValueError('There is no predictor assigned to this '
                             'PredictorBase class yet')
        self.predictor.scores_ = pd.Series(index=self.X.columns, data=scores)
        self.predictor.score_cutoff_ = \
            self.score_cutoff_fun(self.predictor.scores_)
        # self.predictor.good_features_ =
        self.important_features = self.predictor.scores_ > self.predictor.score_cutoff_
        self.predictor.n_good_features_ = \
            np.sum(self.important_features)
        self.predictor.subset_ = self.X.T[self.important_features].T

        sys.stdout.write("\tFinished.\n")
        self.has_been_scored = True


class Regressor(PredictorBase):
    """
    Regressor - for continuous data
    """

    default_regressor = ExtraTreesRegressor

    def __init__(self, data, trait, predictor=ExtraTreesRegressor,
                 name="ExtraTreesRegressor",
                 predictor_kwargs=None, predictor_scoring_fun=None,
                 score_cutoff_fun=None):
        """Train a regressor on data.

        Parameters
        ----------
        data : pandas.DataFrame
            Samples x features data to train the predictor on
        trait : pandas.Series
            Samples x trait (single) data that you want to tell the
            difference between
        predictor : scikit-learn regressor
            Which regression to use. Default ExtraTreesRegressor
        name : str
            Titles for plots and things... Default "ExtraTreesRegressor"
        predictor_kwargs : dict
            Extra arguments to the predictor. Default:
            {'n_estimators': 100,
             'bootstrap': True,
             'max_features': 'auto',
             'random_state': 0,
             'oob_score': True,
             'n_jobs': 2,
             'verbose': True}
        predictor_scoring_fun : function
            Function to get the feature scores for a scikit-learn regressor.
            This can be different for different classifiers, e.g. for a
            regressor named "x" it could be x.scores_, for other it's
            x.feature_importances_. Default: lambda x: x.feature_importances_
        score_cutoff_fun : function
            Function to cut off insignificant scores
            Default: lambda x: np.mean(x) + 2 * np.std(x)
        """
        super(Regressor, self).__init__(data=data, trait=trait,
                                        predictor=predictor, name=name,
                                        predictor_kwargs=predictor_kwargs,
                                        predictor_scoring_fun=predictor_scoring_fun,
                                        score_cutoff_fun=score_cutoff_fun)
        self.y = self.trait
        self.predictor_class = ExtraTreesRegressor \
            if self.predictor_class is None else self.predictor_class


class Classifier(PredictorBase):
    """
    Classifier - for categorical data
    """
    boosting_classifier_kwargs = {'n_estimators': 80, 'max_features': 1000,
                                  'learning_rate': 0.2, 'subsample': 0.6, }

    boosting_scoring_fun = lambda clf: clf.feature_importances_
    boosting_scoring_cutoff_fun = lambda scores: np.mean(scores) + 2 * np.std(
        scores)

    default_classifier, default_classifier_name = ExtraTreesClassifier, "ExtraTreesClassifier"

    def __init__(self, data, trait, predictor=ExtraTreesClassifier,
                 name="ExtraTreesClassifier",
                 predictor_kwargs=None, predictor_scoring_fun=None,
                 score_cutoff_fun=None):
        """Train a classifier on data.

        Parameters
        ----------
        data : pandas.DataFrame
            Samples x features data to train the predictor on
        trait : pandas.Series
            Samples x trait (single) data that you want to tell the
            difference between
        predictor : scikit-learn classifier
            Which classifier to use. Default ExtraTreesClassifier
        name : str
            Titles for plots and things... Default "ExtraTreesClassifier"
        predictor_kwargs : dict
            Extra arguments to the predictor. Default:
            {'n_estimators': 100,
             'bootstrap': True,
             'max_features': 'auto',
             'random_state': 0,
             'oob_score': True,
             'n_jobs': 2,
             'verbose': True}
        predictor_scoring_fun : function
            Function to get the feature scores for a scikit-learn classifier.
            This can be different for different classifiers, e.g. for a
            classifier named "x" it could be x.scores_, for other it's
            x.feature_importances_. Default: lambda x: x.feature_importances_
        score_cutoff_fun : function
            Function to cut off insignificant scores
            Default: lambda scores: np.mean(x) + 2 * np.std(x)
        """
        super(Classifier, self).__init__(data=data, trait=trait,
                                         predictor=predictor, name=name,
                                         predictor_kwargs=predictor_kwargs,
                                         predictor_scoring_fun=predictor_scoring_fun,
                                         score_cutoff_fun=score_cutoff_fun)
        self.predictor_class = ExtraTreesClassifier \
            if self.predictor_class is None else self.predictor_class

        # traits encoded to do some work -- "target" variable
        self.traitset = \
            self.trait.groupby(self.trait).describe().index.levels[0]
        try:
            assert len(
                self.trait.groupby(
                    self.trait).describe().index.levels[
                    0]) == 2
        except AssertionError:
            warnings.warn("WARNING: trait {} has >2 categories".format(
                self.trait_name))

        # categorical encoder
        le = LabelEncoder().fit(self.traitset)

        # categorical encoding
        self.y = pd.Series(data=le.transform(self.trait),
                           index=self.X.index,
                           name=self.trait.name)


