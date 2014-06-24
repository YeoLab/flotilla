import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder


class Predictor(object):
    default_predictor_kwargs = {'n_estimators': 100,
                                'bootstrap': True,
                                'max_features': 'auto',
                                'random_state': 0,
                                'oob_score': True,
                                'n_jobs': 2,
                                'verbose': True}
    default_predictor_scoring_fun = lambda clf: clf.feature_importances_

    default_scoring_cutoff_fun = lambda scores: np.mean(scores) \
                                                + 2 * np.std(scores)
    # 2 std's above mean



    default_classifier, default_classifier_name = ExtraTreesClassifier, "ExtraTreesClassifier"
    default_regressor, default_regressor_name = ExtraTreesRegressor, "ExtraTreesRegressor"

    def __init__(self, data, trait_data, name="Predictor",
                 predictor_kwargs=None, predictor_scoring_fun=None,
                 scoring_cutoff_fun=None):
        """train regressors_ or classifiers_ on data.

{       data : pandas.DataFrame
            Samples x features data to train the predictor on
        trait_data : pandas.Series
            Samples x trait (single) data that you want to tell the
            difference between
        name : str
            Titles for plots and things... Default "Predictor"
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
        self.has_been_fit_yet = False
        self.has_been_scored_yet = False
        self.name = name
        self.X = data
        self.important_features = {}

        # Set the keyword argument to the default if it's not already specified
        self.predictor_kwargs = {} if predictor_kwargs is None else predictor_kwargs
        for k, v in self.default_predictor_kwargs:
            self.predictor_kwargs.setdefault(k, v)

        self.predictor_scoring_fun = self.default_predictor_scoring_fun \
            if predictor_scoring_fun is None else predictor_scoring_fun


        # traits from source, in case they're needed later
        self.trait_data = trait_data
        self.trait_name = self.trait_data.name
        sys.stdout.write("Initializing predictor for "
                         "{}\n".format(self.trait_name))

        self.X, self.trait_data = self.X.align(self.trait_data, axis=0,
                                               join='inner')

        # This will be set after calling fit()
        self.predictor = None


class Regressor(Predictor):
    """
    Predictor for continuous data
    """

    def __init__(self, data, trait_data, name="Classifier"):
        super(Regressor, self).__init__()
        self.y = self.trait_data

    def score(self,
              traits=None,
              regressor_name=default_regressor_name,
              feature_scoring_fun=default_regressor_scoring_fun,
              score_cutoff_fun=default_regressor_scoring_cutoff_fun):
        """
        collect scores from classifiers_
        feature_scoring_fun: fxn that yields higher values for better features
        score_cutoff_fun fxn that that takes output of feature_scoring_fun and returns a cutoff
        """
        raise NotImplementedError("Untested, should be close to working.")
        if traits is None:
            traits = self.continuous_traits

        for trait in traits:

            try:
                assert trait in self.regressors_
            except:
                print "trait: %s" % trait, "is missing, continuing"
                continue
            try:
                assert regressor_name in self.regressors_[trait]
            except:
                print "predictor: %s" % regressor_name, "is missing, continuing"
                continue

            print "Scoring predictor: %s for trait: %s... please wait." % (
                regressor_name, trait)

            clf = self.regressors_[trait][regressor_name]
            clf.scores_ = pd.Series(feature_scoring_fun(clf),
                                    index=self.X.columns)
            clf.score_cutoff_ = score_cutoff_fun(clf.scores_)
            self.important_features[trait][regressor_name] = clf.good_features_
            clf.good_features_ = clf.scores_ > clf.score_cutoff_
            clf.n_good_features_ = np.sum(clf.good_features_)
            clf.subset_ = self.X.T[clf.good_features_].T
            print "Finished..."

    def fit(self,
            traits=None,
            regressor_name=default_regressor_name,
            regressor=default_regressor,
            regressor_kwargs=default_regressor_kwargs):
        raise NotImplementedError("Untested, should be close to working.")

        if traits is None:
            traits = self.continuous_traits

        for trait in traits:
            clf = regressor(**regressor_kwargs)
            print "Fitting a predictor for trait %s... please wait." % trait
            clf.fit(self.X, self.y[trait])
            self.regressors_[trait][regressor_name] = clf
            print "Finished..."


class Classifier(Predictor):
    """
    Predictor for categorical data
    """
    boosting_classifier_kwargs = {'n_estimators': 80, 'max_features': 1000,
                                  'learning_rate': 0.2, 'subsample': 0.6, }

    boosting_scoring_fun = lambda clf: clf.feature_importances_
    boosting_scoring_cutoff_fun = lambda scores: np.mean(scores) + 2 * np.std(
        scores)

    def __init__(self, data, trait_data, name="Classifier"):
        super(Classifier, self).__init__()

        # traits encoded to do some work -- "target" variable


        # for trait in self.categorical_traits:
        self.traitset = \
            self.trait_data.groupby(self.trait_data).describe().index.levels[0]
        try:
            assert len(
                self.trait_data.groupby(
                    self.trait_data).describe().index.levels[
                    0]) == 2
        except AssertionError:
            print "WARNING: trait \"%s\" has >2 categories"

        # categorical encoder
        le = LabelEncoder().fit(self.traitset)
        # categorical encoding
        self.y = self.y = pd.Series(data=le.transform(self.trait_data),
                                    index=self.X.index,
                                    name=self.trait_data.name)

    def fit(self,
            classifier_name=default_classifier_name,
            classifier=default_classifier,
            classifier_kwargs=default_classifier_kwargs):
        """ fit classifiers to the data

        predictor : sklearn predictor
            sklearn predictor object such as ExtraTreesClassifier
        classifier_kwargs :
            dictionary for paramters to predictor
        """
        self.predictor = classifier(**classifier_kwargs)
        sys.stdout.write("Fitting a predictor for trait {}... please wait.\n"
                         .format(self.trait))
        self.predictor.fit(self.X, self.y)
        sys.stdout.write("\tFinished.\n")

    def score(self, classifier_name=default_classifier_name,
              feature_scoring_fun=default_classifier_scoring_fun,
              score_cutoff_fun=default_classifier_scoring_cutoff_fun):
        """
        collect scores from classifiers_
        traits - list of trait(s) to score. Retrieved from self.classifiers_[trait]
        classifier_name - a name for this predictor to be retrieved from self.classifiers_[trait][classifier_name]
        feature_scoring_fun - fxn that yields higher values for better features
        score_cutoff_fun - fxn that that takes output of feature_scoring_fun and returns a cutoff
        """
        sys.stdout.write("Scoring predictor: {} for trait: {}... please "
                         "wait\n".format(classifier_name, self.trait))

        # clf = self.classifiers_[self.trait][classifier_name]
        self.predictor.scores_ = pd.Series(
            feature_scoring_fun(self.predictor),
            index=self.X.columns)
        self.predictor.score_cutoff_ = score_cutoff_fun(
            self.predictor.scores_)
        self.predictor.good_features_ = self.predictor.scores_ > self.predictor.score_cutoff_
        self.important_features[self.trait][classifier_name] = self.predictor \
            .good_features_
        self.predictor.n_good_features_ = np.sum(
            self.predictor.good_features_)
        self.predictor.subset_ = self.X.T[self.predictor.good_features_].T

        sys.stdout.write("\tFinished.\n")
        self.has_been_scored_yet = True