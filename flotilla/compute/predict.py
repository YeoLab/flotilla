__author__ = 'olga'

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import pandas as pd


class Predictor(object):
    extratrees_default_params = {'n_estimators': 100,
                                 'bootstrap': True,
                                 'max_features': 'auto',
                                 'random_state': 0,
                                 'verbose': 1,
                                 'oob_score': True,
                                 'n_jobs': 2,
                                 'verbose': True}

    extratreees_scoring_fun = lambda clf: clf.feature_importances_
    extratreees_scoring_cutoff_fun = lambda scores: np.mean(
        scores) + 2 * np.std(scores)  # 2 std's above mean

    boosting_classifier_params = {'n_estimators': 80, 'max_features': 1000,
                                  'learning_rate': 0.2, 'subsample': 0.6, }
    boosting_scoring_fun = lambda clf: clf.feature_importances_
    boosting_scoring_cutoff_fun = lambda scores: np.mean(scores) + 2 * np.std(
        scores)

    default_classifier, default_classifier_name = ExtraTreesClassifier, "ExtraTreesClassifier"
    default_regressor, default_regressor_name = ExtraTreesRegressor, "ExtraTreesRegressor"

    default_classifier_scoring_fun = default_regressor_scoring_fun = extratreees_scoring_fun
    default_classifier_scoring_cutoff_fun = default_regressor_scoring_cutoff_fun = extratreees_scoring_cutoff_fun
    default_classifier_params = default_regressor_params = extratrees_default_params

    def __init__(self, data_df, metadata_df,
                 name="Classifier",
                 categorical_traits=None,
                 continuous_traits=None):
        """train regressors_ or classifiers_ on data.

        name : str
            titles for plots and things...
        sample_list : list of str
            a list of sample ids for this comparer
        critical_variable : str
            a response variable to test or a list of them
        data_df : pandas.DataFrame
            containing arrays in question
        metadata_df : pandas.DataFrame
            pd.DataFrame with metadata about data_df
        categorical_traits : list of str
            which traits are catgorical? - if None, assumed to be all traits
        continuous_traits : list of str
            which traits are continuous - i.e. build a regressor, not a
            classifier
        """

        self.has_been_fit_yet = False
        self.has_been_scored_yet = False
        self.name = name
        self.X = data_df
        self.important_features = {}
        self.traits = []
        self.categorical_traits = categorical_traits
        if categorical_traits is not None:
            self.traits.extend(categorical_traits)

        self.continuous_traits = continuous_traits
        if continuous_traits is not None:
            self.traits.extend(continuous_traits)

        print "Initializing predictors for %s" % " and ".join(self.traits)


        #print "Using traits: ", self.traits

        self.trait_data = metadata_df[
            self.traits]  #traits from source, in case they're needed later
        self.X, self.trait_data = self.X.align(self.trait_data, axis=0,
                                               join='inner')
        self.y = pd.DataFrame(index=self.X.index,
                              columns=self.traits)  #traits encoded to do some work -- "target" variable

        self.classifiers_ = {}
        from sklearn.preprocessing import LabelEncoder

        for trait in self.traits:
            self.important_features[trait] = {}

        for trait in self.categorical_traits:
            try:
                assert len(
                    metadata_df.groupby(trait).describe().index.levels[0]) == 2
            except AssertionError:
                print "WARNING: trait \"%s\" has >2 categories"
            self.classifiers_[trait] = {}
            traitset = metadata_df.groupby(trait).describe().index.levels[0]
            le = LabelEncoder().fit(traitset)  #categorical encoder
            self.y[trait] = le.transform(
                self.trait_data[trait])  #categorical encoding

        self.continuous_traits = continuous_traits
        self.regressors_ = {}
        if self.continuous_traits is not None:

            for trait in self.continuous_traits:
                self.regressors_[trait] = {}
                self.y[trait] = self.trait_data[trait]

    def fit_classifiers(self,
                        traits=None,
                        classifier_name=default_classifier_name,
                        classifier=default_classifier,
                        classifier_params=default_classifier_params):
        """ fit classifiers_ to the data
        traits - list of trait(s) to fit a classifier upon,
        if None, fit all traits that were initialized.
        Classifiers on each trait will be stored in: self.classifiers_[trait]

        classifier_name - a name for this classifier to be stored in self.classifiers_[trait][classifier_name]
        classifier - sklearn classifier object such as ExtraTreesClassifier
        classifier_params - dictionary for paramters to classifier
        """

        if traits is None:
            traits = self.categorical_traits
        else:
            assert type(traits) == list or type(traits) == set
            traits = traits

        for trait in traits:
            clf = classifier(**classifier_params)
            print "Fitting a classifier for trait %s... please wait." % trait
            clf.fit(self.X, self.y[trait])
            self.classifiers_[trait][classifier_name] = clf
            print "Finished..."
        self.has_been_fit_yet = True


    def score_classifiers(self,
                          traits=None,
                          classifier_name=default_classifier_name,
                          feature_scoring_fun=default_classifier_scoring_fun,
                          score_cutoff_fun=default_classifier_scoring_cutoff_fun):
        """
        collect scores from classifiers_
        traits - list of trait(s) to score. Retrieved from self.classifiers_[trait]
        classifier_name - a name for this classifier to be retrieved from self.classifiers_[trait][classifier_name]
        feature_scoring_fun - fxn that yields higher values for better features
        score_cutoff_fun - fxn that that takes output of feature_scoring_fun and returns a cutoff
        """

        if traits is None:
            traits = self.categorical_traits

        for trait in traits:

            try:
                assert trait in self.classifiers_
            except:
                print "trait: %s" % trait, "is missing, continuing"
                continue
            try:
                assert classifier_name in self.classifiers_[trait]
            except:
                print "classifier: %s" % classifier_name, "is missing, continuing"
                continue

            print "Scoring classifier: %s for trait: %s... please wait." % (
                classifier_name, trait)

            clf = self.classifiers_[trait][classifier_name]
            clf.scores_ = pd.Series(feature_scoring_fun(clf),
                                    index=self.X.columns)
            clf.score_cutoff_ = score_cutoff_fun(clf.scores_)
            clf.good_features_ = clf.scores_ > clf.score_cutoff_
            self.important_features[trait][classifier_name] = clf.good_features_
            clf.n_good_features_ = np.sum(clf.good_features_)
            clf.subset_ = self.X.T[clf.good_features_].T

            print "Finished..."
        self.has_been_scored_yet = True

    def fit_regressors(self,
                       traits=None,
                       regressor_name=default_regressor_name,
                       regressor=default_regressor,
                       regressor_params=default_regressor_params,
    ):
        raise NotImplementedError("Untested, should be close to working.")

        if traits is None:
            traits = self.continuous_traits

        for trait in traits:
            clf = regressor(**regressor_params)
            print "Fitting a classifier for trait %s... please wait." % trait
            clf.fit(self.X, self.y[trait])
            self.regressors_[trait][regressor_name] = clf
            print "Finished..."

    def score_regressors(self,
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
                print "classifier: %s" % regressor_name, "is missing, continuing"
                continue

            print "Scoring classifier: %s for trait: %s... please wait." % (
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