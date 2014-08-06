"""
Compute predictors on data, i.e. regressors or classifiers
"""

import sys
import warnings
from collections import defaultdict
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

from ..util import memoize
from ..visualize.decomposition import PCAViz, NMFViz


default_classifier = 'ExtraTreesClassifier'
default_regressor = 'ExtraTreesRegressor'
default_score_coefficient = 2


def default_predictor_scoring_fun(cls):
    """most predictors score output coefficients in the variable cls.feature_importances_
    others may use another name for scores"""
    return cls.feature_importances_


def default_score_cutoff_fun(arr, std_multiplier=default_score_coefficient):
    """a way of calculating which features have a high score in a predictor
    default - f(x) = mean(x) + 2 * np.std(x)"""
    return np.mean(arr) + std_multiplier * np.std(arr)


from pandas.util.testing import assert_frame_equal, assert_series_equal


class PredictorConfig(object):
    """
    A configuration for a predictor, names and tracks/sets parameters for predictor
    Dynamically configures some args for predictor based on n_features (if this attribute exists)
    set general parameters with __init__
    yield instances, set by your parameters, with __call__
    """


    def __init__(self, predictor_name, obj=None,
                 predictor_scoring_fun=default_predictor_scoring_fun,
                 score_cutoff_fun=default_score_cutoff_fun,
                 n_features_dependent_parameters=None,
                 constant_parameters=None):
        """


        Parameters
        ==========


        predictor_name - a name for this predictor
        obj - predictor object, like ExtraTreesClassifier
        predictor_scoring_fun - a function that returns the scores
          from predictor, like ExtraTreesClassifier.feature_importances_
        score_cutoff_fun - a function that takes the scores from predictor_scoring_fun
          and returns a cutoff to label features important/not important
        constant_parameters - dict of kwargs for obj initialization

        Options to set parameters by input dataset size:

        n_features_dependent_parameters - dict of (key, setter) kwargs for obj
        initialization, setter is a function that uses dataset shape to scale parameters

        """

        if obj is None:
            raise ValueError

        if n_features_dependent_parameters is None:
            n_features_dependent_parameters = {}
        self.n_features_dependent_parameters = n_features_dependent_parameters
        if constant_parameters is None:
            constant_parameters = {}
        self.constant_parameters = constant_parameters
        self.predictor_scoring_fun = predictor_scoring_fun
        self.score_cutoff_fun = score_cutoff_fun
        self.predictor_name = predictor_name
        sys.stderr.write(
            "predictor {} is of type {}\n".format(self.predictor_name, obj))
        self._parent = obj
        self.__doc__ = obj.__doc__
        sys.stderr.write(
            "added {} to default predictors\n".format(self.predictor_name))

    @memoize
    def parameters(self, n_features):
        """

        recommended parameters for this classifier for n_features-sized dataset

         Parameters:
         ===========
         n_features - int is used to scale appropriate kwargs to predictor

         """
        parameters = {}
        for parameter, setter in self.n_features_dependent_parameters.items():
            parameters[parameter] = setter(n_features)

        for parameter, value in self.constant_parameters.items():
            parameters[parameter] = value

        return parameters

    def __call__(self, n_features=None):
        """return an instance of this object, initialized and with extra attributes"""
        if n_features is None:
            raise Exception
        parameters = self.parameters(n_features)

        sys.stderr.write(
            "configuring predictor type: {} with {} features".format(
                self.predictor_name, n_features))

        prd = self._parent(**parameters)
        prd.score_cutoff_fun = self.score_cutoff_fun
        prd.predictor_scoring_fun = self.predictor_scoring_fun
        prd.has_been_fit = False
        prd.has_been_scored = False
        prd._score_coefficient = default_score_coefficient
        return prd


class PredictorConfigScalers(object):
    """because it's useful to have scaling parameters for predictors
    that adjust according to the dataset size"""

    _default_coef = 2.5
    _default_nfeatures = 500

    @staticmethod
    def max_feature_scaler(n_features=_default_nfeatures, coef=_default_coef):
        if n_features is None:
            raise Exception
        return int(math.ceil(np.sqrt(np.sqrt(n_features) ** (coef + .3))))

    @staticmethod
    def n_estimators_scaler(n_features=_default_nfeatures, coef=_default_coef):
        if n_features is None:
            raise Exception
        return int(math.ceil((n_features / 50.) * coef))

    @staticmethod
    def n_jobs_scaler(n_features=_default_nfeatures):
        if n_features is None:
            raise Exception

        return int(min(4, math.ceil(n_features / 2000.)))


class ConfigOptimizer(object):
    """choose the coef that makes some result most likely at all n_features
    (or some other function of the dataset)"""


    @staticmethod
    def objective_average_times_seen(n_features,
                                     coef=PredictorConfigScalers._default_coef,
                                     max_feature_scaler=PredictorConfigScalers.max_feature_scaler,
                                     n_estimators_scaler=PredictorConfigScalers.n_estimators_scaler):
        return ((n_features / max_feature_scaler(n_features, coef))
                * n_estimators_scaler(n_features, coef)) / float(n_features)

    @staticmethod
    def optimize_target(n_features, target, **kwargs):
        """choose the  coefficient that optimizes the result of some
        objective function of a predictor's parameters to return a user-chosen target value"""
        raise NotImplementedError
        return PredictorConfigScalers(coef=optimized_coef)
        #scipy.optimize(.....)


class PredictorConfigManager(object):
    """
    A container for predictor configurations, includes several built-ins

    >>> pcm = PredictorConfigManager()
    #add a new type of predictor
    >>> pcm.new_predictor_config(ExtraTreesClassifier,
                          'ExtraTreesClassifier',
                          n_features_dependent_parameters={'max_features': PredictorConfigScalers.max_feature_scaler,
                                                           'n_estimators': PredictorConfigScalers.n_estimators_scaler,
                                                           'n_jobs': PredictorConfigScalers.n_jobs_scaler},

                          constant_parameters={'bootstrap': True,
                                               'random_state': 0,
                                               'oob_score': True,
                                               'verbose': True}))
    """

    @property
    def builtin_predictor_configs(self):
        return self.predictor_configs.keys()

    @property
    def predictor_configs(self):
        if not hasattr(self, '_predictors'):
            self._predictors = {}

        return self._predictors

    def predictor_config(self, name, **kwargs):

        prd = self.new_predictor_config(name, **kwargs)
        if name in self.predictor_configs and self.predictor_configs[
            name] != prd:
            sys.stderr.write(
                "WARNING: over-writing predictor named: {}".format(name))
        self.predictor_configs[name] = prd
        return prd

    @memoize
    def new_predictor_config(self, name, obj=None,
                             predictor_scoring_fun=None,
                             score_cutoff_fun=None,
                             n_features_dependent_parameters=None,
                             constant_parameters=None,
    ):

        """add a predictor to self.predictors"""

        if obj is None:

            args = np.array([predictor_scoring_fun, score_cutoff_fun,
                             n_features_dependent_parameters,
                             constant_parameters])
            if np.any([i is not None for i in args]):
                #if obj is None, you'd better not be asking to set parameters on it.
                raise Exception

            try:
                return self.predictor_configs[name]
            except KeyError:
                raise KeyError("No such predictor: {}".format(name))

        if predictor_scoring_fun is None:
            predictor_scoring_fun = default_predictor_scoring_fun

        if score_cutoff_fun is None:
            score_cutoff_fun = default_score_cutoff_fun

        if n_features_dependent_parameters is not None:
            if type(n_features_dependent_parameters) is not dict:
                raise TypeError
        else:
            n_features_dependent_parameters = {}

        if constant_parameters is not None:
            if type(constant_parameters) is not dict:
                raise TypeError
        else:
            constant_parameters = {}

        return PredictorConfig(name, obj,
                               predictor_scoring_fun=predictor_scoring_fun,
                               score_cutoff_fun=score_cutoff_fun,
                               n_features_dependent_parameters=n_features_dependent_parameters,
                               constant_parameters=constant_parameters)


    def __init__(self):

        constant_extratrees_params = {'bootstrap': True,
                                      'random_state': 0,
                                      'oob_score': True,
                                      'verbose': True}

        self.predictor_config('ExtraTreesClassifier',
                              obj=ExtraTreesClassifier,
                              n_features_dependent_parameters={
                                  'max_features': PredictorConfigScalers.max_feature_scaler,
                                  'n_estimators': PredictorConfigScalers.n_estimators_scaler,
                                  'n_jobs': PredictorConfigScalers.n_jobs_scaler},
                              constant_parameters=constant_extratrees_params)

        self.predictor_config('ExtraTreesRegressor',
                              obj=ExtraTreesRegressor,
                              n_features_dependent_parameters={
                                  'max_features': PredictorConfigScalers.max_feature_scaler,
                                  'n_estimators': PredictorConfigScalers.n_estimators_scaler,
                                  'n_jobs': PredictorConfigScalers.n_jobs_scaler},
                              constant_parameters=constant_extratrees_params)

        constant_boosting_params = {'n_estimators': 80, 'max_features': 1000,
                                    'learning_rate': 0.2, 'subsample': 0.6, }

        self.predictor_config('GradientBoostingClassifier',
                              obj=GradientBoostingClassifier,
                              constant_parameters=constant_boosting_params)

        self.predictor_config('GradientBoostingRegressor',
                              obj=GradientBoostingRegressor,
                              constant_parameters=constant_boosting_params)


class PredictorDataSet(object):
    """
    X and y dataset
    """

    @property
    def X(self):
        return self._data.align(self._y, axis=0,
                                join='inner')[0]

    @property
    def y(self):
        return self._data.align(self._y, axis=0,
                                join='inner')[1]

    @property
    def traitset(self):

        """
         a set of all values that appear in self.trait
        """
        return self.trait.groupby(self.trait).describe().index.levels[0]

    @property
    def predictors(self):

        """instances of PredictorConfig, keep predictors with datasets"""

        if not hasattr(self, '_predictors'):
            self._predictors = defaultdict(dict)

        return self._predictors

    @memoize
    def predictor(self, name, **kwargs):

        """a single, initialized, PredictorConfig instance"""

        prd = self.predictor_config_manager.predictor_config(name, **kwargs)
        initialized = prd(self.n_features)
        self.predictors[name] = initialized
        return initialized

    def __init__(self, data, trait,
                 data_name="MyDataset",
                 categorical_trait=False,
                 predictor_config_manager=None):

        """
         data - X
         trait - y
         data_name - name to store this dataset, to be used with trait.name
         categorical_trait - is y categorical?
        """

        if type(trait) != pd.Series:
            raise TypeError("Traits must be pandas.Series objects")

        self.dataset_name = (data_name, trait.name)
        self.data_name = data_name
        self._data = data
        self.trait = trait
        self.trait_name = self.trait.name
        self.categorical_trait = categorical_trait

        if categorical_trait:

            if len(self.traitset) > 2:
                warnings.warn("WARNING: trait {} has >2 categories".format(
                    self.trait_name))

            # categorical encoder
            le = LabelEncoder().fit(self.traitset)

            # categorical encoding
            self._y = pd.Series(data=le.transform(self.trait),
                                index=trait.index,
                                name=self.trait.name)

        else:
            self._y = trait

        self.predictor_config_manager = predictor_config_manager \
            if predictor_config_manager is not None \
            else PredictorConfigManager()

        self.n_features = self.X.shape[1]

    def check_if_equal(self, data, trait, categorical_trait):

        """Check if this is the same as another dataset. Raises an Exception if not the same"""

        raise NotImplementedError("not tested yet")
        assert_frame_equal(data, self._data)
        assert_series_equal(trait, self.trait)
        if categorical_trait != self.categorical_trait:
            raise RuntimeError


class PredictorDataSetManager(object):
    """
     a collection of PredictorDataSet instances.
     use self.datasets for convienient retrieval of predictors.
    """

    def __init__(self, predictor_config_manager=None):
        self.predictor_config_manager = predictor_config_manager \
            if predictor_config_manager is not None \
            else PredictorConfigManager()

    @property
    def datasets(self):
        if not hasattr(self, '_datasets'):
            # 3 layer deep (data, trait, categorical?)
            # will almost always be either categorical true or false, rarely both
            self._datasets = defaultdict(lambda: defaultdict(dict))
        return self._datasets

    def dataset(self, data_name, trait_name, categorical_trait=False,
                **kwargs):
        kwargs['categorical_trait'] = categorical_trait
        dataset = self.new_dataset(data_name, trait_name, **kwargs)

        if data_name in self.datasets:
            if trait_name in self.datasets[data_name]:
                if categorical_trait in self.datasets[data_name][trait_name] and \
                                self.datasets[data_name][trait_name][
                                    categorical_trait] != dataset:
                    sys.stderr.write(
                        "WARNING: over-writing dataset named: {}".format(
                            (data_name,
                             trait_name,
                             categorical_trait)))
                    self.datasets[data_name][trait_name][
                        categorical_trait] = dataset
                else:
                    self.datasets[data_name][trait_name][
                        categorical_trait] = dataset
            else:
                self.datasets[data_name][trait_name][
                    categorical_trait] = dataset
        else:
            self.datasets[data_name][trait_name][categorical_trait] = dataset

        return dataset

    @memoize
    def new_dataset(self, data_name, trait_name,
                    categorical_trait=False,
                    data=None, trait=None,
                    predictor_config_manager=None):

        if data is None:
            #try to get this dataset by key in the dictionary
            args = np.array([data, trait, predictor_config_manager])
            if np.any([i is not None for i in args]):
                #if data is None, you'd better not be asking to set other parameters
                raise Exception
            try:
                return self.datasets[data_name][trait_name][categorical_trait]
            except KeyError:
                raise KeyError("No such dataset: {}".format(
                    (data_name, trait_name, categorical_trait)))

        if trait is None:
            raise Exception

        if trait_name != trait.name:
            raise ValueError

        if data_name is None:
            data_name = "MyData"

        predictor_config_manager = predictor_config_manager if predictor_config_manager is not None \
            else self.predictor_config_manager

        return PredictorDataSet(data, trait, data_name,
                                categorical_trait=categorical_trait,
                                predictor_config_manager=predictor_config_manager)


class PredictorBase(object):
    """

    One datset, one predictor, from dataset manager.


    Parameters
    ----------

    predictor_name : str
        Name for predictor
    data_name : str
        Name for this (subset of the) data
    trait_name : str
        Name for this trait
    X_data : pandas.DataFrame
        Samples-by-features (row x col) dataset to train the predictor on
    trait : pandas.Series
        A variable you want to predict using X_data. Indexed like X_data.
    predictor_obj : scikit-learn object that implements fit and score on (X_data,trait)
        Which classifier to use. Default ExtraTreesClassifier
    predictor_scoring_fun : function
        Function to get the feature scores for a scikit-learn classifier.
        This can be different for different classifiers, e.g. for a
        classifier named "x" it could be x.scores_, for other it's
        x.feature_importances_. Default: lambda x: x.feature_importances_
    score_cutoff_fun : function
        Function to cut off insignificant scores
        Default: lambda scores: np.mean(x) + 2 * np.std(x)
    n_features_dependent_parameters : dict
        kwargs to the predictor that depend on n_features
        Default:
        {}
    constant_parameters : dict
        kwargs to the predictor that are constant, i.e.:
        {'n_estimators': 100,
         'bootstrap': True,
         'max_features': 'auto',
         'random_state': 0,
         'oob_score': True,
         'n_jobs': 2,
         'verbose': True}

    """

    def __init__(self, predictor_name, data_name, trait_name,
                 X_data=None,
                 trait=None,
                 predictor_obj=None,
                 predictor_scoring_fun=None,
                 score_cutoff_fun=None,
                 n_features_dependent_parameters=None,
                 constant_parameters=None,
                 is_categorical_trait=None,
                 predictor_dataset_manager=None,
                 predictor_config_manager=None,
                 feature_renamer=None,
                 groupby=None, color=None, pooled=None, order=None,
                 violinplot_kws=None, data_type=None):

        self.predictor_name = predictor_name
        self.data_name = data_name
        self.trait_name = trait_name

        self.feature_renamer = feature_renamer
        self.groupby = groupby
        self.color = color
        self.pooled = pooled
        self.order = order
        self.violinplot_kws = violinplot_kws
        self.data_type = data_type

        if trait is not None:
            trait = trait.copy()
            trait.name = trait_name

        if predictor_dataset_manager is None:
            if predictor_config_manager is None:
                self.predictor_config_manager = PredictorConfigManager()
            else:
                self.predictor_config_manager = predictor_config_manager

            self.predictor_data_manager = PredictorDataSetManager(
                self.predictor_config_manager)
        else:
            self.predictor_data_manager = predictor_dataset_manager

        #load all args and kwargs into instance attributes

        self._data = X_data
        self.trait = trait
        self.predictor_obj = predictor_obj
        self.predictor_scoring_fun = predictor_scoring_fun
        self.score_cutoff_fun = score_cutoff_fun
        self.constant_parameters = constant_parameters
        self.n_features_dependent_parameters = n_features_dependent_parameters
        self.categorical_trait = is_categorical_trait if \
            is_categorical_trait is not None else False

        self.__doc__ = '{}\n\n{}\n\n{}\n\n'.format(self.__doc__,
                                                     self.dataset.__doc__,
                                                     self.predictor.__doc__)

    #thin reference to self.dataset
    @property
    def dataset(self):
        return self.predictor_data_manager.dataset(
            self.data_name, self.trait_name, data=self._data, trait=self.trait,
            categorical_trait=self.categorical_trait)

    @property
    def X(self):
        """predictive variables, aligned with target"""
        return self.dataset.X

    @property
    def y(self):
        """target variable, aligned with predictive variables"""
        return self.dataset.y

    #thin reference to self.predictor

    @property
    def predictor(self):
        return self.dataset.predictor(self.predictor_name,
                                      obj=self.predictor_obj,
                                      predictor_scoring_fun=self.predictor_scoring_fun,
                                      score_cutoff_fun=self.score_cutoff_fun,
                                      n_features_dependent_parameters=self.n_features_dependent_parameters,
                                      constant_parameters=self.constant_parameters)

    def fit(self):
        """Fit predictor to the dataset
        """
        sys.stdout.write(
            "Fitting a predictor for X:{}, y:{}, method:{}... please wait.\n"
            .format(self.dataset.data_name,
                    self.dataset.trait_name,
                    self.predictor_name))

        self.predictor.fit(self.dataset.X, self.dataset.y)
        self.has_been_fit = True
        sys.stdout.write("\tFinished.\n")
        #Collect scores from predictor, rename innate scores variable to self.scores_
        scores = self.predictor.predictor_scoring_fun(self.predictor)
        self.scores_ = pd.Series(index=self.X.columns, data=scores)
        self.has_been_scored = True

    @memoize
    def predict(self, other):
        if not type(other) == pd.DataFrame:
            raise TypeError("please predict on a DataFrame")
        other_aligned, _ = other.align(self.X, axis=1, join='right').fillna(0)
        sys.stderr.write("predicting value, there are \
                         {} common and {} not-common features.".format(
            len(set(other.columns) and self.X.columns),
            len(other.columns and not self.X.columns)))
        return pd.Series(self.predictor.predict(other_aligned.values),
                         index=other.index)

    @property
    def oob_score_(self):
        return self.predictor.oob_score_

    @property
    def has_been_fit(self):
        return self.predictor.has_been_fit

    @has_been_fit.setter
    def has_been_fit(self, value):
        self.predictor.has_been_fit = value

    @property
    def has_been_scored(self):
        return self.predictor.has_been_scored

    @has_been_scored.setter
    def has_been_scored(self, value):
        self.predictor.has_been_scored = value

    @property
    def score_coefficient(self):
        return self.predictor._score_coefficient

    @score_coefficient.setter
    def score_coefficient(self, value):
        self.predictor._score_coefficient = value

    @property
    def scores_(self):
        return self.predictor.scores_

    @scores_.setter
    def scores_(self, value):
        self.predictor.scores_ = value
        if self.n_good_features_ <= 1:
            sys.stderr.write("cutoff: %.4f\n" % self.score_cutoff_)
            sys.stderr.write("WARNING: These classifier settings produced "
                             "<= 1 important feature, consider reducing score_coefficient. "
                             "PCA will fail with this error: \"ValueError: failed "
                             "to create intent(cache|hide)|optional array-- must have defined "
                             "dimensions but got (0,)\"\n")

    @property
    def score_cutoff_(self):
        return self.predictor.score_cutoff_fun(self.scores_,
                                               self.score_coefficient)

    @property
    def important_features_(self):
        return self.scores_ > self.score_cutoff_

    @property
    def subset_(self):
        return self.X.T[self.important_features_].T

    @property
    def n_good_features_(self):
        return np.sum(self.important_features_)

    @memoize
    def pca(self, **plotting_kwargs):
        return PCAViz(self.subset_,
                      **plotting_kwargs)

    @memoize
    def nmf(self, **plotting_kwargs):
        return NMFViz(self.subset_,
                      **plotting_kwargs)


class Regressor(PredictorBase):
    __doc__ = "Regressor for continuous response variables \n" + PredictorBase.__doc__


    def __init__(self, data_name, trait_name,
                 predictor_name=None,
                 *args, **kwargs):
        if predictor_name is None:
            predictor_name = default_regressor
        kwargs['is_categorical_trait'] = False
        super(Regressor, self).__init__(predictor_name, data_name, trait_name,
                                        *args, **kwargs)


class Classifier(PredictorBase):
    __doc__ = "Classifier for categorical response variables.\n" + PredictorBase.__doc__

    def __init__(self, data_name, trait_name,
                 predictor_name=None,
                 *args, **kwargs):
        if predictor_name is None:
            predictor_name = default_classifier
        kwargs['is_categorical_trait'] = True
        super(Classifier, self).__init__(predictor_name, data_name, trait_name,
                                         *args, **kwargs)
