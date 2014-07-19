"""
Compute predictors on data, i.e. regressors or classifiers



"""
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import math

def default_predictor_scoring_fun(cls):
    return cls.feature_importances_

def default_score_cutoff_fun(arr, std_multiplier=2):
    return np.mean(arr) + std_multiplier * np.std(arr)


class PredictorConfig(object):

    """
    A configuration for a predictor, names and tracks/sets parameters for predictor
    Dynamically configures some args for predictor based on n_features (if this attribute exists)
    """


    def __init__(self, obj, predictor_name=None,
                 predictor_scoring_fun=default_predictor_scoring_fun,
                 score_cutoff_fun=default_score_cutoff_fun,
                 n_features_dependent_parameters=None,
                 constant_parameters=None):

        """


        Parameters
        ==========

        obj - predictor object
        predictor_name - a predictor_name for this predictor
        constant_parameters - dict of (key, value) where values are constants

        Options to set parameters by input data size:

        n_features - number of features this predictor will be predicting (default = 500)
        n_feature_dependent_parameters - dict of (key, setter) where setter \
        is a function that takes n_features as an arg

        """

        if predictor_name is None:
            predictor_name = 'MyPredictor'
        if n_features_dependent_parameters is None:
            n_features_dependent_parameters = {}
        self.n_features_dependent_parameters = n_features_dependent_parameters
        if constant_parameters is None:
            constant_parameters = {}
        self.constant_parameters = constant_parameters
        self.predictor_scoring_fun = predictor_scoring_fun
        self.score_cutoff_fun = score_cutoff_fun
        self.predictor_name = predictor_name
        self._parent = obj
        self.__doc__ = obj.__doc__


    def parameters(self, n_features):
        """

        recommended parameters for this classifier

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
        """initialize predictor object with n_features-dependent settings"""
        if n_features is None:
            raise Exception
        parameters = self.parameters(n_features)

        return self._parent(**parameters)


class PredictorConfigManager(object):

    """
    A container for predictor configurations, includes several built-ins
    """
    #TODO: protect built-ins
    default_ceof = 2.5 #for managing n_estimators and max_features
    @property
    def builtin_predictors(self):
        return self.predictors.keys()

    @property
    def predictors(self):
        if not hasattr(self, '_predictors'):
            self._predictors = {}

        return self._predictors

    @predictors.setter
    def set_predictors(self, value):
        self._predictors = value

    @staticmethod
    def max_feature_scaler(n_features, coef=default_ceof):
        if n_features is None:
            raise Exception
        return int(math.ceil(np.sqrt(np.sqrt(n_features)**coef)))

    @staticmethod
    def n_estimators_scaler(n_features, coef=default_ceof):
        if n_features is None:
            raise Exception
        return int(math.ceil(100*coef))

    @staticmethod
    def n_jobs_scaler(n_features):
        if n_features is None:
            raise Exception

        return int(min(4, math.ceil(n_features/2000.)))

    def get_predictor(self, name, n_features=None):

        return self.predictors[name](n_features)

    def register_predictor(self, obj, name=None,
                           predictor_scoring_fun=None,
                           score_cutoff_fun=None,
                           n_features_dependent_parameters=None,
                           constant_parameters=None,
                           ):

        """add a predictor to self.predictors"""
        if predictor_scoring_fun is None:
            predictor_scoring_fun = default_predictor_scoring_fun
        if score_cutoff_fun is None:
            score_cutoff_fun = default_score_cutoff_fun

        if name in self.predictors:
            sys.stderr.write("warning, this predictor is already registered")

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

        self.predictors[name] = PredictorConfig(obj, predictor_name =name,
                                           predictor_scoring_fun=predictor_scoring_fun,
                                           score_cutoff_fun=score_cutoff_fun,
                                           n_features_dependent_parameters=n_features_dependent_parameters,
                                           constant_parameters=constant_parameters)



    def __init__(self):



        constant_extratrees_params = {'bootstrap': True,
                                     'random_state': 0,
                                     'oob_score': True,
                                     'verbose': True}

        self.register_predictor(ExtraTreesClassifier,
                                'ExtraTreesClassifier',

                                n_features_dependent_parameters={'max_features': self.max_feature_scaler,
                                                                 'n_estimators': self.n_estimators_scaler,
                                                                 'n_jobs': self.n_jobs_scaler},

                                constant_parameters=constant_extratrees_params)

        self.register_predictor(ExtraTreesRegressor,
                                'ExtraTreesRegressor',

                                n_features_dependent_parameters={'max_features': self.max_feature_scaler,
                                                                 'n_estimators': self.n_estimators_scaler,
                                                                 'n_jobs': self.n_jobs_scaler},

                                constant_parameters=constant_extratrees_params)

        constant_boosting_params = {'n_estimators': 80, 'max_features': 1000,
                                                      'learning_rate': 0.2, 'subsample': 0.6, }

        self.register_predictor(GradientBoostingClassifier, 'GradientBoostingClassifier',
                                constant_parameters=constant_boosting_params)

        self.register_predictor(GradientBoostingRegressor, 'GradientBoostingRegressor',
                                constant_parameters=constant_boosting_params)

class PredictorDataSet(object):
    """
    X and y data
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
        return self.trait.groupby(self.trait).describe().index.levels[0]

    @property
    def predictors(self):
        if not hasattr(self, '_predictors'):
            self._predictors = defaultdict(dict)

        return self._predictors

    def post_predictor(self, predictor_name, predictor):
        self.predictors[predictor_name] = predictor


    def __init__(self, data, trait,  data_name="MyDataset", categorical_trait=True):
        """
         data - X
         trait - y
         data_name - name to store this dataset, to be used with trait.name
         categorical_trait - is y categorical?
        """

        if type(trait) != pd.Series:
            raise TypeError("Traits must be pandas.Series objects")

        self.has_been_fit = False
        self.has_been_scored = False
        self.dataset_name = (data_name, trait.name)
        self.data_name = data_name
        self._data = data
        self.trait = trait
        self.trait_name = self.trait.name

        if categorical_trait:

            if len(self.traitset) > 2:
                warnings.warn("WARNING: trait {} has >2 categories".format(
                    self.trait_name))

            # categorical encoder
            le = LabelEncoder().fit(self.traitset)

            # categorical encoding
            self._y = pd.Series(data=le.transform(self.trait),
                               index=self.X.index,
                               name=self.trait.name)

        else:
            self._y = trait

        self.n_features = self.X.shape[1]



class PredictorDataManager(object):

    @property
    def datasets(self):
        if not hasattr(self, '_datasets'):
            self._datasets = defaultdict(dict)
        return self._datasets

    def register_dataset(self, data, trait, data_name, categorical_trait=False):
        dataset = PredictorDataSet(data, trait, data_name, categorical_trait=categorical_trait)
        self.datasets[data_name][dataset.trait_name] = dataset

    def get_dataset(self, data_name, trait_name):
        return self.datasets[data_name][trait_name]


predictor_config_manager = PredictorConfigManager()
predictor_data_manager = PredictorDataManager()


class PredictorBase(object):

    """Train a classifier on data. Remember and organize things.

        Parameters
        ----------


        predictor_name : str
            Name for predictor
        predictor_name : str
            Name for predictor
        data : pandas.DataFrame
            Samples x features data to train the predictor on
        trait : pandas.Series
            Samples x trait (single) data that you want to tell the
            difference between
        predictor_name : str
            Name for predictor combination, can be used to retrieve predictors that are already registered by name
        predictor_obj : scikit-learn classifier
            Which classifier to use. Default ExtraTreesClassifier

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
        predictor_scoring_fun : function
            Function to get the feature scores for a scikit-learn classifier.
            This can be different for different classifiers, e.g. for a
            classifier named "x" it could be x.scores_, for other it's
            x.feature_importances_. Default: lambda x: x.feature_importances_
        score_cutoff_fun : function
            Function to cut off insignificant scores
            Default: lambda scores: np.mean(x) + 2 * np.std(x)


        """

    def __init__(self, predictor_name, dataset_name, trait_name,
                 data=None,
                 trait=None,
                 predictor_obj=None,
                 predictor_scoring_fun=None,
                 score_cutoff_fun=None,
                 n_features_dependent_parameters=None,
                 constant_parameters=None,
                 categorical_trait=None,
                 ):

        if trait is not None:
            trait = trait.copy()
            trait.name = trait_name

        args = np.array([data, trait, predictor_obj, predictor_scoring_fun, score_cutoff_fun,
                         n_features_dependent_parameters, constant_parameters])
        is_custom = np.any(np.array([i is not None for i in args]))
        if is_custom:

            args = np.array([data, trait])
            custom_data = np.any(np.array([i is not None for i in args]))
            if custom_data:
                if np.any(map(lambda x: x is None, args)):
                    raise RuntimeError("Set both trait and data both, or use a pre-registered dataset")

                predictor_data_manager.register_dataset(data, trait, dataset_name, categorical_trait)

            args = np.array([predictor_obj, predictor_scoring_fun, score_cutoff_fun,
                         n_features_dependent_parameters, constant_parameters])

            custom_predictor = np.any(np.array([i is not None for i in args]))
            if custom_predictor:
                if predictor_obj is None:
                    raise ValueError("Set predictor_obj if predictor_name not one of the pre-built predictor types")

                predictor_config_manager.register_predictor(predictor_name, predictor_obj,
                                        predictor_scoring_fun=predictor_scoring_fun,
                                        score_cutoff_fun=score_cutoff_fun,
                                        n_features_dependent_parameters=n_features_dependent_parameters,
                                        constant_parameters=constant_parameters
                                        )
        self.predictor_name = predictor_name
        self.data_name = dataset_name
        self.trait_name = trait_name


    @property
    def predictor(self):
        prd = predictor_config_manager.get_predictor(self.predictor_name, self.data.n_features)
        print "posting predictor"
        self.data.post_predictor(self.predictor_name, prd)
        return prd

    @property
    def data(self):
        return predictor_data_manager.get_dataset(self.data_name, self.trait_name)

    @property
    def X(self):
        return self.data.X

    def fit(self):
        """Fit predictor to the data
        """

        sys.stdout.write("Fitting a predictor for X:{}, y:{}, method:{}... please wait.\n"
                         .format(self.data.data_name, self.data.trait_name, self.predictor_name))
        self.predictor.fit(self.data.X, self.data.y)
        sys.stdout.write("\tFinished.\n")
        self.has_been_fit = True



    def score(self, score_coefficient=None):
        """Collect scores from predictor"""

        if score_coefficient is None:
            score_coefficient = 2

        sys.stdout.write("Scoring predictor for {}... please "
                         "wait\n".format(self.data.dataset_name))

        try:
            scores = self.predictor.predictor_scoring_fun(self.predictor)
        except:
            raise
        self.scores_ = pd.Series(index=self.X.columns, data=scores)
        self.score_cutoff_ = \
            self.predictor.score_cutoff_fun(self.scores_, score_coefficient)

        self.important_features = \
            self.scores_ > self.score_cutoff_

        self.n_good_features_ = \
            np.sum(self.important_features)

        if self.n_good_features_ <= 1:
            sys.stderr.write("cutoff: %.4f\n" % self.score_cutoff_)
            sys.stderr.write("WARNING: These classifier settings produced "
            "<= 1 important feature, consider reducing score_coefficient. "
            "PCA will fail with this error: \"ValueError: failed "
            "to create intent(cache|hide)|optional array-- must have defined "
            "dimensions but got (0,)\"\n")

        self.subset_ = self.X.T[self.important_features].T

        sys.stdout.write("\tFinished.\n")
        self.has_been_scored = True



class Regressor(PredictorBase):

    __doc__ = "Regressor for continuous response variables \n" + PredictorBase.__doc__


    def __init__(self, dataset_name,
                 predictor_name='ExtraTreesRegressor',
                 data=None,
                 trait=None,
                 predictor_obj=None,
                 predictor_scoring_fun=None,
                 score_cutoff_fun=None,
                 n_features_dependent_parameters=None,
                 constant_parameters=None,
                 ):

        super(Regressor, self).__init__(predictor_name, dataset_name,
                                         data=data,
                                         trait=trait,
                                         predictor_obj=predictor_obj,
                                         predictor_scoring_fun=predictor_scoring_fun,
                                         score_cutoff_fun=score_cutoff_fun,
                                         n_features_dependent_parameters=n_features_dependent_parameters,
                                         constant_parameters=constant_parameters,
                                        )



class Classifier(PredictorBase):

    __doc__ = "Classifier for categorical response variables.\n" + PredictorBase.__doc__

    def __init__(self, dataset_name,
                 predictor_name='ExtraTreesClassifier',
                 data=None,
                 trait=None,
                 predictor_obj=None,
                 predictor_scoring_fun=None,
                 score_cutoff_fun=None,
                 n_features_dependent_parameters=None,
                 constant_parameters=None,
                 ):


        super(Classifier, self).__init__(predictor_name=predictor_name, dataset_name =dataset_name,
                                        data=data, trait=trait,
                                        predictor_obj=predictor_obj,
                                        predictor_scoring_fun=predictor_scoring_fun,
                                        score_cutoff_fun=score_cutoff_fun,
                                        n_features_dependent_parameters=n_features_dependent_parameters,
                                        constant_parameters=constant_parameters,
                                        categorical_trait=True
                                        )
