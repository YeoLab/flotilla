"""
Compute predictors on data, e.g. classify or regress on features/samples
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
import pandas.util.testing as pdt

from ..util import memoize, timestamp
from .decomposition import DataFramePCA


CLASSIFIER = 'ExtraTreesClassifier'
REGRESSOR = 'ExtraTreesRegressor'
SCORE_COEFFICIENT = 2


def default_predictor_scoring_fun(cls):
    """Return scores of how important a feature is to the prediction

    Most predictors score output coefficients in the variable
    cls.feature_importances_ and others may use another name for scores, so
    this function bridges the gap

    Parameters
    ----------
    cls : sklearn predictor
        A scikit-learn prediction class, such as ExtraTreesClassifier or
        ExtraTreesRegressor

    Returns
    -------
    scores : pandas.Series
        A (n_features,) size series of how important each feature was to the
        classification (bigger is better)
    """
    return cls.feature_importances_


def default_score_cutoff_fun(arr, std_multiplier=SCORE_COEFFICIENT):
    """Calculate a minimum score cutoff for the best features

    By default, this function calculates: :math:`f(x) = mean(x) + 2 * std(x)`

    Parameters
    ----------
    arr : numpy.ndarray
        A numpy array of scores
    std_multiplier : float, optional (default=2)
        What to multiply the standard deviation by. E.g. if you want only
        features that are 6 standard deviations away, set this to 6.

    Returns
    -------
    cutoff : float
        Minimum score of "best" features, given these parameters
    """
    return np.mean(arr) + std_multiplier * np.std(arr)


class PredictorConfig(object):
    """A configuration for a predictor, names and tracks/sets parameters

    Dynamically configures some args for predictor based on n_features
    (if this attribute exists)
    set general parameters with __init__
    yield instances, set by your parameters, with __call__
    """

    def __init__(self, predictor_name, obj,
                 predictor_scoring_fun=default_predictor_scoring_fun,
                 score_cutoff_fun=default_score_cutoff_fun,
                 n_features_dependent_kwargs=None,
                 **kwargs):
        """Construct a predictor configuration

        Parameters
        ----------
        predictor_name : str
            A name for this predictor
        obj : sklearn predictor
            A scikit-learn predictor, eg sklearn.ensemble.ExtraTreesClassifier
        predictor_scoring_fun : function, optional
            A function which returns the scores of a predictor. May be
            different for different predictor objects
        score_cutoff_fun : function, optional
            A function which returns the minimum "good" score of a predictor
        n_features_dependent_kwargs : dict, optional (default None)
            A dictionary of (key, function) arguments for the classifier, for
            keyword arguments that are dependent on the dataset input size
        kwargs : other keyword arguments, optional
            All other keyword arguments are passed along to the predictor
        """
        if n_features_dependent_kwargs is None:
            n_features_dependent_kwargs = {}
        self.n_features_dependent_kwargs = n_features_dependent_kwargs
        self.constant_kwargs = kwargs
        self.predictor_scoring_fun = predictor_scoring_fun
        self.score_cutoff_fun = score_cutoff_fun
        self.predictor_name = predictor_name
        sys.stdout.write(
            "{}\tPredictor {} is of type {}\n".format(timestamp(),
                                                      self.predictor_name,
                                                      obj))
        self._parent = obj
        self.__doc__ = obj.__doc__
        sys.stdout.write(
            "{}\tAdded {} to default predictors\n".format(timestamp(),
                                                          self.predictor_name))

    @memoize
    def parameters(self, n_features):
        """Given a number of features, return the appropriately scaled keyword
        arguments

        Parameters
        ----------
        n_features : int
            Number of features in the data to scale appropriate keyword
            arguments to the predictor object

        """
        kwargs = {}
        for parameter, setter in self.n_features_dependent_kwargs.items():
            kwargs[parameter] = setter(n_features)

        for parameter, value in self.constant_kwargs.items():
            kwargs[parameter] = value

        return kwargs

    def __call__(self, n_features):
        """Initialize a predictor with this number of features

        Parameters
        ----------
        n_features : int
            The number of features in the data

        Returns
        -------
        predictor : sklearn predictor
            A scikit-learn predictor object inialized with keyword arguments
            specified in __init__, and the
            :py:attr:`n_feature_dependent_kwargs` scaled to this number of
            features
        """
        parameters = self.parameters(n_features)

        sys.stdout.write(
            "{} Configuring predictor type: {} with {} features".format(
                timestamp(), self.predictor_name, n_features))

        predictor = self._parent(**parameters)
        predictor.score_cutoff_fun = self.score_cutoff_fun
        predictor.predictor_scoring_fun = self.predictor_scoring_fun
        predictor.has_been_fit = False
        predictor.has_been_scored = False
        predictor._score_coefficient = SCORE_COEFFICIENT
        return predictor


class PredictorConfigScalers(object):
    """Scale parameters specified in the keyword arugments based on the
    dataset size
    """

    _default_coef = 2.5
    _default_nfeatures = 500

    @staticmethod
    def max_feature_scaler(n_features=_default_nfeatures, coef=_default_coef):
        """Scale the maximum number of features per estimator

        # TODO: @mlovci what are the principles behind this scaler? to see each
        feature "x" number of times?

        Parameters
        ----------
        n_features : int, optional (default 500)
            Number of features in the data
        coef : float, optional (default 2.5)
            # TODO: What does this do?

        Returns
        -------
        n_features : int
            Maximum number of features per estimator

        Raises
        ------
        ValueError
            If n_features is None
        """
        if n_features is None:
            raise ValueError
        return int(math.ceil(np.sqrt(np.sqrt(n_features) ** (coef + .3))))

    @staticmethod
    def n_estimators_scaler(n_features=_default_nfeatures, coef=_default_coef):
        """Scale the number of estimators based on the input features

        # TODO: @mlovci what are the principles behind this scaler? to see each
        feature "x" number of times?

        Parameters
        ----------
        n_features : int, optional (default 500)
            Number of features in the data
        coef : float, optional (default 2.5)
            # @mlovci TODO: What does this do?

        Returns
        -------
        n_estimators : int
            Number of estimators to use

        Raises
        ------
        ValueError
            If n_features is None
        """
        if n_features is None:
            raise ValueError
        return int(math.ceil((n_features / 50.) * coef))

    @staticmethod
    def n_jobs_scaler(n_features=_default_nfeatures):
        """Scale the number of jobs based on how many features are in the data

        # TODO: @mlovci what are the principles behind this scaler? to see each
        feature "x" number of times?

        Parameters
        ----------
        n_features : int
            Number of features in the data

        Returns
        -------
        n_jobs : int
            Number of jobs to use

        Raises
        ------
        ValueError
            If n_features is None
        """
        if n_features is None:
            raise ValueError

        return int(min(4, math.ceil(n_features / 2000.)))


class ConfigOptimizer(object):
    """choose the coef that makes some result most likely at all n_features
    (or some other function of the dataset)
    """

    @staticmethod
    def objective_average_times_seen(
            n_features, coef=PredictorConfigScalers._default_coef,
            max_feature_scaler=PredictorConfigScalers.max_feature_scaler,
            n_estimators_scaler=PredictorConfigScalers.n_estimators_scaler):
        """I have no idea what this does. @mlovci

        Parameters
        ----------
        n_features : int
            ???
        coef : float
            ???
        max_feature_scaler : function
            ???
        n_estimators_scaler : function
            ???

        Returns
        -------
        ???
        """
        return ((n_features / max_feature_scaler(n_features, coef)) *
                n_estimators_scaler(n_features, coef)) / float(n_features)


class PredictorConfigManager(object):
    """Manage several predictor configurations

    A container for predictor configurations, includes several built-ins
    @mlovci: built-ins such as ........ ?
    What is predictor_config vs new_predictor_config? Why are they separate?

    Attributes
    ----------
    predictor_config :

    predictor_configs :

    builtin_predictor_configs :

    Methods
    -------
    new_predictor_config
        Create a new predictor configuration


    >>> pcm = PredictorConfigManager()
    >>> # add a new type of predictor
    >>> pcm.new_predictor_config(ExtraTreesClassifier, 'ExtraTreesClassifier',
    ...                          n_features_dependent_kwargs=
    ...     {'max_features': PredictorConfigScalers.max_feature_scaler,
    ...      'n_estimators': PredictorConfigScalers.n_estimators_scaler,
    ...      'n_jobs': PredictorConfigScalers.n_jobs_scaler},
    ...                          bootstrap=True, random_state=0,
    ...                                           oob_score=True,
    ...                                           verbose=True})
    """

    def __init__(self):
        """Construct a predictor configuration manager with
        ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
        and GradientBoostingRegressor as default predictors.
        """

        constant_extratrees_kwargs = {'bootstrap': True,
                                      'random_state': 0,
                                      'oob_score': True,
                                      'verbose': True}

        self.predictor_config(
            'ExtraTreesClassifier', obj=ExtraTreesClassifier,
            n_features_dependent_kwargs={
                'max_features': PredictorConfigScalers.max_feature_scaler,
                'n_estimators': PredictorConfigScalers.n_estimators_scaler,
                'n_jobs': PredictorConfigScalers.n_jobs_scaler},
            **constant_extratrees_kwargs)

        self.predictor_config(
            'ExtraTreesRegressor', obj=ExtraTreesRegressor,
            n_features_dependent_kwargs={
                'max_features': PredictorConfigScalers.max_feature_scaler,
                'n_estimators': PredictorConfigScalers.n_estimators_scaler,
                'n_jobs': PredictorConfigScalers.n_jobs_scaler},
            **constant_extratrees_kwargs)

        constant_boosting_kwargs = {'n_estimators': 80, 'max_features': 1000,
                                    'learning_rate': 0.2, 'subsample': 0.6, }

        self.predictor_config('GradientBoostingClassifier',
                              obj=GradientBoostingClassifier,
                              **constant_boosting_kwargs)

        self.predictor_config('GradientBoostingRegressor',
                              obj=GradientBoostingRegressor,
                              **constant_boosting_kwargs)

    @property
    def builtin_predictor_configs(self):
        """Names of the predictor configurations
        """
        return self.predictor_configs.keys()

    @property
    def predictor_configs(self):
        """Dict of predictor configurations
        """
        if not hasattr(self, '_predictors'):
            self._predictors = {}

        return self._predictors

    def predictor_config(self, name, **kwargs):
        """Create a new predictor configuration, added to
        :py:attr:`.predictors`

        Parameters
        ----------
        name : str
            Name of the predictor
        kwargs : other keyword arguments, optional
            All other keyword arguments are passed to
            :py:meth:`predictor_configs`

        Returns
        -------
        predictor : sklearn predictor
            An initalized scikit-learn predictor
        """
        predictor = self.new_predictor_config(name, **kwargs)
        if name in self.predictor_configs and \
                self.predictor_configs[name] != predictor:
            sys.stderr.write(
                "WARNING: over-writing predictor named: {}".format(name))
        self.predictor_configs[name] = predictor
        return predictor

    @memoize
    def new_predictor_config(self, name, obj=None,
                             predictor_scoring_fun=None,
                             score_cutoff_fun=None,
                             n_features_dependent_kwargs=None,
                             **kwargs):
        """Create a new predictor configuration

        Parameters
        ----------
        name : str
            Name of the predictor configuration
        obj : sklearn predictor object, optional (default=None)
            @mlovci: what is the point of setting the default to None if it's
            not really allowed?
        predictor_scoring_fun : function, optional (default=None)
            If None, get feature scores from obj.feature_importances_
        score_cutoff_fun : function, optional (default=None)
            If None, get the cutoff for important features with by taking
            features with scores that are 2 standard deviations away from the
            mean score
        n_features_dependent_kwargs : dict, optional (default=None)
            A (key, function) dictionary of keyword argument names and
            functions which scale their values based on the dataset input size
        kwargs : other keyword arguments
            All other keyword arguments are passed to
            :py:class:`PredictorConfig`

        Returns
        -------
        predictorconfig : PredictorConfig
            A predictor configuration

        Raises
        ------
        ValueError
            If `obj` is None and any of the other keyword arguments are None
        KeyError
            If `obj` is None and "name" is not already in
            :py:attr:`.predictor_configs`
        """
        if obj is None:
            # If obj is None, then this is probably just a "name" and you can't
            # change any of the parameters
            n_features_dependent_kwargs = None if \
                n_features_dependent_kwargs == {} else \
                n_features_dependent_kwargs
            kwargs = None if kwargs == {} else kwargs
            args = [predictor_scoring_fun, score_cutoff_fun,
                    n_features_dependent_kwargs, kwargs]
            if any([i is not None for i in args]):
                # if obj is None, you'd better not be asking to set parameters
                # on it.
                raise ValueError

            try:
                return self.predictor_configs[name]
            except KeyError:
                raise KeyError("No such predictor: {}".format(name))

        if predictor_scoring_fun is None:
            predictor_scoring_fun = default_predictor_scoring_fun

        if score_cutoff_fun is None:
            score_cutoff_fun = default_score_cutoff_fun

        if n_features_dependent_kwargs is not None:
            if type(n_features_dependent_kwargs) is not dict:
                raise TypeError
        else:
            n_features_dependent_kwargs = {}

        return PredictorConfig(
            name, obj, predictor_scoring_fun=predictor_scoring_fun,
            score_cutoff_fun=score_cutoff_fun,
            n_features_dependent_kwargs=n_features_dependent_kwargs,
            **kwargs)


class PredictorDataSet(object):
    def __init__(self, data, trait,
                 data_name="MyDataset",
                 categorical_trait=False,
                 predictor_config_manager=None):
        """Store a (n_samples, n_features) matrix and (n_samples,) trait pair

        In scikit-learn parlance, store an X (data of independent variables)
        and y (target prediction) pair

        Parameters
        ----------
        data : pandas.DataFrame
            A (n_samples, n_features) datafarme
        trait : pandas.Series



        Returns
        -------


        Raises
        ------

         data - X
         trait - y
         data_name - name to store this dataset, to be used with trait.name
         categorical_trait - is y categorical?
        """

        if not isinstance(trait, pd.Series):
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
        self._predictors = defaultdict(dict)

    @property
    def X(self):
        """(n_samples, n_features) matrix"""
        return self._data.align(self._y, axis=0,
                                join='inner')[0]

    @property
    def y(self):
        """(n_samples,) vector of traits"""
        return self._data.align(self._y, axis=0,
                                join='inner')[1]

    @property
    def traitset(self):
        """All unique values in :py:attr:`self.trait`"""
        return self.trait.groupby(self.trait).groups.keys()

    @property
    def predictors(self):
        """dict of PredictorConfig instances

        The idea here is to keep the predictors tied to their datasets
        """
        if hasattr(self, '_predictors'):
            return self._predictors

    @memoize
    def predictor(self, name, **kwargs):
        """A single, initialized PredictorConfig instance

        Parameters
        ----------
        name : str
            Name of the predictor to retrieve or initialize
        kwargs : other keyword arguments
            All other keyword arguments are passed to
            :py:class:`PredictorConfig`

        Returns
        -------
        predictorconfig : PredictorConfig
            An initialized scikit-learn classifier or regressor
        """
        predictor = self.predictor_config_manager.predictor_config(name,
                                                                   **kwargs)
        initialized = predictor(self.n_features)
        self.predictors[name] = initialized
        return initialized

    def check_if_equal(self, data, trait, categorical_trait):
        """Check if this is the same as another dataset.

        Parameters
        ----------
        data : pandas.DataFrame
            Input data of another dataset
        trait : pandas.Series
            Response variable of another dataset
        categorical_trait : bool
            Whether or not ``trait`` is categorical

        Raises
        ------
        AssertionError
            If datasets are not the same
        """
        pdt.assert_frame_equal(data, self._data)
        pdt.assert_series_equal(trait, self.trait)
        pdt.assert_equal(categorical_trait, self.categorical_trait)


class PredictorDataSetManager(object):
    """A collection of PredictorDataSet instances.

    Parameters
    ----------
    predictor_config_manager : PredictorConfigManager, optional (default None)
        A predictor configuration manager. If None, instantiate a new one.

    Attributes
    ----------
    datasets : dict
        Dict of dicts of {data: {trait: {categorical: dataset}}}. For
        convenient retrieval of predictors
    """

    def __init__(self, predictor_config_manager=None):
        self.predictor_config_manager = predictor_config_manager \
            if predictor_config_manager is not None \
            else PredictorConfigManager()

    @property
    def datasets(self):
        """3-layer deep dict of {data: {trait: {categorical: dataset}}}
        """
        if not hasattr(self, '_datasets'):
            # 3 layer deep (data, trait, categorical?)
            # will almost always be either categorical true or false, rarely
            # both
            self._datasets = defaultdict(lambda: defaultdict(dict))
        return self._datasets

    def dataset(self, data_name, trait_name, categorical_trait=False,
                **kwargs):
        """???? @mlovci please fill in

        Parameters
        ----------
        data_name : str
            Name of this data
        trait_name : str
            Name of this trait
        categorical_trait : bool, optional (default=False)
            If True, then this trait is treated as a categorical, rather than a
            sequential trait

        Returns
        -------
        dataset : PredictorDataSet
            ???
        """
        kwargs['categorical_trait'] = categorical_trait
        dataset = self.new_dataset(data_name, trait_name, **kwargs)

        if data_name in self.datasets:
            if trait_name in self.datasets[data_name]:
                if categorical_trait in self.datasets[data_name][
                        trait_name] and \
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
        """??? Difference betwen this and ``dataset``??? @mlovci

        Parameters
        ----------
        data_name : str
            Name of this data
        trait_name : str
            Name of this trait
        categorical_trait : bool, optional (default=False)
            If True, then this trait is treated as a categorical, rather than a
            sequential trait
        data : pandas.DataFrame, optional (default=None)
            ??? WHy is this optional!?!??!?!
        trait : pandas.Series, optional (default=None)
            ???? Why is this optional!?!?!?
        predictor_config_manager : PredictorConfigManager (default=None)

        Returns
        -------
        dataset : PredictorDataSet
            ???
        """

        if data is None:
            # try to get this dataset by key in the dictionary
            args = np.array([data, trait, predictor_config_manager])
            if np.any([i is not None for i in args]):
                # if data is None, you'd better not be asking to set other
                # parameters
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

        predictor_config_manager = predictor_config_manager \
            if predictor_config_manager is not None \
            else self.predictor_config_manager

        return PredictorDataSet(
            data, trait, data_name, categorical_trait=categorical_trait,
            predictor_config_manager=predictor_config_manager)


class PredictorBase(object):
    def __init__(self, predictor_name, data_name, trait_name,
                 X_data=None,
                 trait=None,
                 predictor_obj=None,
                 predictor_scoring_fun=None,
                 score_cutoff_fun=None,
                 n_features_dependent_kwargs=None,
                 constant_kwargs=None,
                 is_categorical_trait=None,
                 predictor_dataset_manager=None,
                 predictor_config_manager=None,
                 feature_renamer=None,
                 groupby=None, color=None, pooled=None, order=None,
                 violinplot_kws=None, data_type=None,
                 label_to_color=None, label_to_marker=None,
                 singles=None, outliers=None):
        """A dataset-predictor pair from PredictorDatasetManager

        One datset, one predictor, from dataset manager.


        Parameters
        ----------
        predictor_name : str
            Name for predictor
        data_name : str
            Name for this (subset of the) data
        trait_name : str
            Name for this trait
        X_data : pandas.DataFrame, optional
            Samples-by-features (row x col) dataset to train the predictor on
        trait : pandas.Series, optional
            A variable you want to predict using X_data. Indexed like X_data.
        predictor_obj : sklearn predictor, optional
            A scikit-learn predictor that implements fit and score on
            (X_data,trait) Default ExtraTreesClassifier
        predictor_scoring_fun : function, optional
            Function to get the feature scores for a scikit-learn classifier.
            This can be different for different classifiers, e.g. for a
            classifier named "x" it could be x.scores_, for other it's
            x.feature_importances_. Default: lambda x: x.feature_importances_
        score_cutoff_fun : function, optional
            Function to cut off insignificant scores
            Default: lambda scores: np.mean(x) + 2 * np.std(x)
        n_features_dependent_kwargs : dict, optional
            kwargs to the predictor that depend on n_features
            Default: {}
        constant_kwargs : dict, optional
            kwargs to the predictor that are constant, i.e.:
            {'n_estimators': 100, 'bootstrap': True, 'max_features': 'auto',
            'random_state': 0, 'oob_score': True, 'n_jobs': 2, 'verbose': True}
        """

        self.predictor_name = predictor_name
        self.data_name = data_name
        self.trait_name = trait_name

        self.feature_renamer = feature_renamer
        self.groupby = groupby
        self.color = color
        self.pooled = pooled
        self.singles = singles
        self.outliers = outliers
        self.order = order
        self.violinplot_kws = violinplot_kws
        self.data_type = data_type
        self.label_to_color = label_to_color
        self.label_to_marker = label_to_marker

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

        # load all args and kwargs into instance attributes

        self._data = X_data
        self.trait = trait
        self.predictor_obj = predictor_obj
        self.predictor_scoring_fun = predictor_scoring_fun
        self.score_cutoff_fun = score_cutoff_fun
        self.constant_kwargs = {} if constant_kwargs is None \
            else constant_kwargs
        self.n_features_dependent_kwargs = {} \
            if n_features_dependent_kwargs is None else \
            n_features_dependent_kwargs
        self.categorical_trait = is_categorical_trait if \
            is_categorical_trait is not None else False

        self.__doc__ = '{}\n\n{}\n\n{}\n\n'.format(self.__doc__,
                                                   self.dataset.__doc__,
                                                   self.predictor.__doc__)

    @property
    def dataset(self):
        """Thin reference to `dataset`"""
        return self.predictor_data_manager.dataset(
            self.data_name, self.trait_name, data=self._data, trait=self.trait,
            categorical_trait=self.categorical_trait)

    @property
    def X(self):
        """Predictive variables, aligned with target.

        Thin reference to `dataset.X`
        """
        return self.dataset.X

    @property
    def y(self):
        """Target variable, aligned with predictive variables

        Thin reference to `dataset.y`
        """
        return self.dataset.y

    @property
    def predictor(self):
        """Thin reference to ``dataset.predictor``"""
        return self.dataset.predictor(
            self.predictor_name, obj=self.predictor_obj,
            predictor_scoring_fun=self.predictor_scoring_fun,
            score_cutoff_fun=self.score_cutoff_fun,
            n_features_dependent_kwargs=self.n_features_dependent_kwargs,
            **self.constant_kwargs)

    def fit(self):
        """Fit predictor to the dataset"""
        sys.stdout.write(
            "Fitting a predictor for X:{}, y:{}, method:{}... please wait.\n"
            .format(self.dataset.data_name,
                    self.dataset.trait_name,
                    self.predictor_name))

        self.predictor.fit(self.dataset.X, self.dataset.y)
        self.has_been_fit = True
        sys.stdout.write("\tFinished.\n")
        # Collect scores from predictor, rename innate scores variable to
        # self.scores_
        scores = self.predictor.predictor_scoring_fun(self.predictor)
        self.scores_ = pd.Series(index=self.X.columns, data=scores)
        self.has_been_scored = True

    @memoize
    def predict(self, other):
        """Predict

        Parameters
        ----------
        other : pandas.DataFrame
            Given a (m_samples, n_features) dataframe, predict the response

        Returns
        -------
        prediction : pandas.Series
            (m_samples,) sized series of prediction of response

        Raises
        ------
        TypeError
            If ``other`` is not a pandas DataFrame
        """
        if not isinstance(other, pd.DataFrame):
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
        """Thin reference to `predictor.oob_score_`"""
        return self.predictor.oob_score_

    @property
    def has_been_fit(self):
        """Thin reference to `predictor.has_been_fit`"""
        return self.predictor.has_been_fit

    @has_been_fit.setter
    def has_been_fit(self, value):
        """Set whether the predictor has been fit"""
        self.predictor.has_been_fit = value

    @property
    def has_been_scored(self):
        """Thin reference to :py:attr:`.predictor.has_been_scored`"""
        return self.predictor.has_been_scored

    @has_been_scored.setter
    def has_been_scored(self, value):
        """Set whether the predictor has been scored"""
        self.predictor.has_been_scored = value

    @property
    def score_coefficient(self):
        """Thin reference to ``predictor._score_coefficient``"""
        return self.predictor._score_coefficient

    @score_coefficient.setter
    def score_coefficient(self, value):
        """Set the predictor's score coefficient"""
        self.predictor._score_coefficient = value

    @property
    def scores_(self):
        """Scores of these features' importances in this predictor"""
        return self.predictor.scores_

    @scores_.setter
    def scores_(self, value):
        """Set the predictor scores

        If zero important features found, raise a warning
        """
        self.predictor.scores_ = value
        if self.n_good_features_ <= 1:
            sys.stderr.write("cutoff: %.4f\n" % self.score_cutoff_)
            UserWarning("These classifier settings produced <= 1 important "
                        "feature, consider reducing score_coefficient. "
                        "DataFramePCA will fail with this error: "
                        "\"ValueError: failed to create intent("
                        "cache|hide)|optional array-- must have defined "
                        "dimensions but got (0,)\"\n")

    @property
    def score_cutoff_(self):
        """Get the minimum score of the 'good' features"""
        return self.predictor.score_cutoff_fun(self.scores_,
                                               self.score_coefficient)

    @property
    def important_features_(self):
        """Get all features with scores greater than ``score_cutoff_``"""
        return self.scores_ > self.score_cutoff_

    @property
    def subset_(self):
        """Get the subset of the data with only important features"""
        return self.X.ix[:, self.important_features_]

    @property
    def n_good_features_(self):
        """Get the number of good features"""
        return np.sum(self.important_features_)

    @memoize
    def pca(self):
        """Perform PCA on the top-performing features"""
        return DataFramePCA(self.subset_)



class Regressor(PredictorBase):

    categorical = False

    __doc__ = "Regressor for continuous response variables.\n" + \
              PredictorBase.__init__.__doc__

    def __init__(self, data_name, trait_name,
                 predictor_name=None,
                 *args, **kwargs):
        if predictor_name is None:
            predictor_name = REGRESSOR
        kwargs['is_categorical_trait'] = False
        super(Regressor, self).__init__(predictor_name, data_name, trait_name,
                                        *args, **kwargs)


class Classifier(PredictorBase):

    categorical = True

    __doc__ = "Classifier for categorical response variables.\n" + \
              PredictorBase.__init__.__doc__

    def __init__(self, data_name, trait_name,
                 predictor_name=None,
                 *args, **kwargs):
        if predictor_name is None:
            predictor_name = CLASSIFIER
        kwargs['is_categorical_trait'] = True
        super(Classifier, self).__init__(predictor_name, data_name, trait_name,
                                         *args, **kwargs)
