import sklearn
import pandas as pd


class OutlierDetection(object):
    """

    """


    def __init__(self, X,
                 outlier_detection_method=None,
                 outlier_detection_method_kwargs=None,
    ):
        """Detect outliers. Uses OneClassSVM as the default outlier detection method.

        Parameters
        ----------
        X : pandas.DataFrame
            A samples x features dataframe.
        outlier_detection_method : An object that implements fit, fit_transform and transform, or None.
            If None, use default: sklearn.svm.OneClassSVM
        outlier_detection_method_kwargs : kwargs to send to outlier_detection_method, or None.
            If None, use default: {'nu': 0.1,
                                   'kernel': "rbf",
                                   'gamma': 0.1,
                                   'random_state': 2014}

        Returns
        -------
        None, but sets self.outliers
        """

        if outlier_detection_method is None:
            outlier_detection_method = sklearn.svm.OneClassSVM

        if outlier_detection_method_kwargs is None:
            outlier_detection_method_kwargs = {}

        outlier_kwargs = {'nu': 0.1,
                          'kernel': "rbf",
                          'gamma': 0.1,
                          'random_state': 2014}

        outlier_kwargs.update(outlier_detection_method_kwargs)
        self.outlier_detector_parameters = outlier_kwargs
        self.outlier_kwargs = outlier_kwargs
        self.outlier_detector = outlier_detection_method(**outlier_kwargs)
        self.X = X
        self.outlier_detector.fit(self.X)

    def predict(self, X=None):
        """
        Parameters
        ----------
        X : samples x features data to predict outliers with existing outlier_detection_method,
            set in __init or None. If None, return outlier prediciton for input data

        Returns
        -------
        outliers : pd.Series of predict output from outlier_detection_method.predict, with the same index as X
        """
        if X is None:
            self.outliers = pd.Series(
                self.outlier_detector.predict(self.X) == -1, index=X.index)
            return self.outliers
        else:
            self.outliers = pd.Series(
                self.outlier_detector.predict(X.fillna(0)) == -1, index=X.index)
            return self.outliers