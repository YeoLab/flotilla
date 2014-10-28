import sklearn
import pandas as pd

class OutlierDetection(object):


    """
    Detect outliers. Uses OneClassSVM by default
    """


    def __init__(self, X,
                 outlier_detection_method=None,
                 outlier_detection_method_kwargs=None,
                 ):
        """

        :param X: data on which to detect outliers
        :param outlier_detection_method: method for outlier detection that takes .fit(X) and has a .predict(X) method
        :param outlier_detection_method_kwargs: a dictionary of args to be passed to outlier_detection_method
        :return:
        """
        if outlier_detection_method is None:
            outlier_detection_method = sklearn.svm.OneClassSVM

        if outlier_detection_method_kwargs is None:
            outlier_detection_method_kwargs = {}

        outlier_kwargs = {'nu':0.1,
                          'kernel':"rbf",
                          'gamma':0.1}

        outlier_kwargs.update(outlier_detection_method_kwargs)
        self.outlier_detector_parameters = outlier_kwargs
        self.outlier_kwargs = outlier_kwargs
        self.outlier_detector = outlier_detection_method(**outlier_kwargs)
        self.X = X
        self.outlier_detector.fit(self.X)

    def predict(self, X=None):
        """

        :param X: data to predict using outlier_detection_method established at init.
        If X is None, return outlier prediciton for input data
        :return: outliers
        """
        if X is None:
            self.outliers = pd.Series(self.outlier_detector.predict(self.X) == -1, index=X.index)
            return self.outliers
        else:
            self.outliers = pd.Series(self.outlier_detector.predict(X.fillna(0)) == -1, index=X.index)
            return self.outliers