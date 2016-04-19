"""
Detect outlier samples in data
"""
import sys

import sklearn
import pandas as pd


class OutlierDetection(object):

    """Construct an outlier detection object

    Parameters
    ----------
    X : pandas.DataFrame
        A (n_samples, n_features) dataframe, where the outliers will be
        detected from the rows (the samples)
    method : sklearn classifier, optional
        If None, defaults to OneClassSVM. The method class must have both
        method.fit() and method.predict() methods
    nu : float, optional (default 0.1)
        An upper bound on the fraction of training errors and a lower
        bound of the fraction of support vectors. Should be in the
        interval (0, 1]. By default 0.5 will be taken.
    kernel : str, optional (default='rbf')
        The kernel to be used by the outlier detection algorihthm
    gamma : float, optional (default=0.1)
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma is
        0.0 then 1/n_features will be used instead.
    random_state : int, optional (default=0)
        Random state of the method, for reproducibility.
    kwargs : other keyword arguments, optional
        All other keyword arguments are passed to method()
    """
    def __init__(self, X, method=None, nu=0.1, kernel='rbf', gamma=0.1,
                 random_state=0, **kwargs):
        if method is None:
            method = sklearn.svm.OneClassSVM

        sys.stdout.write('SVM Kernel is: {}\n'.format(kernel))

        kwargs.update(dict(nu=nu, kernel=kernel, gamma=gamma,
                           random_state=random_state))
        self.kwargs = kwargs
        self.outlier_detector = method(**kwargs)
        self.X = X
        self.outlier_detector.fit(self.X)

    def predict(self, X=None):

        """Predict which samples are outliers

        Parameters
        ----------
        X : pandas.DataFrame, optional (default None)
            A (n_samples, n_features) Dataframe. If None, predict outliers of
            the original input data, where the new data has the same number of
            features as the original data. Otherwise, use the original input
            data to detect outliers on this new data.

        Returns
        -------
        outliers : pandas.Series
            A boolean
        """
        X = X if X is not None else self.X
        self.outliers = pd.Series(
            self.outlier_detector.predict(X.fillna(0)) == -1,
            index=X.index)
        # TODO: Since you can run this on self.X OR new X, then "self.outliers"
        # can change and not be consistent....... this is a problem
        return self.outliers
