"""
Perform various dimensionality reduction algorithms on data
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import sys

from sklearn import decomposition
import pandas as pd


class DataFrameReducerBase(object):
    """Just like scikit-learn's reducers, but with prettied up DataFrames."""

    def __init__(self, df, n_components=None, **kwargs):
        """Initialize and fit a dataframe to a decomposition algorithm

        Parameters
        ----------
        df : pandas.DataFrame
            A (samples, features) dataframe of data to fit to the reduction
            algorithm
        n_components : int
            Number of components to calculate. If None, use as many
            components as there are samples
        kwargs : keyword arguments
            Any other arguments to the reduction algorithm
        """
        # This magically initializes the reducer like PCA or NMF
        if df.shape[1] <= 3:
            raise ValueError(
                "Too few features (n={}) to reduce".format(df.shape[1]))
        super(DataFrameReducerBase, self).__init__(n_components=n_components,
                                                   **kwargs)
        self.reduced_space = self.fit_transform(df)

    @staticmethod
    def _check_dataframe(X):
        """Check that the input is a pandas dataframe

        Parameters
        ----------
        X : input
            Input to check if this is a pandas dataframe.

        Raises
        ------
        ValueError
            If the input is not a pandas Dataframe

        """
        try:
            assert isinstance(X, pd.DataFrame)
        except AssertionError:
            sys.stdout.write("Try again as a pandas DataFrame")
            raise ValueError('Input X was not a pandas DataFrame, '
                             'was of type {} instead'.format(str(type(X))))

    @staticmethod
    def relabel_pcs(x):
        """Given a list of integers, change the name to be a 1-based
        principal component representation"""
        return "pc_" + str(int(x) + 1)

    def fit(self, X):
        """Perform a scikit-learn fit and relabel dimensions to be
        informative names

        Parameters
        ----------
        X : pandas.DataFrame
            A (n_samples, n_features) Dataframe of data to reduce

        Returns
        -------
        self : DataFrameReducerBase
            A instance of the data, now with components_,
            explained_variance_, and explained_variance_ratio_ attributes

        """
        self._check_dataframe(X)
        self.X = X
        super(DataFrameReducerBase, self).fit(X)
        self.components_ = pd.DataFrame(self.components_,
                                        columns=self.X.columns).rename_axis(
            self.relabel_pcs, 0)
        try:
            self.explained_variance_ = pd.Series(
                self.explained_variance_).rename_axis(self.relabel_pcs, 0)
            self.explained_variance_ratio_ = pd.Series(
                self.explained_variance_ratio_).rename_axis(self.relabel_pcs,
                                                            0)
        except AttributeError:
            pass

        return self

    def transform(self, X):
        """Transform a matrix into the component space

        Parameters
        ----------
        X : pandas.DataFrame
            A (n_samples, n_features) sized DataFrame to transform into the
            current compoment space

        Returns
        -------
        component_space : pandas.DataFrame
            A (n_samples, self.n_components) sized DataFrame transformed into
            component space

        """
        component_space = super(DataFrameReducerBase, self).transform(X)
        self._check_dataframe(X)
        component_space = pd.DataFrame(component_space,
                                       index=X.index).rename_axis(
            self.relabel_pcs, 1)
        return component_space

    def fit_transform(self, X):
        """Perform both a fit and a transform on the input data

        Fit the data to the reduction algorithm, and transform the data to
        the reduced space.

        Parameters
        ----------
        X : pandas.DataFrame
            A (n_samples, n_features) dataframe to both fit and transform

        Returns
        -------
        self : DataFrameReducerBase
            A fit and transformed instance of the object

        Raises
        ------
        ValueError
            If the input is not a pandas DataFrame, will not perform the fit
            and transform

        """
        self._check_dataframe(X)
        self.fit(X)
        return self.transform(X)


class DataFramePCA(DataFrameReducerBase, decomposition.PCA):
    """Perform Principal Components Analaysis on a DataFrame
    """
    pass


class DataFrameICA(DataFrameReducerBase, decomposition.FastICA):
    """Perform Independent Comopnent Analysis on a DataFrame
    """
    pass


class DataFrameTSNE(DataFrameReducerBase):
    """Perform t-Distributed Stochastic Neighbor Embedding on a DataFrame

    Read more: http://homepage.tudelft.nl/19j49/t-SNE.html
    """

    def fit_transform(self, X):
        """Perform both a fit and a transform on the input data

        Fit the data to the reduction algorithm, and transform the data to
        the reduced space.

        Parameters
        ----------
        X : pandas.DataFrame
            A (n_samples, n_features) dataframe to both fit and transform

        Returns
        -------
        self : DataFrameReducerBase
            A fit and transformed instance of the object

        Raises
        ------
        ValueError
            If the input is not a pandas DataFrame, will not perform the fit
            and transform

        """
        from tsne import bh_sne

        self._check_dataframe(X)
        return pd.DataFrame(bh_sne(X), index=X.index)
