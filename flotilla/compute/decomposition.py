import sklearn
import pandas as pd

from sklearn import decomposition


class PrettyReducer(object):
    """

    Just like scikit-learn's reducers, but with prettied up DataFrames.

    """

    def relabel_pcs(self, x):
        return "pc_" + str(int(x) + 1)

    def fit(self, X):

        try:
            assert type(X) == pd.DataFrame
        except:
            print "Try again as a pandas study_data frame"
            raise

        self.X = X
        super(PrettyReducer, self).fit(X)
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
        component_space = super(PrettyReducer, self).transform(X)
        if type(self.X) == pd.DataFrame:
            component_space = pd.DataFrame(component_space,
                                           index=X.index).rename_axis(
                self.relabel_pcs, 1)
        return component_space

    def fit_transform(self, X):
        try:
            assert type(X) == pd.DataFrame
        except:
            print "Try again as a pandas study_data frame"
            raise
        self.fit(X)
        return self.transform(X)


class PCA(PrettyReducer, decomposition.PCA):
    pass


class NMF(PrettyReducer, decomposition.NMF):
    here = True

    def fit(self, X):
        """
        duplicated fit code for NMF because sklearn's NMF cheats for efficiency
        and calls fit_transform. MRO resolves the closest (in this package)
        fit_transform first and so there's a recursion error:

            def fit(self, X, y=None, **params):
                self.fit_transform(X, **params)
                return self
        """

        try:
            assert type(X) == pd.DataFrame
        except:
            print "Try again as a pandas study_data frame"
            raise

        self.X = X
        # notice this is fit_transform, not fit
        super(sklearn.decomposition.NMF, self).fit_transform(X)
        self.components_ = pd.DataFrame(self.components_,
                                        columns=self.X.columns).rename_axis(
            self.relabel_pcs, 0)
