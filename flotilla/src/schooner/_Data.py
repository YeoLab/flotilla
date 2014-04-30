import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import seaborn as sns

class Data(object):
    """Generic data model for both splicing and expression data

    Attributes
    ----------


    Methods
    -------

    """
    def __init__(self, data, n_components, step=0.1, reducer=PCA):
        """Constructor for Data

        Specific implementation in the SplicingData and ExpressionData classes
        """
        raise NotImplementedError

    def calculate_distances(self, metric='euclidean'):
        """Creates a squareform distance matrix for clustering fun

        Needed for some clustering algorithms

        Parameters
        ----------
        metric : str
            One of any valid scipy.distance metric strings
        """
        self.pdist = squareform(pdist(self.binned, metric=metric))
        return self

    def correlate(self, method='spearman', between='measurements'):
        """Find correlations between either splicing/expression measurements
        or cells
        """
        raise NotImplementedError

    def jsd(self):
        """Jensen-Shannon divergence showing most varying measurements within a
        celltype and between celltypes

        Returns
        -------
        fig : matplotlib.Figure
            A figure object for saving.
        """
        raise NotImplementedError

    def plot_explained_variance(self, pca, title):
        """If the reducer is a form of PCA, then plot the explained variance
        ratio by the components.
        """
        # Plot the explained variance ratio
        fig, ax = plt.subplots()
        ax.plot(pca.explained_variance_ratio_, 'o-')
        ax.set_xticks(range(pca.n_components))
        ax.set_xticklabels(map(str, np.arange(pca.n_components)+1))
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Fraction explained variance')
        ax.set_title(title)
        sns.despine()
        return fig

    def pca(self):
        """Principal component analysis of all measurements, labeled by
        celltype

        Returns
        -------
        fig : matplotlib.Figure
            A figure object for saving.
        """
        raise NotImplementedError


    def reduce(self, data, reducer=PCA, n_components=2):
        """Reduces dimensionality of data, by default using PCA

        Q: each scatter point in PCA an event or a cell?
        """
        self.reducer = reducer(n_components=n_components).fit(data)
        reduced_data = self.reducer.transform(data)
        if hasattr(self.reducer, 'explained_variance_ratio_'):
            self.plot_explained_variance(self.reducer,
                                         '{} on binned data'.format(
                                             self.reducer))
        return reduced_data