__author__ = 'olga'

import seaborn as sns

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt



import brewer2mpl
from itertools import cycle
# from scipy.spatial.distance import pdist, squareform
# import skfuzzy
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans, spectral_clustering
from sklearn.decomposition import PCA


from singlesail.splicing.utils import get_switchy_score_order
from singlesail.splicing.viz import lavalamp

sns.set(style='white', context='talk')

# class FuzzyCMeans(BaseEstimator, ClusterMixin, TransformerMixin):
#     """Class for Fuzzy C-means clustering in scikit-learn cluster class format
#
#     Implements the objective function as described:
#     http://en.wikipedia.org/wiki/Fuzzy_clustering
#
#     Parameters
#     ----------
#     n_clusters : int
#         number of clusters to estimate
#     exponent : float
#         Exponent of objective function
#     min_error : float
#         The threshold at which if the objective function does not change by
#         more than this, the iterations are stopped.
#     max_iter : int
#          Maximum number of iterations
#
#     Attributes
#     ----------
#     `cluster_centers_` : array, [n_clusters, n_features]
#         Coordinates of cluster centers
#
#     `labels_` :
#         Labels of each point
#
#     `inertia_` : float
#         The value of the inertia criterion associated with the chosen
#         partition.
#
#     Methods
#     -------
#     fit
#         Like sklearn.cluster.KMeans, fit a matrix with samples to classify on
#         the rows and features on the columns to the number of clusters.
#         Basically, 'perform clustering'
#     """
#     def __in , exponent=2, min_error=0.01, max_iter=100):
#         """Initialize a Fuzzy C-Means clusterer
#
#         Parameters
#         ----------
#         n_clusters : int
#             number of clusters to estimate
#         exponent : float
#             Exponent of objective function
#         min_error : float
#             The threshold at which if the objective function does not change by
#             more than this, the iterations are stopped.
#         max_iter : int
#              Maximum number of iterations
#
#         Returns
#         -------
#         self : FuzzyCMeans
#             An instantiated class of FuzzyCMeans, ready for clustering!
#         """
#         self.n_clusters = n_clusters
#         self.exponent = exponent
#         self.min_error = min_error
#         self.max_iter = max_iter
#
#     def fit(self, data, prob_thresh=None):
#         """Fit the data to the number of clusters
#
#         Parameters
#         ----------
#         data : numpy.array
#             A numpy array of values (no NAs!) in the same format as required
#             by scikit-learn, that in the shape [n_samples, n_features]
#         """
#         cluster_centers, fuzzy_matrix, initial_guess, distance_matrix, \
#             objective_function_history, n_iter, fuzzy_partition_coeff = \
#             skfuzzy.cmeans(data.T, c=self.n_clusters, m=self.exponent,
#                           error=self.min_error, maxiter=self.max_iter)
#
#         # rewrite as sklearn terminology
#         self.cluster_centers_ = cluster_centers
#         self.probability_of_labels = fuzzy_matrix
#         self.labels_ = np.apply_along_axis(np.argmax, axis=0, arr=self.probability_of_labels)
#
#         # Adjust labels for everything that didn't have a cluster membership
#         # prob >= prob_thresh
#         if prob_thresh is not None:
#             self.unclassifiable = (self.probability_of_labels >= prob_thresh)\
#                                       .sum(axis=0) == 0
#             self.labels_[self.unclassifiable] = -1
#
#         self.distance_matrix = distance_matrix
#         self.objective_function_history = objective_function_history
#         self.n_iter = n_iter
#         self.fuzzy_partition_coeff = fuzzy_partition_coeff



class Data(object):
    def __init__(self, psi, n_components, binsize=0.1, figure_dir='.',
                 reducer=PCA, reducer_kws=None):
        """Instantiate a object for df scores with binned and reduced data

        Parameters
        ----------
        df : pandas.DataFrame
            A [n_events, n_samples] dataframe of splicing events
        n_components : int
            Number of components to use in the reducer
        binsize : float
            Value between 0 and 1, the bin size for binning the df scores
        reducer : sklearn.decomposition object
            An scikit-learn class that reduces the dimensionality of data
            somehow. Must accept the parameter n_components, have the
            functions fit, transform, and have the attribute components_

        """
        self.psi = psi
        # self.reducer = reducer
        self.reducer_kws = reducer_kws

        self.figure_dir = figure_dir
        self.reducer_name = str(reducer).split('.')[-1].rstrip("'>")

        # self.psi_fillna_mean = self.df.T.fillna(self.df.mean(axis=1)).T
        self.binsize = binsize
        self.n_components = n_components
        self.binify().reduce(reducer)

    def binify(self):
        """Bins df scores from 0 to 1 on the provided binsize size"""
        self.bins = np.arange(0, 1+self.binsize, self.binsize)
        ncol = int(1/self.binsize)
        nrow = self.psi.shape[0]
        self.binned = np.zeros((nrow, ncol))
        for i, (name, row) in enumerate(self.psi.iterrows()):
            self.binned[i,:] = np.histogram(row, bins=self.bins, normed=True)[0]
        return self

    def reduce(self, reducer):
        """Reduces dimensionality of the binned df score data
        """
        reducer_kws = {} if self.reducer_kws is None else self.reducer_kws
        reducer_kws.setdefault('n_components', self.n_components)
        self.reducer = reducer(**reducer_kws).fit(self.binned)
        self.reduced_binned = self.reducer.transform(self.binned)
        if hasattr(self.reducer, 'explained_variance_ratio_'):
            self.plot_explained_variance(self.reducer,
                                         '{} on binned data'.format(self.reducer))
        return self


    def plot_explained_variance(self, pca, title):
        # Plot the explained variance ratio
        fig, ax = plt.subplots()
        ax.plot(pca.explained_variance_ratio_, 'o-')
        ax.set_xticks(range(pca.n_components))
        ax.set_xticklabels(map(str, np.arange(pca.n_components)+1))
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Fraction explained variance')
        ax.set_title(title)
        sns.despine()
        fig.savefig('{}/{}_binsize={}_ncomponents={}_explained_variance.pdf'
                    .format(self.figure_dir, self.reducer_name, self.binsize,
                            self.n_components))

    # def calculate_distances(self, metric='euclidean'):
    #     """Creates a squareform distance matrix for clustering fun
    #
    #     Needed for some clustering algorithms
    #
    #     Parameters
    #     ----------
    #     metric : str
    #         One of any valid scipy.distance metric strings
    #     """
    #     self.pdist = squareform(pdist(self.binned, metric=metric))
    #     return self



class ClusteringTester(object):
    """Class for consistent evaluation of clustering methods

    Attributes
    ----------


    Methods
    -------
    hist_lavalamp
        Plot a histogram and lavalamp of df scores from each cluster
    pca_viz
        Vizualize the clusters on the PCA of the data
    """
    def __init__(self, data, ClusterMethod, reduced='binned', cluster_kws=None,
                 colors=None):
        """Initialize ClusterTester and cluster the data

        Parameters
        ----------
        data : Data
            An object of the Data class
        ClusterMethod : sklearn.cluster class
            An object of the format from sklearn.cluster. Must have the fit()
            method, and create the attributes labels_
        reduced : str
            Specified which PCA-reduced data to use. Either the
            histogram-binned data ("binned") or the raw df scores ("df")
        """
        self.data = data
        self.reduced = self._get_reduced(reduced)

        cluster_kws = cluster_kws if cluster_kws is not None else {}
        self.clusterer = ClusterMethod(**cluster_kws)
        if ClusterMethod != spectral_clustering:
            self.clusterer.fit(self.reduced)
            self.labels = self.clusterer.labels_
        else:
            self.labels = self.clusterer
        self.labels_unique = set(self.labels)
        self.n_clusters = len(self.labels_unique)
        print 'n_clusters:', self.n_clusters

        if colors is None:
            self.colors = brewer2mpl.get_map('Set1', 'Qualitative', 8).mpl_colors
        else:
            self.colors = colors
        self.color_cycle = cycle(self.colors)

    def _annotate_centers(self, ax):
        """If the clusterer has cluster_centers_, plot the centroids

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            Axes object to plot the annotation on
        """
        try:
            # Plot the centroids as a white X
            centroids = self.clusterer.cluster_centers_
        except AttributeError:
            return

        ax.scatter(centroids[:, 0], centroids[:, 1],
                   marker='x', s=169, linewidths=3,
                   color='k', zorder=10)
        for i in range(self.n_clusters):
            try:
                ax.annotate(str(i),
                            (centroids[i, 0], centroids[i, 1]),
                            fontsize=24, xytext=(-20, 0),
                            textcoords='offset points')
            except:
                pass

    def _get_reduced(self, reduced):
        """Sanely extracts the PCA-reduced data from the Data object

        Parameters
        ----------
        reduced : str
            Either "binned" or "df"

        Returns
        -------
        reduced_data : numpy.array
            The PCA-reduced data from the specified array
        """
        # if reduced.lower() == 'df':
        #     reduced_data = self.data.reduced_psi
        if reduced.lower() == 'binned':
        # elif reduced.lower() == 'binned':
            reduced_data = self.data.reduced_binned
        else:
            raise ValueError('only "df" can be specified as a reduced '
                             'dataset. the option for this is historic and I '
                             'am keeping it around just in case.')
            # raise ValueError('Reduced data must be specified as one of "df" '
            #                  'or "binned", not {}'.format(reduced))
        return reduced_data

    def _hist(self, ax, label, color):
        """Plot histograms of the df scores of one label"""
        ax.hist(self.data.psi.ix[self.data.psi.index[self.labels == label],:].values.flat,
                bins=np.arange(0, 1.05, 0.05), facecolor=color, linewidth=0.1)
        ax.set_title('Cluster: {}'.format(label))
        ax.set_xlim(0,1)
        sns.despine()

    def _lavalamp(self, ax, label, color):
        """makes a lavalamp of df scores of one label"""
        ind = self.labels == label
        n_events = (ind).sum()
        y = self.data.psi.ix[self.data.psi.index.values[ind], :]
        lavalamp(y, color=color, ax=ax, title='n = {}'.format(n_events))

    def hist_lavalamp(self):
        """Plot a histogram and _lavalamp for all the clusters

        Returns
        -------
        fig : matplotlib.pyplot.figure
            A figure instance with all the histograms and _lavalamp plots of
            all labels, for saving.
        """
        # Reset the color cycle in case we already cycled through it
        self.color_cycle = cycle(self.colors)

        fig = plt.figure(figsize=(16, 4*self.n_clusters))
        for i, (label, color) in enumerate(zip(self.labels_unique,
                                          self.color_cycle)):
            if label % 10 == 0:
                print 'plotting cluster {} of {}'.format(label, self.n_clusters)
            if label == -1:
                color = 'k'
            n_samples_in_cluster = (self.labels == label).sum()
            if n_samples_in_cluster <= 5:
                continue

            # fig = plt.figure(figsize=(16, 4))
            hist_ax = plt.subplot2grid((self.n_clusters, 5), (i, 0), colspan=1,
                                       rowspan=1)
            lavalamp_ax = plt.subplot2grid((self.n_clusters, 5), (i, 1),
                                           colspan=4, rowspan=1)

            self._hist(hist_ax, label, color=color)
            self._lavalamp(lavalamp_ax, label, color=color)
        fig.tight_layout()
        return fig

    def _plot_pca_vectors(self, ax):
        """Plot the component vectors of the principal components
        """
        # # sort features by magnitude/contribution to transformation
        # x_loading, y_loading = self.components_[:, 0], \
        #                        self.components_.ix[:, 1]
        # c_scale = .75 * max([norm(point)
        #                      for point in self.reduced[:,(0,1)]]) / \
        #           max([norm(vector) for vector in zip(x_loading, y_loading)])
        #
        # comp_magn = []
        # magnitudes = []
        # for (x, y, an_id) in zip(x_loading, y_loading, self.X.columns):
        #
        #     x = x * c_scale
        #     y = y * c_scale
        #
        #     if distance_metric == 'L1':
        #         mg = L1_distance(x, y)
        #
        #     elif distance_metric == 'L2':
        #         mg = L2_distance(x, y)
        #
        #     comp_magn.append((x, y, an_id, mg))
        #     magnitudes.append(mg)
        #
        # vectors = sorted(comp_magn, key=lambda item: item[3], reverse=True)[
        #           :num_vectors]
        #
        # for x, y, marker, distance in vectors:
        #
        #     try:
        #         color = vector_colors_dict[marker]
        #     except:
        #         color = 'black'
        #
        #     if show_vectors:
        #         ax.plot(ax, [0, x], [0, y], color=color,
        #                  linewidth=vector_width)
        #
        #         if show_vector_labels:
        #             ax.annotate(1.1 * x, 1.1 * y, marker,
        #                         color=color,
        #                     size=vector_label_size)
        pass

    def pca_viz(self, celltype=''):
        """Visualizes the clusters on the PCA of the data

        Returns
        -------
        fig : matplotlib.pyplot.figure
            A figure instance with the PCA, for saving.

        """

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = self.reduced[:, 0].min(), self.reduced[:, 0].max()
        y_min, y_max = self.reduced[:, 1].min(), self.reduced[:, 1].max()

        fig, ax = plt.subplots(figsize=(12, 8))

        # Reset the color cycle in case we already cycled through it
        self.color_cycle = cycle(self.colors)

        # Get the list of colors to choose from so we can index it with the
        # cluster label. Otherwise we'd get an error because you can't index
        # an itertools.cycle object, which is what self.color_cycle is
        colors = [color for _, color in
                  zip(range(self.n_clusters), self.color_cycle)]

        # Any label that is -1 (relevant for FuzzyCMeans with a probability
        # cutoff, and DBSCAN clustering which only makes "confident"
        # clusters, and everything else is -1), color these labels as black
        # ("k" in matplotlib colors)
        color_list = [colors[int(label)] if label>=0 else 'k'
                      for label in self.labels]
        ax.scatter(self.reduced[:, 0], self.reduced[:, 1],
                   color=color_list, alpha=0.25, linewidth=0.1, edgecolor='#262626')
        self._annotate_centers(ax)

        self._plot_pca_vectors(ax)

        ax.set_title('{} clustering on the {} dataset ({}-reduced data)\n'
                 'Centroids are marked with black cross (binsize={:.2f})'
                     .format(celltype, type(self.clusterer),
                             repr(self.data.reducer), self.data.binsize))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        sns.despine(left=True, bottom=True)
        return fig

    def violinplot_random_cluster_members(self, n=20):
        """Make violin plot of n random cluster members.

        Useful for seeing whether a cluster has bimodal events or not,
        which is not obvious from the lava lamp plot

        Parameters
        ----------
        n : int
            Number of cluster members to plot. Default 20.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            A figure instance with all the violin plots of all labels,
            for saving.

        """
        self.color_cycle = cycle(self.colors)
        fig, axes_array = plt.subplots(nrows=self.n_clusters, ncols=n,
                                       figsize=(n, 2*self.n_clusters),
                                       sharey=True)
        for axes, label, color in zip(axes_array, self.labels_unique,
                                self.color_cycle):
            if label == -1:
                color = 'k'
            these_labels = self.labels == label
            events = np.random.choice(self.data.psi.index.values[
                                          these_labels], size=n)
            y = self.data.psi.ix[events,:].values.T
            order = get_switchy_score_order(y)
            events = events[order]
            for event, ax in zip(events, axes):
        #         if i % 20 == 0:
                sns.violinplot(self.data.psi.ix[event], bw=0.1, inner='points',
                               color=color, linewidth=0, ax=ax, alpha=0.75)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_xlabel(label)
                sns.despine()
        fig.tight_layout()
        return fig
