from collections import defaultdict
from itertools import cycle
import math

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..compute.decomposition import NMF, PCA
from .color import set1


def L1_distance(x, y):
    """Really should just be using TODO:scipy.linalg.norm with order=1"""
    return abs(y) + abs(x)


def L2_distance(x, y):
    """Really should just be using TODO:scipy.linalg.norm with order=2"""
    return math.sqrt((y ** 2) + (x ** 2))


class DecompositionViz(object):
    """
    Given a pandas dataframe, performs PCA and plots the results in a
    convenient single function.


    @param c_scale: Component scaling of the plot, e.g. for making the
    plotted vectors larger or smaller.
    @param x_pc: Integer, which principal component to use for the x-axis
    (usually 1)
    @param y_pc: Integer, which principal component to use for the y-axis
    (usually 2)
    @param distance:
    @param colors_dict: A dictionary of index (samples) to matplotlib colors
    @param markers_dict: A dictionary of index (samples) to matplotlib markers
    @param markers_size_dict: A dictionary of index (samples) to matplotlib
        marker sizes
    @param title: A string, the title of the plot
    @param show_vectors: Boolean, whether or not to show vectors
    @param show_point_labels: Boolean, whether or not to show the index,
    e.g. the sample name, on the plot
    @param column_ids_dict: A dictionary of column names to another
    value, e.g. if the columns are splicing events with a strange ID,
    this could be a dictionary that matches the ID to a gene name.
    @param index_ids_dict: A dictionary of index names to another
    value, e.g. if the indexes are samples with a strange ID, this could be a
     dictionary that matches the ID to a more readable sample name.
    @param show_vector_labels: Boolean. Can be helpful if the vector labels
    are gene names.
    @param scale_by_variance: Boolean. Scale vector components by explained
        variance
    @return: x, y, marker, distance of each vector in the study_data.
    """


    def __init__(self, df, title='', n_components=None, whiten=False,
                 reduction_args=None, feature_renamer=None, groupby=None,
                 color=None, order=None, violinplot_kws=None,
                 data_type=None, label_to_color=None, label_to_marker=None,
                 DataModel=None,
                 **kwargs):

        self.DataModel = DataModel

        self.title = title
        self._default_reduction_kwargs = {}

        self.groupby = groupby
        self.color = color
        self.order = order
        self.violinplot_kws = violinplot_kws
        self.data_type = data_type
        self.label_to_color = label_to_color
        self.label_to_marker = label_to_marker

        if reduction_args is None:
            reduction_args = self._default_reduction_kwargs
        else:
            reduction_args = self._default_reduction_kwargs.update(
                reduction_args)

        self.feature_renamer = feature_renamer
        if self.feature_renamer is None:
            self.feature_renamer = lambda x: x

        # This magically initializes the reducer like PCA or NMF
        super(DecompositionViz, self).__init__(n_components=n_components,
                                               whiten=whiten, **reduction_args)

        assert isinstance(df, pd.DataFrame)
        self.df = df
        if self.groupby is None:
            self.groupby = dict.fromkeys(self.df.index, 'all')

        self.reduced_space = self.fit_transform(self.df)

    def __call__(self, ax=None,
                 x_pc='pc_1', y_pc='pc_2', num_vectors=20,
                 **kwargs):
        gs_x = 14
        gs_y = 12

        if ax is None:
            self.reduced_fig, ax = plt.subplots(1, 1, figsize=(25, 12))
            gs = GridSpec(gs_x, gs_y)

        else:
            gs = GridSpecFromSubplotSpec(gs_x, gs_y, ax.get_subplotspec())
            self.reduced_fig = plt.gcf()

        ax_components = plt.subplot(gs[:, :5])
        ax_loading1 = plt.subplot(gs[:, 6:8])
        ax_loading2 = plt.subplot(gs[:, 10:14])

        kwargs.update({'ax': ax_components})
        self.num_vectors = num_vectors

        self.plot_samples(num_vectors=self.num_vectors, **kwargs)
        self.plot_loadings(pc=x_pc, ax=ax_loading1)
        self.plot_loadings(pc=y_pc, ax=ax_loading2)
        sns.despine()
        self.reduced_fig.tight_layout()

        self.plot_violins()
        return self

    def plot_samples(self, x_pc='pc_1', y_pc='pc_2',
                     show_point_labels=True,
                     distance='L1', num_vectors=20,
                     title='PCA', show_vectors=True,
                     show_vector_labels=True, markersize=10,
                     three_d=False, legend=True, ax=None,
                     scale_by_variance=True):

        """
        Given a pandas dataframe, performs PCA and plots the results in a
        convenient single function.

        Parameters
        ----------
        groupby : groupby
            How to group the samples by color/label
        x_pc : str
            which Principal Component to plot on the x-axis
        y_pc : str
            Which Principal Component to plot on the y-axis
        distance : str
            either 'L1' or 'L2' distance to plot the vectors
        num_vectors : int
            Number of vectors to plot of the principal components
        label_to_color : dict
            Group labels to a matplotlib color E.g. if you've already chosen
            specific colors to indicate a particular group. Otherwise will
            auto-assign colors
        label_to_marker : dict
            Group labels to matplotlib marker
        title : str
            title of the plot
        show_vectors : bool
            Whether or not to draw the vectors indicating the supporting
            principal components
        show_vector_labels : bool
            whether or not to draw the names of the vectors
        show_point_labels : bool
            Whether or not to label the scatter points
        markersize : int
            size of the scatter markers on the plot
        text_group : list of str
            Group names that you want labeled with text
        three_d : bool
            if you want hte plot in 3d (need to set up the axes beforehand)

        Returns
        -------
        For each vector in data:
        x, y, marker, distance
        """
        # if three_d:
        #     from mpl_toolkits.mplot3d import Axes3D
        #
        #     fig = plt.figure(figsize=(10, 10))
        #     ax = fig.add_subplot(111, projection='3d')
        # else:
        #     fig, ax = plt.subplots(figsize=(10, 10))

        if ax is None:
            ax = plt.gca()

        if self.label_to_color is None:
            colors = cycle(set1)

            def color_factory():
                return colors.next()

            label_to_color = defaultdict(color_factory)

        if self.label_to_marker is None:
            markers = cycle(['o', '^', 's', 'v', '*', 'D', 'h'])

            def marker_factory():
                return markers.next()

            label_to_marker = defaultdict(marker_factory)



        # Plot the samples
        grouped = self.reduced_space.groupby(self.groupby, axis=0)
        for name, df in grouped:
            color = self.label_to_color[name]
            marker = self.label_to_marker[name]
            x = df[x_pc]
            y = df[y_pc]
            ax.plot(x, y, color=color, marker=marker, linestyle='None',
                    label=name, markersize=markersize, alpha=0.75)
            if show_point_labels:
                for args in zip(x, y, df.index):
                    ax.text(*args)

        # Get the explained variance
        try:
            vars = self.explained_variance_ratio_[[x_pc, y_pc]]
        except AttributeError:
            vars = pd.Series([1., 1.], index=[x_pc, y_pc])

        # Plot vectors, if asked
        if show_vectors:
            loadings = self.components_.ix[[x_pc, y_pc]]

            if scale_by_variance:
                loadings = loadings.multiply(vars, axis=0)

            # sort features by magnitude/contribution to transformation
            reduced_space = self.reduced_space[[x_pc, y_pc]]
            farthest_sample = reduced_space.apply(np.linalg.norm, axis=0).max()
            whole_space = loadings.apply(np.linalg.norm).max()
            scale = .25 * farthest_sample / whole_space
            loadings *= scale

            ord = 2 if distance == 'L2' else 1
            self.magnitudes = loadings.apply(np.linalg.norm, ord=ord)
            self.magnitudes.sort(ascending=False)

            for vector_label in self.magnitudes[:num_vectors].index:
                x, y = loadings[vector_label]
                ax.plot([0, x], [0, y], color='k', linewidth=1)
                if show_vector_labels:
                    x_offset = math.copysign(5, x)
                    y_offset = math.copysign(5, y)
                    horizontalalignment = 'left' if x > 0 else 'right'
                    renamed = self.feature_renamer(vector_label)
                    ax.annotate(renamed, (x, y),
                                textcoords='offset points',
                                xytext=(x_offset, y_offset),
                                horizontalalignment=horizontalalignment)

        # Label x and y axes
        ax.set_xlabel(
            'Principal Component {} (Explains {:.2f}% Of Variance)'.format(
                str(x_pc), vars[x_pc]))
        ax.set_ylabel(
            'Principal Component {} (Explains {:.2f}% Of Variance)'.format(
                str(y_pc), vars[y_pc]))
        ax.set_title(title)

        if legend:
            ax.legend()
        sns.despine()

    def plot_loadings(self, pc='pc_1', n_features=50, ax=None):
        x = self.components_.ix[pc].copy()
        x.sort(ascending=True)
        half_features = int(n_features / 2)
        if len(x) > n_features:
            a = x[:half_features]
            b = x[-half_features:]
            dd = np.r_[a, b]
            labels = np.r_[a.index, b.index]
        else:
            dd = x
            labels = x.index

        if ax is None:
            ax = plt.gca()

        ax.plot(dd, np.arange(len(dd)), 'o')

        ax.set_yticks(np.arange(max(len(dd), n_features)))
        ax.set_title("Component " + pc)

        x_offset = max(dd) * .05
        ax.set_xlim(left=min(dd) - x_offset, right=max(dd) + x_offset)

        self.top_features = labels

        labels = map(self.feature_renamer, labels)
        # shorten = lambda x: '{}...'.format(x[:30]) if len(x) > 30 else x
        # ax.set_yticklabels(map(shorten, labels))
        ax.set_yticklabels(labels)
        for lab in ax.get_xticklabels():
            lab.set_rotation(90)
        sns.despine(ax=ax)

    def plot_explained_variance(self, title="PCA"):
        """If the reducer is a form of PCA, then plot the explained variance
        ratio by the components.
        """
        # Plot the explained variance ratio
        assert hasattr(self, 'explained_variance_ratio_')
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots()
        ax.plot(self.explained_variance_ratio_, 'o-')

        ax.set_xticks(range(self.n_components))
        ax.set_xticklabels(map(str, np.arange(self.n_components) + 1))
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Fraction explained variance')
        ax.set_title(title)
        sns.despine()
        return fig

    def plot_violins(self):
        """Make violinplots of each feature
        """
        ncols = 4
        nrows = 1

        vector_labels = set(self.magnitudes[:self.num_vectors].index.union(
            self.top_features))

        while ncols * nrows < len(vector_labels):
            nrows += 1

        self.violins_fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                              figsize=(4 * ncols, 4 * nrows))

        for vector_label, ax in zip(vector_labels, axes.flat):
            # renamed = self.feature_renamer(vector_label)

            self.DataModel._violinplot(feature_id=vector_label,
                                       sample_ids=self.df.index,
                                       phenotype_groupby=self.groupby,
                                       phenotype_order=self.order,
                                       ax=ax, color=self.color,
                                       label_pooled=True)

        # Clear any unused axes
        for ax in axes.flat:
            # Check if the plotting space is empty
            if len(ax.collections) == 0 or len(ax.lines) == 0:
                ax.axis('off')
        self.violins_fig.tight_layout()


class PCAViz(DecompositionViz, PCA):
    pass


class NMFViz(DecompositionViz, NMF):
    _default_reduction_kwargs = \
        {'n_components': 2, 'max_iter': 20000, 'nls_max_iter': 40000}

    def __call__(self, ax=None, **kwargs):
        gs_x = 14
        gs_y = 12

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(25, 12))
            gs = GridSpec(gs_x, gs_y)

        else:
            gs = GridSpecFromSubplotSpec(gs_x, gs_y, ax.get_subplotspec())
            fig = plt.gcf()

        ax_components = plt.subplot(gs[:, :5])
        ax_loading1 = plt.subplot(gs[:, 6:8])
        ax_loading2 = plt.subplot(gs[:, 10:14])

        passed_kwargs = kwargs
        local_kwargs = self.plotting_kwargs.copy()
        local_kwargs.update(passed_kwargs)
        local_kwargs.update({'ax': ax_components})
        self.plot_samples(**local_kwargs)
        self.plot_loadings(pc=local_kwargs['x_pc'], ax=ax_loading1)
        self.plot_loadings(pc=local_kwargs['y_pc'], ax=ax_loading2)
        sns.despine()
        fig.tight_layout()
        return self


def plot_pca(df, **kwargs):
    """ for backwards-compatibility """
    pca = PCAViz(df, **kwargs)
    pca.plot_samples()
