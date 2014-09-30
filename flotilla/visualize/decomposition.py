from collections import defaultdict
from itertools import cycle
import math

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns




# from ..compute.decomposition import DataFrameNMF, DataFramePCA
from .color import set1


class DecompositionViz(object):
    """
    Plots the reduced space from a decomposed dataset. Does not perform any
    reductions of its own
    """

    def __init__(self, reduced_space, components_,
                 explained_variance_ratio_, DataModel=None,
                 feature_renamer=None, groupby=None,
                 featurewise=False,
                 color=None, order=None, violinplot_kws=None,
                 data_type=None, label_to_color=None, label_to_marker=None,
                 # violinplot=None,
                 scale_by_variance=True, x_pc='pc_1',
                 y_pc='pc_2', n_vectors=20, distance='L1',
                 n_top_pc_features=50):
        """

        x_pc : str
            which Principal Component to plot on the x-axis
        y_pc : str
            Which Principal Component to plot on the y-axis
        distance : str
            either 'L1' or 'L2' distance to plot the vectors
        n_vectors : int
            Number of vectors to plot of the principal components

        """

        self.DataModel = DataModel
        self._default_reduction_kwargs = {}

        self.groupby = groupby
        self.color = color
        self.order = order
        self.violinplot_kws = violinplot_kws
        self.data_type = data_type
        self.label_to_color = label_to_color
        self.label_to_marker = label_to_marker
        self.n_vectors = n_vectors
        self.x_pc = x_pc
        self.y_pc = y_pc
        self.pcs = [self.x_pc, self.y_pc]
        self.distance = distance
        self.n_top_pc_features = n_top_pc_features
        self.featurewise = featurewise

        self.reduced_space = reduced_space
        self.components_ = components_

        if self.label_to_color is None:
            colors = cycle(set1)

            def color_factory():
                return colors.next()

            self.label_to_color = defaultdict(color_factory)

        if self.label_to_marker is None:
            markers = cycle(['o', '^', 's', 'v', '*', 'D', 'h'])

            def marker_factory():
                return markers.next()

            self.label_to_marker = defaultdict(marker_factory)

        # if decomposer_kwargs is None:
        #     decomposer_kwargs = self._default_reduction_kwargs
        # else:
        #     decomposer_kwargs = self._default_reduction_kwargs.update(
        #         decomposer_kwargs)

        # This magically initializes the reducer like DataFramePCA or DataFrameNMF
        # self.decomposer = deco
        # mposer(n_components=n_components,
        #                              **decomposer_kwargs)
        # super(DecompositionViz, self).__init__(n_components=n_components,
        #                                        **decomposer_kwargs)

        if self.groupby is None:
            self.groupby = dict.fromkeys(self.reduced_space.index, 'all')

        # self.reduced_space = self.fit_transform(self.df)
        self.loadings = self.components_.ix[[self.x_pc, self.y_pc]]

        # Get the explained variance
        if explained_variance_ratio_ is not None:
            self.vars = explained_variance_ratio_[[self.x_pc, self.y_pc]]
        else:
            self.vars = pd.Series([1., 1.], index=[self.x_pc, self.y_pc])

        if scale_by_variance:
            self.loadings = self.loadings.multiply(self.vars, axis=0)

        # sort features by magnitude/contribution to transformation
        reduced_space = self.reduced_space[[self.x_pc, self.y_pc]]
        farthest_sample = reduced_space.apply(np.linalg.norm, axis=0).max()
        whole_space = self.loadings.apply(np.linalg.norm).max()
        scale = .25 * farthest_sample / whole_space
        self.loadings *= scale

        ord = 2 if self.distance == 'L2' else 1
        self.magnitudes = self.loadings.apply(np.linalg.norm, ord=ord)
        self.magnitudes.sort(ascending=False)

        self.top_features = set([])
        self.pc_loadings_labels = {}
        self.pc_loadings = {}
        for pc in self.pcs:
            x = self.components_.ix[pc].copy()
            x.sort(ascending=True)
            half_features = int(self.n_top_pc_features / 2)
            if len(x) > self.n_top_pc_features:
                a = x[:half_features]
                b = x[-half_features:]
                labels = np.r_[a.index, b.index]
                self.pc_loadings[pc] = np.r_[a, b]
            else:
                labels = x.index
                self.pc_loadings[pc] = x


            self.pc_loadings_labels[pc] = labels
            self.top_features.update(labels)

    def __call__(self, ax=None, title='', plot_violins=True,
                 show_point_labels=False,
                 show_vectors=True,
                 show_vector_labels=True,
                 markersize=10, legend=True):
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

        # kwargs.update({'ax': ax_components})

        self.plot_samples(show_point_labels=show_point_labels,
                          title=title, show_vectors=show_vectors,
                          show_vector_labels=show_vector_labels,
                          markersize=markersize, legend=legend,
                          ax=ax_components)
        self.plot_loadings(pc=self.x_pc, ax=ax_loading1)
        self.plot_loadings(pc=self.y_pc, ax=ax_loading2)
        sns.despine()
        self.reduced_fig.tight_layout()

        if plot_violins and self.DataModel is not None and not self \
                .featurewise:
            self.plot_violins()
        return self

    def plot_samples(self, show_point_labels=True,
                     title='DataFramePCA', show_vectors=True,
                     show_vector_labels=True, markersize=10,
                     three_d=False, legend=True, ax=None):

        """
        Given a pandas dataframe, performs DataFramePCA and plots the results in a
        convenient single function.

        Parameters
        ----------
        groupby : groupby
            How to group the samples by color/label
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



        # Plot the samples
        grouped = self.reduced_space.groupby(self.groupby, axis=0)
        for name, df in grouped:
            color = self.label_to_color[name]
            marker = self.label_to_marker[name]
            x = df[self.x_pc]
            y = df[self.y_pc]
            ax.plot(x, y, color=color, marker=marker, linestyle='None',
                    label=name, markersize=markersize, alpha=0.75)
            if show_point_labels:
                for args in zip(x, y, df.index):
                    ax.text(*args)

        # Plot vectors, if asked
        if show_vectors:
            for vector_label in self.magnitudes[:self.n_vectors].index:
                x, y = self.loadings[vector_label]
                ax.plot([0, x], [0, y], color='k', linewidth=1)
                if show_vector_labels:
                    x_offset = math.copysign(5, x)
                    y_offset = math.copysign(5, y)
                    horizontalalignment = 'left' if x > 0 else 'right'
                    renamed = self.DataModel.feature_renamer(vector_label)
                    ax.annotate(renamed, (x, y),
                                textcoords='offset points',
                                xytext=(x_offset, y_offset),
                                horizontalalignment=horizontalalignment)

        # Label x and y axes
        ax.set_xlabel(
            'Principal Component {} (Explains {:.2f}% Of Variance)'.format(
                str(self.x_pc), 100 * self.vars[self.x_pc]))
        ax.set_ylabel(
            'Principal Component {} (Explains {:.2f}% Of Variance)'.format(
                str(self.y_pc), 100 * self.vars[self.y_pc]))
        ax.set_title(title)

        if legend:
            ax.legend()
        sns.despine()

    def plot_loadings(self, pc='pc_1', n_features=50, ax=None):
        # x = self.components_.ix[pc].copy()
        # x.sort(ascending=True)
        # half_features = int(n_features / 2)
        # if len(x) > n_features:
        #     a = x[:half_features]
        #     b = x[-half_features:]
        #     dd = np.r_[a, b]
        #     labels = np.r_[a.index, b.index]
        # else:
        #     dd = x
        #     labels = x.index
        # import pdb; pdb.set_trace()
        # half_features = n_features/2
        # top_loadings = self.pc_loadings[pc][:half_features]
        # bottom_loadings = self.pc_loadings[pc][-half_features:]
        loadings = self.pc_loadings[pc]
        labels = self.pc_loadings_labels[pc]

        if ax is None:
            ax = plt.gca()

        ax.plot(loadings, np.arange(loadings.shape[0]), 'o')

        ax.set_yticks(np.arange(max(loadings.shape[0], n_features)))
        ax.set_title("Component " + pc)

        x_offset = max(loadings) * .05
        ax.set_xlim(left=loadings.min() - x_offset,
                    right=loadings.max() + x_offset)

        # self.top_features.extend(labels)

        labels = map(self.DataModel.feature_renamer, labels)
        # shorten = lambda x: '{}...'.format(x[:30]) if len(x) > 30 else x
        # ax.set_yticklabels(map(shorten, labels))
        ax.set_yticklabels(labels)
        for lab in ax.get_xticklabels():
            lab.set_rotation(90)
        sns.despine(ax=ax)

    # def plot_explained_variance(self, title="DataFramePCA"):
    #     """If the reducer is a form of DataFramePCA, then plot the explained variance
    #     ratio by the components.
    #     """
    #     # Plot the explained variance ratio
    #     assert hasattr(self, 'explained_variance_ratio_')
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #
    #     fig, ax = plt.subplots()
    #     ax.plot(self.explained_variance_ratio_, 'o-')
    #
    #     ax.set_xticks(range(self.n_components))
    #     ax.set_xticklabels(map(str, np.arange(self.n_components) + 1))
    #     ax.set_xlabel('Principal component')
    #     ax.set_ylabel('Fraction explained variance')
    #     ax.set_title(title)
    #     sns.despine()
    #     return fig

    def plot_violins(self):
        """Make violinplots of each feature

        Must be called after plot_samples because it depends on the existence
        of the "self.magnitudes" attribute.
        """
        ncols = 4
        nrows = 1
        vector_labels = list(set(self.magnitudes[:self.n_vectors].index.union(
            pd.Index(self.top_features))))
        while ncols * nrows < len(vector_labels):
            nrows += 1
        self.violins_fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                              figsize=(4 * ncols, 4 * nrows))
        renamed_vectors = map(self.DataModel.feature_renamer, vector_labels)
        vector_labels = [x for (y, x) in sorted(zip(renamed_vectors,
                                                    vector_labels))]

        # import pdb; pdb.set_trace()

        for vector_label, ax in zip(vector_labels, axes.flat):
            # renamed = self.feature_renamer(vector_label)

            self.DataModel._violinplot(feature_id=vector_label,
                                       sample_ids=self.reduced_space.index,
                                       phenotype_groupby=self.groupby,
                                       phenotype_order=self.order,
                                       ax=ax, color=self.color)

        # Clear any unused axes
        for ax in axes.flat:
            # Check if the plotting space is empty
            if len(ax.collections) == 0 or len(ax.lines) == 0:
                ax.axis('off')
        self.violins_fig.tight_layout()


        # class PCAViz(DecompositionViz, DataFramePCA):
        #     _default_reduction_kwargs = dict(whiten=False)
        #     pass
        #
        #
        # class NMFViz(DecompositionViz, DataFrameNMF):
        #     _default_reduction_kwargs = \
        #         {'n_components': 2, 'max_iter': 20000, 'nls_max_iter': 40000}
        #
        #     def __call__(self, ax=None, **kwargs):
        #         pass
        # gs_x = 14
        # gs_y = 12
        #
        # if ax is None:
        #     fig, ax = plt.subplots(1, 1, figsize=(25, 12))
        #     gs = GridSpec(gs_x, gs_y)
        #
        # else:
        #     gs = GridSpecFromSubplotSpec(gs_x, gs_y, ax.get_subplotspec())
        #     fig = plt.gcf()
        #
        # ax_components = plt.subplot(gs[:, :5])
        # ax_loading1 = plt.subplot(gs[:, 6:8])
        # ax_loading2 = plt.subplot(gs[:, 10:14])
        #
        # passed_kwargs = kwargs
        # local_kwargs = self.plotting_kwargs.copy()
        # local_kwargs.update(passed_kwargs)
        # local_kwargs.update({'ax': ax_components})
        # self.plot_samples(**local_kwargs)
        # self.plot_loadings(pc=local_kwargs['x_pc'], ax=ax_loading1)
        # self.plot_loadings(pc=local_kwargs['y_pc'], ax=ax_loading2)
        # sns.despine()
        # fig.tight_layout()
        # return self

        # def splicing_movies(self):


# def plot_pca(df, **kwargs):
#     """ for backwards-compatibility """
#     pca = PCAViz(df, **kwargs)
#     pca.plot_samples()
#     return pca
