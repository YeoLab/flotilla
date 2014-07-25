from collections import defaultdict
from itertools import cycle
import math
import sys

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import norm
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
    _default_reduction_kwargs = {}
    # _default_plotting_args = {'ax': None, 'x_pc': 'pc_1', 'y_pc': 'pc_2',
    #                           'num_vectors': 20,
    #                           'title': 'Dimensionality Reduction',
    #                           'title_size': 16, 'axis_label_size': 14,
    #                           'colors_dict': None, 'markers_dict': None,
    #                           'markers_size_dict': None,
    #                           'default_marker_size': 100,
    #                           'distance': 'L1',
    #                           'show_vectors': True, 'c_scale': None,
    #                           'vector_width': None, 'vector_colors_dict': None,
    #                           'show_vector_labels': True,
    #                           'vector_label_size': None,
    #                           'show_point_labels': False,
    #                           'point_label_size': None,
    #                           'scale_by_variance': True}
    # _default_reduction_args = {'n_components': None, 'whiten': False}
    # _default_args = dict(
    #     _default_plotting_args.items() + _default_reduction_args.items())

    def __init__(self, df, title='', n_components=None, whiten=False,
                 **kwargs):
        self.title = title
        kwargs.update(self._default_reduction_kwargs)
        # self._validate_params(self._default_args, **kwargs)

        # self.plotting_kwargs = self._default_plotting_args.copy()
        # self.plotting_kwargs.update(kwargs)
        #
        # self.reduction_kwargs = self._default_reduction_args.copy()
        # self.reduction_kwargs.update([(k, v) for (k, v) in kwargs.items() if
        #                               k in self._default_reduction_args])


        # This magically initializes the reducer like PCA or NMF
        super(DecompositionViz, self).__init__(n_components=n_components,
                                               whiten=whiten, **kwargs)

        assert isinstance(df, pd.DataFrame)
        self.df = df
        self.reduced_space = self.fit_transform(self.df)

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

        kwargs.update({'ax': ax_components})

        self.plot_samples(**kwargs)
        self.plot_loadings(pc=kwargs['x_pc'], ax=ax_loading1)
        self.plot_loadings(pc=kwargs['y_pc'], ax=ax_loading2)
        sns.despine()
        fig.tight_layout()
        return self

    # def _validate_params(self, valid, **kwargs):
    #
    #     for key in kwargs.keys():
    #         try:
    #             assert key in valid.keys()
    #         except:
    #             sys.stdout.write(self.__doc__)
    #             raise ValueError("unrecognized parameter for pc plot: "
    #                              "%s. acceptable values are:\n%s" % (
    #                                  key, "\n".join(valid.keys())))

    def plot_samples(self, groupby=None, x_pc='pc_1', y_pc='pc_2',
                     show_point_labels=True,
                     distance='L1', num_vectors=20, label_to_color=None,
                     label_to_marker=None,
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

        if label_to_color is None:
            colors = cycle(set1)
            def color_factory():
                return colors.next()
            label_to_color = defaultdict(color_factory)

        if label_to_marker is None:
            markers = cycle(['o', '^', 's', 'v', '*', 'D', 'h'])
            def marker_factory():
                return markers.next()
            label_to_color = defaultdict(marker_factory)

        if groupby is None:
            groupby = defaultdict(lambda: 'all samples')

        # Plot the samples
        for name, df in self.reduced_space.groupby(groupby, axis=0):
            #         print df
            # ind = df_pivot.index.isin(df.index)
            # x = X[ind]
            # y = Y[ind]
            # if three_d:
            #     z = Z[ind]
            color = label_to_color[name]
            marker = label_to_marker[name]
            x = df[x_pc]
            y = df[y_pc]
            ax.plot(x, y, color=color, marker=marker, linestyle='None',
                    label=name, markersize=markersize, alpha=0.75)
            if show_point_labels:
                for args in zip(x, y, df.index):
                    ax.text(*args)
            # if name in text_group:
            #     zipper = zip(x, y, z, df.index) if three_d else zip(x, y,
            #                                                         df.index)
            #     for args in zipper:
            #         ax.text(*args)
        # Plot vectors, if asked
        if show_vectors:
            loadings = self.components_.ix[[x_pc, y_pc]]
            vars = self.explained_variance_ratio_[[x_pc, y_pc]]

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
                    ax.annotate(vector_label, (x, y),
                                textcoords='offset points',
                                xytext=(x_offset, y_offset),
                                horizontalalignment=horizontalalignment)
                    # ax.text(1.1 * x, 1.1 * y, col_id)

        # get amount of variance explained
        try:
            # not all reduction methods have this attr, if it doesn't assume
            # equal , not true.. but easy!
            var_1 = int(self.explained_variance_ratio_[x_pc] * 100)
            var_2 = int(self.explained_variance_ratio_[y_pc] * 100)
        except AttributeError:
            var_1, var_2 = 1., 1.

        ax.set_xlabel(
            'Principal Component {} (Explains {}% Of Variance)'.format(
                str(x_pc), str(var_1)))
        ax.set_ylabel(
            'Principal Component {} (Explains {}% Of Variance)'.format(
                str(y_pc), str(var_2)))
        ax.set_title(title)

        if legend:
            ax.legend()
        sns.despine()

    def plot_samples_old(self, **kwargs):
        self._validate_params(self._default_plotting_args, **kwargs)
        # fill missing parameters
        default_params = self.plotting_kwargs.copy()
        default_params.update(kwargs)
        kwargs = default_params

        # cheating!
        # move kwargs out of a dict, into local namespace, mostly because I
        # don't want to refactor below

        for key in kwargs.keys():
            # the following makes several errors appear in pycharm.
            #  they're not errors~~! laziness? :(
            # imports variables from dictionaries and uses them as variable
            # names in the code ... cheating because
            # TODO.md: needs to be refactored
            exec (key + " = kwargs['" + key + "']")
        x_loading, y_loading = self.components_.ix[x_pc], self.components_.ix[
            y_pc]

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        self.ax = ax

        reduced_space = self.reduced_space
        x_list = reduced_space[x_pc]
        y_list = reduced_space[y_pc]

        if not c_scale:
            c_scale = .75 * max(
                [norm(point) for point in zip(x_list, y_list)]) / \
                      max([norm(vector) for vector in
                           zip(x_loading, y_loading)])

        figsize = tuple(plt.gcf().get_size_inches())
        size_scale = math.sqrt(figsize[0] * figsize[1]) / 1.1

        default_marker_size = size_scale * 20 if not default_marker_size else \
            default_marker_size

        vector_width = .5 if not vector_width else vector_width

        axis_label_size = size_scale * 3 if not axis_label_size \
            else axis_label_size

        title_size = size_scale * 2 if not title_size else title_size

        vector_label_size = size_scale * 1.5 if not vector_label_size \
            else vector_label_size

        point_label_size = size_scale * 1.5 if not point_label_size \
            else point_label_size

        # get amount of variance explained
        try:
            # not all reduction methods have this attr, if it doesn't assume
            # equal , not true.. but easy!
            var_1 = int(self.explained_variance_ratio_[x_pc] * 100)
            var_2 = int(self.explained_variance_ratio_[y_pc] * 100)
        except AttributeError:
            var_1, var_2 = 1., 1.

        # sort features by magnitude/contribution to transformation
        comp_magn = []
        magnitudes = []
        for (x, y, an_id) in zip(x_loading, y_loading, self.df.columns):
            x = x * c_scale
            y = y * c_scale

            # scale metric by explained variance
            if distance_metric == 'L1':
                if scale_by_variance:
                    mg = L1_distance((x * var_1), (y * var_2))
                else:
                    mg = L1_distance(x, y)

            elif distance_metric == 'L2':
                if scale_by_variance:
                    mg = L2_distance((x * var_1), (y * var_2))
                else:
                    mg = L2_distance(x, y)

            comp_magn.append((x, y, an_id, mg))
            magnitudes.append(mg)

        self.magnitudes = pd.Series(magnitudes, index=self.df.columns)
        self.magnitudes.sort(ascending=False)

        tiny = 0
        for (x, y, an_id) in zip(x_list, y_list, self.df.index):

            try:
                color = colors_dict[an_id]
            except:
                color = 'black'

            try:
                marker = markers_dict[an_id]
            except:
                marker = '.'

            try:
                marker_size = markers_size_dict[an_id]
            except:
                marker_size = default_marker_size

            if show_point_labels:
                ax.text(x, y, an_id, color=color, size=point_label_size)

            thresh = 0.00001
            if abs(x) < thresh and abs(y) < thresh:
                print "error with %s " % an_id
                tiny += 1
            ax.scatter(x, y, marker=marker, color=color, s=marker_size,
                       edgecolor='none')

        vectors = sorted(comp_magn, key=lambda item: item[3],
                         reverse=True)[:num_vectors]
        if show_vectors:

            for x, y, marker, distance in vectors:

                try:
                    color = vector_colors_dict[marker]
                except:
                    color = 'black'
                ax.plot([0, x], [0, y], color=color, linewidth=vector_width)

                if show_vector_labels:
                    ax.text(1.1 * x, 1.1 * y, marker, color=color,
                            size=vector_label_size)

        ax.set_xlabel(
            'Principal Component {} (Explains {}% Of Variance)'.format(
                str(x_pc),
                str(var_1)), size=10)
        ax.set_ylabel(
            'Principal Component {} (Explains {}% Of Variance)'.format(
                str(y_pc),
                str(var_2)), size=10)
        ax.set_title(title, size=title_size)

        return comp_magn[:num_vectors], ax

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

        ax.plot(dd, np.arange(len(dd)), 'o', label='hi')
        ax.set_yticks(np.arange(max(len(dd), n_features)))
        shorten = lambda x: '{}...'.format(x[:30]) if len(x) > 30 else x
        ax.set_yticklabels(map(shorten, labels))
        ax.set_title("Component " + pc)
        x_offset = max(dd) * .05
        ax.set_xlim(left=min(dd) - x_offset, right=max(dd) + x_offset)
        [lab.set_rotation(90) for lab in ax.get_xticklabels()]
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


class PCAViz(DecompositionViz, PCA):
    pass
    # _default_reduction_args = {'n_components': None, 'whiten': False}

    # def __init__(self, *args, **kwargs):
    #     DecompositionViz.__init__(self, *args, **kwargs)
    #     PCA.__init__(self, **self.reduction_kwargs)
    #     self.binned_reduced = self.fit_transform(self.df)


class NMFViz(DecompositionViz, NMF):
    _default_reduction_kwargs = {'n_components': 2,
                               'max_iter': 20000, 'nls_max_iter': 40000}

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
    return_me, ax = pca.plot_samples()
    return return_me
