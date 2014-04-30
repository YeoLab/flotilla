__author__ = 'lovci, ppliu, obot, ....?'

import math
from math import sqrt

import numpy as np
from numpy.linalg import norm
import pandas as pd
import seaborn

seaborn.set_style({'axes.axisbelow': True,
                   'axes.edgecolor': '.15',
                   'axes.facecolor': 'white',
                   'axes.grid': False,
                   'axes.labelcolor': '.15',
                   'axes.linewidth': 1.25,
                   'font.family': 'Helvetica',
                   'grid.color': '.8',
                   'grid.linestyle': '-',
                   'image.cmap': 'Greys',
                   'legend.frameon': False,
                   'legend.numpoints': 1,
                   'legend.scatterpoints': 1,
                   'lines.solid_capstyle': 'round',
                   'text.color': '.15',
                   'xtick.color': '.15',
                   'xtick.direction': 'out',
                   'xtick.major.size': 0,
                   'xtick.minor.size': 0,
                   'ytick.color': '.15',
                   'ytick.direction': 'out',
                   'ytick.major.size': 0,
                   'ytick.minor.size': 0})

seaborn.set_color_palette('deep')

import pylab

from flotilla.src.frigate import PCA, NMF

def L1_distance(x,y):
    return abs(y) + abs(x)

def L2_distance(x,y):
    return math.sqrt((y ** 2) + (x ** 2))


class Reduction_viz(object):

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
        @param markers_size_dict: A dictionary of index (samples) to matplotlib marker sizes
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
        @param scale_by_variance: Boolean. Scale vector components by explained variance
        @return: x, y, marker, distance of each vector in the data.
        """

    _default_plotting_args = {'ax':None, 'x_pc':'pc_1', 'y_pc':'pc_2',
                      'num_vectors':20, 'title':'PCA', 'title_size':None, 'axis_label_size':None,
                      'colors_dict':None, 'markers_dict':None, 'markers_size_dict':None,
                      'default_marker_size':100, 'distance_metric':'L1',
                      'show_vectors':True, 'c_scale':None, 'vector_width':None, 'vector_colors_dict':None,
                      'show_vector_labels':True,  'vector_label_size':None,
                      'show_point_labels':True, 'point_label_size':None, 'scale_by_variance':True}

    _default_reduction_args = { 'n_components':None}

    _default_args = dict(_default_plotting_args.items() + _default_reduction_args.items())

    def __init__(self, df, **kwargs):

        self._validate_params(self._default_args, **kwargs)

        self.plotting_args = self._default_plotting_args.copy()
        self.plotting_args.update([(k,v) for (k,v) in kwargs.items() if k in self._default_plotting_args.keys()])

        self.reduction_args = self._default_reduction_args.copy()
        self.reduction_args.update([(k,v) for (k,v) in kwargs.items() if k in self._default_reduction_args.keys()])

        super(Reduction_viz, self).__init__(**self.reduction_args) #initialize PCA object
        assert type(df) == pd.DataFrame
        self.reduced_space = self.fit_transform(df)

    def __call__(self, ax=None, **kwargs):
        #self._validate_params(self._default_plotting_args, **kwargs)
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
        import pylab

        gs_x = 12
        gs_y = 12

        if ax is None:
            fig, ax = pylab.subplots(1,1,figsize=(18,9))
            gs = GridSpec(gs_x,gs_y)

        else:
            gs = GridSpecFromSubplotSpec(gs_x,gs_y,ax.get_subplotspec())

        ax_components = pylab.subplot(gs[:, :5])
        ax_loading1 = pylab.subplot(gs[1:5, 5:])
        ax_loading2 = pylab.subplot(gs[6:11, 5:])

        passed_kwargs = kwargs
        local_kwargs = self.plotting_args.copy()
        local_kwargs.update(passed_kwargs)
        local_kwargs.update({'ax':ax_components})
        self.plot_samples(**local_kwargs)
        self.plot_loadings(pc=local_kwargs['x_pc'], ax=ax_loading1)
        self.plot_loadings(pc=local_kwargs['y_pc'], ax=ax_loading2)
        pylab.tight_layout()
        return self


    def _validate_params(self, valid, **kwargs):

        for key in kwargs.keys():
            try:
                assert key in valid.keys()
            except:
                print self.__doc__
                raise ValueError("unrecognized parameter for pc plot: "\
                                 "%s. acceptable values are:\n%s" % (key, "\n".join(valid.keys())))

    def plot_samples(self, **kwargs):
        from pylab import gcf
        self._validate_params(self._default_plotting_args, **kwargs)
        default_params = self.plotting_args.copy() #fill missing parameters
        default_params.update(kwargs)
        kwargs = default_params

        #cheating!
        #move kwargs out of a dict, into local namespace, mostly because I don't want to refactor below

        for key in kwargs.keys():
            #
            # the following makes several errors appear in pycharm. they're not errors~~! laziness? :(
            #
            # imports variables from dictionaries and uses them as variable names in the code ... cheating because
            # TODO: needs to be refactored
            #
            exec(key + " = kwargs['" + key + "']")
        x_loading, y_loading = self.components_.ix[x_pc], self.components_.ix[y_pc]

        if ax is None:
            fig, ax = pylab.subplots(1,1, figsize=(5,5))
        self.ax = ax

        reduced_space = self.reduced_space
        x_list = reduced_space[x_pc]
        y_list = reduced_space[y_pc]

        if not c_scale:
            c_scale = .75 * max([norm(point) for point in zip(x_list, y_list)]) / \
                      max([norm(vector) for vector in zip(x_loading, y_loading)])

        figsize = tuple(gcf().get_size_inches())
        size_scale = sqrt(figsize[0] * figsize[1]) / 1.5
        default_marker_size = size_scale*5 if not default_marker_size else default_marker_size
        vector_width = .5 if not vector_width else vector_width
        axis_label_size = size_scale *1.5 if not axis_label_size else axis_label_size
        title_size = size_scale*2 if not title_size else title_size
        vector_label_size = size_scale * 1.5 if not vector_label_size else vector_label_size
        point_label_size = size_scale * 1.5 if not point_label_size else point_label_size

        # get amount of variance explained
        try:
            #not all reduction methods have this attr, if it doesn't assume equal , not true.. but easy!
            var_1 = int(self.explained_variance_ratio_[x_pc] * 100)
            var_2 = int(self.explained_variance_ratio_[y_pc] * 100)
        except AttributeError:
            var_1, var_2 = 1., 1.

        # sort features by magnitude/contribution to transformation
        comp_magn = []
        magnitudes = []
        for (x, y, an_id) in zip(x_loading, y_loading, self.X.columns):

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

        self.magnitudes = pd.Series(magnitudes, index=self.X.columns)
        self.magnitudes.sort(ascending=False)


        for (x, y, an_id) in zip(x_list, y_list, self.X.index):

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

            ax.scatter( x, y, marker=marker, color=color, s=marker_size, edgecolor='none')

        vectors = sorted(comp_magn, key=lambda item: item[3], reverse=True)[:num_vectors]

        for x, y, marker, distance in vectors:

            try:
                color = vector_colors_dict[marker]
            except:
                color = 'black'

            if show_vectors:
                ax.plot( [0, x], [0, y], color=color, linewidth=vector_width)

                if show_vector_labels:

                     ax.text(1.1*x, 1.1*y, marker, color=color, size=vector_label_size)

        ax.set_xlabel(
            'Principal Component {} (Explains {}% Of Variance)'.format(str(x_pc),
                str(var_1)), size=axis_label_size)
        ax.set_ylabel(
            'Principal Component {} (Explains {}% Of Variance)'.format(str(y_pc),
                str(var_2)), size=axis_label_size)
        ax.set_title(title, size=title_size)

        return comp_magn[:num_vectors], ax

    def plot_loadings(self, pc='pc_1', n_features=50, ax=None):

        import pylab
        x = self.components_.ix[pc].copy()
        x.sort(ascending=True)
        half_features = int(n_features/2)
        a = x[:half_features]
        b = x[-half_features:]
        if ax is None:
            ax = pylab.gca()
        ax.plot(np.r_[a,b], 'o')
        ax.set_xticks(np.arange(n_features))
        _ = ax.set_xticklabels(np.r_[a.index, b.index], rotation=90)
        ax.set_title("loadings on " + pc)
        x_offset = 0.5
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(left=xmin-x_offset, right=xmax-x_offset)

        seaborn.despine(ax=ax)

class PCA_viz(Reduction_viz, PCA):
    _default_reduction_args = { 'n_components':None, 'whiten':True}

class NMF_viz(Reduction_viz, NMF):
    pass

def plot_pca(df, **kwargs):
    """ for backwards-compatibility """
    pcaObj = PCA_viz(df, **kwargs)
    return_me, ax = pcaObj.plot_samples()
    return return_me

