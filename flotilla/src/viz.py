__author__ = 'lovci, ppliu, obot, ....?'

import math
from math import sqrt

import numpy as np
from numpy.linalg import norm
import pandas as pd
import seaborn
from ..project.project_params import _default_group_id

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

from flotilla.src.compute import PCA, NMF

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
                      'num_vectors':20, 'title':'Dimensionality Reduction', 'title_size':None, 'axis_label_size':None,
                      'colors_dict':None, 'markers_dict':None, 'markers_size_dict':None,
                      'default_marker_size':100, 'distance_metric':'L1',
                      'show_vectors':True, 'c_scale':None, 'vector_width':None, 'vector_colors_dict':None,
                      'show_vector_labels':True,  'vector_label_size':None,
                      'show_point_labels':True, 'point_label_size':None, 'scale_by_variance':True}

    _default_reduction_args = { 'n_components':None, 'whiten':False}

    _default_args = dict(_default_plotting_args.items() + _default_reduction_args.items())

    def __init__(self, df, **kwargs):

        self._validate_params(self._default_args, **kwargs)

        self.plotting_args = self._default_plotting_args.copy()
        self.plotting_args.update([(k,v) for (k,v) in kwargs.items() if k in self._default_plotting_args.keys()])

        self.reduction_args = self._default_reduction_args.copy()
        self.reduction_args.update([(k,v) for (k,v) in kwargs.items() if k in self._default_reduction_args.keys()])

        super(Reduction_viz, self).__init__(**self.reduction_args) #initialize PCA-like object
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
        ax.set_xticklabels(map(str, np.arange(self.n_components)+1))
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Fraction explained variance')
        ax.set_title(title)
        sns.despine()
        return fig

class PCA_viz(Reduction_viz, PCA):
    _default_reduction_args = { 'n_components':None, 'whiten':False}

class NMF_viz(Reduction_viz, NMF):
    pass

def plot_pca(df, **kwargs):
    """ for backwards-compatibility """
    pcaObj = PCA_viz(df, **kwargs)
    return_me, ax = pcaObj.plot_samples()
    return return_me


def lavalamp(psi, color=None, title='', ax=None):
    from .compute import get_switchy_score_order
    """Make a 'lavalamp' scatter plot of many spliciang events

    Useful for visualizing many splicing events at once.

    Parameters
    ----------
    df : array
        A (n_events, n_samples) matrix either as a numpy array or as a pandas
        DataFrame

    color : matplotlib color
        Color of the scatterplot. Defaults to a dark teal

    title : str
        Title of the plot. Default ''

    ax : matplotlib.Axes object
        The axes to plot on. If not provided, will be created


    Returns
    -------
    fig : matplotlib.Figure
        A figure object for saving.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(16,4))
    else:
        fig = plt.gcf()
    nrow, ncol = psi.shape
    x = np.vstack(np.arange(nrow) for _ in range(ncol))

    color = '#FFFFFF' if color is None else color

    try:
        # This is a pandas Dataframe
        y = psi.values.T
    except AttributeError:
        # This is a numpy array
        y = psi.T

    order = get_switchy_score_order(y)
    y = y[:, order]

    # Add one so the last value is actually included instead of cut off
    xmax = x.max() + 1
    ax.scatter(x, y, color=color, alpha=0.5, edgecolor='#262626', linewidth=0.1)
    seaborn.despine()
    ax.set_ylabel('$\Psi$')
    ax.set_xlabel('{} splicing events'.format(nrow))
    ax.set_xticks([])

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_title(title)

    # Return the figure for saving
    # return fig

from .compute import Networker
from matplotlib.gridspec import GridSpec

import networkx as nx
from .utils import dict_to_str

class Networker_Viz(Networker, Reduction_viz):

    def __init__(self, data_obj):
        self.data_obj = data_obj

    def draw_graph(self, list_name='default', group_id=_default_group_id,
                   x_pc=1, y_pc=2,
                   n_pcs=5,
                   pc_1=True, pc_2=True, pc_3=True, pc_4=True,
                   degree_cut=2, cov_std_cut = 1.8,
                   wt_fun = 'abs',
                   featurewise=False, #else feature_components
                   rpkms_not_events=False, #else event features
                   feature_of_interest='RBFOX2', custom_list='', draw_labels=True,
                   graph_file=''):

        """
        genelist_name - name of genelist used in making pcas
        group_id - celltype code
        x_pc - x component for PCA
        y_pc - y component for PCA
        n_pcs - n components to use for cells' covariance calculation
        cov_std_cut - covariance cutoff for edges
        pc{1-4} use these pcs in cov calculation (default True)
        degree_cut - miniumum degree for a node to be included in graph display
        wt_fun - weight function (arctan (arctan cov), sq (sq cov), abs (abs cov), arctan_sq (sqared arctan of cov))
        gene_of_interest - map a gradient representing this gene's rpkm onto nodes



        """
        node_color_mapper = self._default_node_color_mapper
        node_size_mapper = self._default_node_color_mapper
        settings = locals().copy()
        #not pertinent to the graph, these are what we want to be able to re-apply to the same graph if it exists
        pca_settings = dict()
        pca_settings['group_id'] = group_id

        adjacency_settings = dict((k, settings[k]) for k in ['pc_1', 'pc_2', 'pc_3', 'pc_4', 'n_pcs'])

        #del settings['gene_of_interest']
        #del settings['graph_file']
        #del settings['draw_labels']

        f = pylab.figure(figsize=(24,18))
        gs = GridSpec(3, 4)
        ax1 = pylab.subplot(gs[0,0:2])
        ax2 = pylab.subplot(gs[0,3])
        ax3 = pylab.subplot(gs[0,2])
        ax4 = pylab.subplot(gs[1:,:2])
        ax5 = pylab.subplot(gs[1:,2:])

        if custom_list != '':
            list_name = custom_list

        #decide which type of analysis to do.

        pca = self.data_obj.get_reduced(list_name, group_id)

        pca(show_point_labels=False,
            markers_size_dict=lambda x: 400,
            title=group_id + " cells", ax=ax1, show_vectors=False,
            title_size=10,
            axis_label_size=10,
            x_pc = "pc_" + x_pc,#this only affects the plot, not the data.
            y_pc = "pc_" + y_pc,#this only affects the plot, not the data.
            )

        if featurewise:
            node_color_mapper = lambda x: 'r' if x == feature_of_interest else 'k'
            node_size_mapper = lambda x: pca.X.feature_of_interest.ix[x]
        else:
            node_color_mapper = lambda x: self.sample_info.cell_color[x]
            node_size_mapper = lambda x: 300

        ax3.plot(pca.explained_variance_ratio_ * 100.)
        ax3.axvline(n_pcs)
        ax3.set_ylabel("% explained variance")
        ax3.set_xlabel("component")
        adjacency_name = "_".join(map(dict_to_str, [adjacency_settings, pca_settings]))
        #adjacency_settings['name'] = adjacency_name
        adjacency = self.get_adjacency(pca.pca_space, name=adjacency_name, **adjacency_settings)
        #f.savefig("tmp/" + fname + ".pca.png")
        cov_dist = np.array([i for i in adjacency.values.ravel() if np.abs(i) > 0])
        cov_cut = np.mean(cov_dist) + cov_std_cut * np.std(cov_dist)

        graph_settings = dict((k, settings[k]) for k in ['wt_fun', 'degree_cut', ])
        graph_settings['cov_cut'] = cov_cut

        this_graph_name = "_".join(map(dict_to_str, [pca_settings, adjacency_settings, graph_settings]))

        seaborn.kdeplot(cov_dist, ax=ax2)
        ax2.axvline(cov_cut)

        g, pos = self.get_graph(adjacency, cov_cut, this_graph_name, **graph_settings)

        nx.draw_networkx_nodes(g, pos, node_color=map(node_color_mapper, g.nodes()),
                               node_size=map(node_size_mapper, g.nodes()),
                               ax=ax4, alpha=0.5)
        nx.draw_networkx_nodes(g, pos, node_color=map(node_color_mapper, g.nodes()),
                               node_size=map(node_size_mapper, g.nodes()),
                               ax=ax5, alpha=0.5)
        try:
            nx.draw_networkx_nodes(g, pos, node_color=map(lambda x: pca.X[feature_of_interest].ix[x], g.nodes()),
                               cmap=pylab.cm.Greys,
                               node_size=map(lambda x: node_size_mapper(x) * .75, g.nodes()), ax=ax5, alpha=1)
        except:
            pass
        nmr = lambda x:x
        labels = dict([(nm, nmr(nm)) for nm in g.nodes()])
        if draw_labels:
            nx.draw_networkx_labels(g, pos, labels = labels, ax=ax4)
        #mst = nx.minimum_spanning_tree(g, weight='inv_weight')
        nx.draw_networkx_edges(g, pos,ax = ax4,alpha=0.1)
        #nx.draw_networkx_edges(g, pos, edgelist=mst.edges(), edge_color="m", edge_width=200, ax=ax4)
        ax4.set_axis_off()
        ax5.set_axis_off()
        f.tight_layout(pad=5)
        if graph_file != '':
            try:
                nx.write_gml(g, graph_file)
            except Exception as e:
                print "error writing graph file:"
                print e

        return g#, mst
