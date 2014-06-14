__author__ = 'lovci, ppliu, obot, ....?'

""""plotting tools"""

import math
from math import sqrt

import numpy as np
from numpy.linalg import norm
import pandas as pd
import seaborn
# from ..neural_diff_project.project_params import _default_group_id

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

seaborn.set_palette('deep')
blue = seaborn.color_palette()[0]
green =seaborn.color_palette()[1]
red = seaborn.color_palette()[2]

import matplotlib.pyplot as plt

from ._frigate_compute import PCA, NMF, TwoWayGeneComparisonLocal, Networker, nx
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from ._barge_utils import dict_to_str

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
        @return: x, y, marker, distance of each vector in the study_data.
        """

    _default_plotting_args = {'ax':None, 'x_pc':'pc_1', 'y_pc':'pc_2',
                      'num_vectors':20, 'title':'Dimensionality Reduction',
                      'title_size':None, 'axis_label_size':None,
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
        gs_x = 14
        gs_y = 12

        if ax is None:

            fig, ax = plt.subplots(1,1,figsize=(12,6))
            gs = GridSpec(gs_x,gs_y)

        else:
            gs = GridSpecFromSubplotSpec(gs_x,gs_y,ax.get_subplotspec())

        ax_components = plt.subplot(gs[:, :5])
        #ax_components.set_aspect('equal')
        ax_loading1 = plt.subplot(gs[:, 6:8])
        ax_loading2 = plt.subplot(gs[:, 10:14])

        passed_kwargs = kwargs
        local_kwargs = self.plotting_args.copy()
        local_kwargs.update(passed_kwargs)
        local_kwargs.update({'ax':ax_components})
        self.plot_samples(**local_kwargs)
        self.plot_loadings(pc=local_kwargs['x_pc'], ax=ax_loading1)
        self.plot_loadings(pc=local_kwargs['y_pc'], ax=ax_loading2)
        plt.tight_layout()
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
            fig, ax = plt.subplots(1,1, figsize=(5,5))
        self.ax = ax

        reduced_space = self.reduced_space
        x_list = reduced_space[x_pc]
        y_list = reduced_space[y_pc]

        if not c_scale:
            c_scale = .75 * max([norm(point) for point in zip(x_list, y_list)]) / \
                      max([norm(vector) for vector in zip(x_loading, y_loading)])

        figsize = tuple(plt.gcf().get_size_inches())
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

        tiny=0
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
            if x>=-0.00001 and x<=0.00001 and y>=-0.00001 and y<=0.00001:
                print "error with %s " % an_id
                tiny+=1
            ax.scatter( x, y, marker=marker, color=color, s=marker_size, edgecolor='none')

        #print "there were %d errors of %d" % (tiny, len(x_list))
        vectors = sorted(comp_magn, key=lambda item: item[3], reverse=True)[:num_vectors]
        if show_vectors:

            for x, y, marker, distance in vectors:

                try:
                    color = vector_colors_dict[marker]
                except:
                    color = 'black'
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

        x = self.components_.ix[pc].copy()
        x.sort(ascending=True)
        half_features = int(n_features/2)
        if len(x) > half_features:

            a = x[:half_features]
            b = x[-half_features:]
            dd = np.r_[a,b]
            labels = np.r_[a.index, b.index]

        else:
            dd = x
            labels=x.index

        if ax is None:
            ax = plt.gca()

        ax.plot(dd,np.arange(len(dd)), 'o', label='hi')
        ax.set_yticks(np.arange(max(len(dd), n_features)))
        shorten = lambda x: "id too long" if len(x) > 30 else x
        _ = ax.set_yticklabels(map(shorten, labels))#, rotation=90)
        ax.set_title("loadings on " + pc)
        x_offset = max(dd) * .05
        #xmin, xmax = ax.get_xlim()
        ax.set_xlim(left=min(dd)-x_offset, right=max(dd)+x_offset)
        [lab.set_rotation(90) for lab in ax.get_xticklabels()]
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


def lavalamp(psi, color=None, jitter=None, title='', ax=None):
    """Make a 'lavalamp' scatter plot of many spliciang events

    Useful for visualizing many splicing events at once.

    Parameters
    ----------
    TODO: (n_events, n_samples).transpose()
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
    from ._frigate_compute import get_switchy_score_order
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
        y = psi.values
    except AttributeError:
        # This is a numpy array
        y = psi

    if jitter is None:
        jitter = np.zeros(len(color))
    else:
        assert np.all(np.abs(jitter) < 1)
        assert np.min(jitter) > -.0000000001

    order = get_switchy_score_order(y.T)
    print order.shape
    y = y[:, order]
    assert type(color) == pd.Series
    # Add one so the last value is actually included instead of cut off
    xmax = x.max() + 1
    x_jitter = np.apply_along_axis(lambda r: r+jitter, 0, x)

    for co, ji, xx, yy in zip(color, jitter, x_jitter, y.T):
        ax.scatter(xx, yy, color=co, alpha=0.5, edgecolor='#262626', linewidth=0.1)
    seaborn.despine()
    ax.set_ylabel('$\Psi$')
    ax.set_xlabel('{} splicing events'.format(nrow))
    ax.set_xticks([])

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_title(title)

    # Return the figure for saving
    # return fig

class NetworkerViz(Networker, Reduction_viz):
    #TODO: needs to be decontaminated, as it requires methods from data_object;
    #maybe this class should move to schooner.Data
    def __init__(self, data_obj):
        self.data_obj = data_obj
        Networker.__init__(self)

    def draw_graph(self,
                   n_pcs=5,
                   use_pc_1=True, use_pc_2=True, use_pc_3=True, use_pc_4=True,
                   degree_cut=2, cov_std_cut = 1.8,
                   wt_fun = 'abs',
                   featurewise=False, #else feature_components
                   rpkms_not_events=False, #else event features
                   feature_of_interest='RBFOX2', draw_labels=True,
                   reduction_name=None,
                   list_name=None,
                   group_id=None,
                   graph_file='',
                   compare=""):

        """
        list_name - name of genelist used in making pcas
        group_id - celltype code
        x_pc - x component for PCA
        y_pc - y component for PCA
        n_pcs - n components to use for cells' covariance calculation
        cov_std_cut - covariance cutoff for edges
        pc{1-4} use these pcs in cov calculation (default True)
        degree_cut - miniumum degree for a node to be included in graph display
        wt_fun - weight function (arctan (arctan cov), sq (sq cov), abs (abs cov), arctan_sq (sqared arctan of cov))
        gene_of_interest - map a gradient representing this gene's df onto nodes
        """

        node_color_mapper = self._default_node_color_mapper
        node_size_mapper = self._default_node_color_mapper
        settings = locals().copy()
        #not pertinent to the graph, these are what we want to be able to re-apply to the same graph if it exists
        pca_settings = dict()
        pca_settings['group_id'] = group_id
        pca_settings['featurewise'] = featurewise
        pca_settings['list_name'] = list_name
        pca_settings['obj_id'] = reduction_name

        adjacency_settings = dict((k, settings[k]) for k in ['use_pc_1', 'use_pc_2', 'use_pc_3', 'use_pc_4', 'n_pcs', ])

        f= plt.figure(figsize=(10,10))

        plt.axis((-0.2, 1.2, -0.2, 1.2))
        main_ax = plt.gca()
        ax_pev = plt.axes([0.1, .8, .2, .15])
        ax_cov = plt.axes([0.1, 0.1, .2, .15])
        ax_degree = plt.axes([0.9,.8,.2,.15])

        pca = self.data_obj.get_reduced(**pca_settings)


        if featurewise:
            node_color_mapper = lambda x: 'r' if x == feature_of_interest else 'k'
            node_size_mapper = lambda x: (pca.means.ix[x]**2) + 10
        else:
            node_color_mapper = lambda x: self.data_obj.sample_metadata.color[x]
            node_size_mapper = lambda x: 75

        ax_pev.plot(pca.explained_variance_ratio_ * 100.)
        ax_pev.axvline(n_pcs, label='cutoff')
        ax_pev.legend()
        ax_pev.set_ylabel("% explained variance")
        ax_pev.set_xlabel("component")
        ax_pev.set_title("Explained variance from dim reduction")
        seaborn.despine(ax=ax_pev)

        adjacency_name = "_".join([dict_to_str(adjacency_settings), pca.obj_id])

        adjacency = self.get_adjacency(pca.reduced_space, name=adjacency_name, **adjacency_settings)
        cov_dist = np.array([i for i in adjacency.values.ravel() if np.abs(i) > 0])
        cov_cut = np.mean(cov_dist) + cov_std_cut * np.std(cov_dist)

        graph_settings = dict((k, settings[k]) for k in ['wt_fun', 'degree_cut', ])
        graph_settings['cov_cut'] = cov_cut
        this_graph_name = "_".join(map(dict_to_str, [pca_settings, adjacency_settings, graph_settings]))
        graph_settings['name'] = this_graph_name

        seaborn.kdeplot(cov_dist, ax=ax_cov)
        ax_cov.axvline(cov_cut, label='cutoff')
        ax_cov.set_title("covariance in dim reduction space")
        ax_cov.set_ylabel("density")
        ax_cov.legend()
        seaborn.despine(ax=ax_cov)
        g, pos = self.get_graph(adjacency, **graph_settings)

        nx.draw_networkx_nodes(g, pos, node_color=map(node_color_mapper, g.nodes()),
                               node_size=map(node_size_mapper, g.nodes()),
                               ax=main_ax, alpha=0.5)

        try:
            nx.draw_networkx_nodes(g, pos, node_color=map(lambda x: pca.X[feature_of_interest].ix[x], g.nodes()),
                                   cmap=plt.cm.Greys,
                                   node_size=map(lambda x: node_size_mapper(x) * .5, g.nodes()), ax=main_ax, alpha=1)
        except:
            pass
        nmr = lambda x:x
        labels = dict([(nm, nmr(nm)) for nm in g.nodes()])
        if draw_labels:
            nx.draw_networkx_labels(g, pos, labels = labels, ax=main_ax)
        #mst = nx.minimum_spanning_tree(g, weight='inv_weight')
        nx.draw_networkx_edges(g, pos,ax = main_ax,alpha=0.1)
        #nx.draw_networkx_edges(g, pos, edgelist=mst.edges(), edge_color="m", edge_width=200, ax=main_ax)
        main_ax.set_axis_off()
        degree = nx.degree(g)
        seaborn.kdeplot(np.array(degree.values()), ax=ax_degree)
        ax_degree.set_xlabel("degree")
        ax_degree.set_ylabel("density")
        try:
            ax_degree.axvline(x=degree[feature_of_interest], label=feature_of_interest)
            ax_degree.legend()

        except Exception as e:
            print e
            pass

        seaborn.despine(ax=ax_degree)
        #f.tight_layout(pad=5)
        if graph_file != '':
            try:
                nx.write_gml(g, graph_file)
            except Exception as e:
                print "error writing graph file:"
                print e

        return g, pos

    def draw_nonreduced_graph(self,
                   degree_cut=2, cov_std_cut = 1.8,
                   wt_fun = 'abs',
                   featurewise=False, #else feature_components
                   rpkms_not_events=False, #else event features
                   feature_of_interest='RBFOX2', draw_labels=True,
                   list_name=None,
                   group_id=None,
                   graph_file='',
                   compare=""):

        """
        list_name - name of genelist used in making pcas
        group_id - celltype code
        x_pc - x component for PCA
        y_pc - y component for PCA
        n_pcs - n components to use for cells' covariance calculation
        cov_std_cut - covariance cutoff for edges
        pc{1-4} use these pcs in cov calculation (default True)
        degree_cut - miniumum degree for a node to be included in graph display
        wt_fun - weight function (arctan (arctan cov), sq (sq cov), abs (abs cov), arctan_sq (sqared arctan of cov))
        gene_of_interest - map a gradient representing this gene's df onto nodes
        """

        node_color_mapper = self._default_node_color_mapper
        node_size_mapper = self._default_node_color_mapper
        settings = locals().copy()

        adjacency_settings = dict(('non_reduced', True))

        #del settings['gene_of_interest']
        #del settings['graph_file']
        #del settings['draw_labels']

        f= plt.figure(figsize=(10,10))
        #gs = GridSpec(2, 2)
        plt.axis((-0.2, 1.2, -0.2, 1.2))
        main_ax = plt.gca()
        ax_cov = plt.axes([0.1, 0.1, .2, .15])
        ax_degree = plt.axes([0.9,.8,.2,.15])

        data = self.data_obj.df


        if featurewise:
            node_color_mapper = lambda x: 'r' if x == feature_of_interest else 'k'
            node_size_mapper = lambda x: (data.mean().ix[x]**2) + 10
        else:
            node_color_mapper = lambda x: self.data_obj.sample_metadata.color[x]
            node_size_mapper = lambda x: 75

        adjacency_name = "_".join([dict_to_str(adjacency_settings)])
        #adjacency_settings['name'] = adjacency_name

        #import pdb
        #pdb.set_trace()
        adjacency = self.get_adjacency(data, name=adjacency_name, **adjacency_settings)
        cov_dist = np.array([i for i in adjacency.values.ravel() if np.abs(i) > 0])
        cov_cut = np.mean(cov_dist) + cov_std_cut * np.std(cov_dist)

        graph_settings = dict((k, settings[k]) for k in ['wt_fun', 'degree_cut', ])
        graph_settings['cov_cut'] = cov_cut
        this_graph_name = "_".join(map(dict_to_str, [adjacency_settings, graph_settings]))
        graph_settings['name'] = this_graph_name

        seaborn.kdeplot(cov_dist, ax=ax_cov)
        ax_cov.axvline(cov_cut, label='cutoff')
        ax_cov.set_title("covariance in original space")
        ax_cov.set_ylabel("density")
        ax_cov.legend()
        seaborn.despine(ax=ax_cov)
        g, pos = self.get_graph(adjacency, **graph_settings)

        nx.draw_networkx_nodes(g, pos, node_color=map(node_color_mapper, g.nodes()),
                               node_size=map(node_size_mapper, g.nodes()),
                               ax=main_ax, alpha=0.5)
        #nx.draw_networkx_nodes(g, pos, node_color=map(node_color_mapper, g.nodes()),
        #                       node_size=map(node_size_mapper, g.nodes()),
        #                       ax=ax4, alpha=0.5)
        try:
            nx.draw_networkx_nodes(g, pos, node_color=map(lambda x: data[feature_of_interest].ix[x], g.nodes()),
                                   cmap=plt.cm.Greys,
                                   node_size=map(lambda x: node_size_mapper(x) * .5, g.nodes()), ax=main_ax, alpha=1)
        except:
            pass
        nmr = lambda x:x
        labels = dict([(nm, nmr(nm)) for nm in g.nodes()])
        if draw_labels:
            nx.draw_networkx_labels(g, pos, labels = labels, ax=main_ax)
        #mst = nx.minimum_spanning_tree(g, weight='inv_weight')
        nx.draw_networkx_edges(g, pos,ax = main_ax,alpha=0.1)
        #nx.draw_networkx_edges(g, pos, edgelist=mst.edges(), edge_color="m", edge_width=200, ax=main_ax)
        main_ax.set_axis_off()
        degree = nx.degree(g)
        seaborn.kdeplot(np.array(degree.values()), ax=ax_degree)
        ax_degree.set_xlabel("degree")
        ax_degree.set_ylabel("density")
        try:
            ax_degree.axvline(x=degree[feature_of_interest], label=feature_of_interest)
            ax_degree.legend()

        except Exception as e:
            print e
            pass

        seaborn.despine(ax=ax_degree)
        #f.tight_layout(pad=5)
        if graph_file != '':
            try:
                nx.write_gml(g, graph_file)
            except Exception as e:
                print "error writing graph file:"
                print e

        return g, pos

from ._frigate_compute import Predictor
import itertools

class PredictorViz(Predictor, Reduction_viz):

    _reducer_plotting_args = {}
    def set_reducer_plotting_args(self, rpa):
        self._reducer_plotting_args.update(rpa)

    def __call__(self, trait=None, ax=None, feature_score_std_cutoff=None):

        if trait is None:
            trait = self.traits[0]
        else:
            assert type(trait) == str or type(trait) == unicode

        if feature_score_std_cutoff is None:
            feature_scoring_cut_fun = self.default_classifier_scoring_cutoff_fun
        else:
            feature_scoring_cut_fun = lambda scores: np.mean(scores) + feature_score_std_cutoff*np.std(scores)

        if not self.has_been_fit_yet:
            self.fit_classifiers([trait])

        self.score_classifiers([trait], score_cutoff_fun=feature_scoring_cut_fun)

        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        import matplotlib.pyplot as plt
        gs_x = 18
        gs_y = 12

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(18, 8))
            gs = GridSpec(gs_x,gs_y)

        else:
            gs = GridSpecFromSubplotSpec(gs_x,gs_y,ax.get_subplotspec())
        ax_pca = plt.subplot(gs[:, 2:])

        ax_scores= plt.subplot(gs[5:10, :2])
        ax_scores.set_xlabel("Feature Importance")
        ax_scores.set_ylabel("Density Estimate")
        self.plot_classifier_scores([trait], ax=ax_scores)
        pca = self.do_pca(trait, ax=ax_pca, show_vectors=True)
        plt.tight_layout()
        return pca


    def plot_classifier_scores(self, traits, ax=None, classifier_name=None):
        """
        plot kernel density of classifier scores and draw a vertical line where the cutoff was selected
        ax - ax to plot on. if None: plt.gca()
        """

        if classifier_name is None:
            classifier_name = self.default_classifier_name

        if ax==None:
            ax = plt.gca()

        for trait in traits:
            clf = self.classifiers_[trait][classifier_name]
            seaborn.kdeplot(clf.scores_, shade=True, ax=ax, label="%s\n%d features\noob:%.2f" % (trait,
                                                                                       np.sum(clf.good_features_),
            clf.oob_score_))
            ax.axvline(x=clf.score_cutoff_)

        [lab.set_rotation(90) for lab in ax.get_xticklabels()]
        seaborn.despine(ax=ax)


    def generate_scatter_table(self,
                              excel_out=None, external_xref=[]):
        """
        make a table to make scatterplots... maybe for plot.ly
        excelOut: full path to an excel output location for scatter data
        external_xref: list of tuples containing (attribute name, function to map row index -> an attribute)
        """
        raise NotImplementedError
        trait, classifier_name = self.attributes
        X = self.X
        sorter = np.array([np.median(i[1]) - np.median(j[1]) for (i, j) in \
                           itertools.izip(X[self.y[trait]==0].iteritems(),
                                          X[self.y[trait]==1].iteritems())])

        sort_by_sorter = X.columns[np.argsort(sorter)]
        c0_values = X[sort_by_sorter][self.y[trait]==0]
        c1_values = X[sort_by_sorter][self.y[trait]==1]

        x = []
        s = []
        y1 = []
        y2 = []
        field_names = ['x-position', 'probe intensity', "condition0", "condition1"]
        n_default_fields = len(field_names)
        external_attrs = {}
        for external_attr_name, external_attr_fun in external_xref:
            external_attrs[external_attr_name] = []
            field_names.append(external_attr_name)


        for i, (a, b) in enumerate(itertools.izip(c0_values.iteritems(), c1_values.iteritems())):

            mn = np.mean(np.r_[a[1], b[1]])
            _ = [x.append(i) for _ in a[1]]
            _ = [s.append(mn) for val in a[1]]
            _ = [y1.append(val- mn) for val in a[1]]
            _ = [y2.append(np.nan) for val in a[1]]

            _ = [x.append(i) for _ in b[1]]
            _ = [s.append(mn) for val in b[1]]
            _ = [y1.append(np.nan) for val in b[1]]
            _ = [y2.append(val - mn) for val in b[1]]


            for external_attr_name, external_attr_fun in external_xref:
                external_attrs[external_attr_name].extend([external_attr_fun(i) for i in a[1].index])
                external_attrs[external_attr_name].extend([external_attr_fun(i) for i in b[1].index])

        zz = pd.DataFrame([x, s, y1, y2] + [external_attrs[attr] for attr in field_names[n_default_fields:]],
                          index=field_names)

        if excel_out is not None:
            try:
                E = pd.ExcelWriter('%s' % excel_out)
                zz.T.to_excel(E, self.descrip)
                E.save()
            except Exception as e:
                print "excel save failed with error %s" % e

        return zz

    def check_a_feature(self, feature_name, traits=None,  **vp_params):
        """Make Violin Plots for a gene/probe's value in the sets defined in sets
        feature_name - gene/probe id. must be in the index of self._parent.X
        sets - list of sample ids
        vp_params - extra parameters for violinplot

        returns a list of lists with values for feature_name in each set of sets
        """
        if traits is None:
            traits = self.categorical_traits

        for trait in traits:
            seaborn.violinplot(self.X[feature_name], linewidth=0, groupby=trait,
                   alpha=0.5, bw='silverman', inner='points', names=None, **vp_params)
        seaborn.despine()


    def do_pca(self, trait, ax=None, classifier_name=None, **plotting_args):

        """
        plot kernel density of classifier scores and draw a vertical line where the cutoff was selected
        ax - ax to plot on. if None: plt.gca()
        """

        assert trait in self.traits
        assert self.has_been_fit_yet
        assert self.has_been_scored_yet

        if ax is None:
            ax = plt.gca()
        if classifier_name is None:
            classifier_name = self.default_classifier_name

        local_plotting_args = self._reducer_plotting_args
        local_plotting_args.update(plotting_args)
        pca = PCA_viz(self.X.ix[:, self.classifiers_[trait][classifier_name].good_features_], **local_plotting_args)
        pca(ax=ax)
        return pca

def clusterGram(dataFrame, distance_metric = 'euclidean', linkage_method = 'average',
            outfile = None, clusterRows=True, clusterCols=True, timeSeries=False, doCovar=False,
            figsize=(8, 10), row_label_color_fun=lambda x: 'k',
            col_label_color_fun=lambda x: 'k',
            link_color_func = lambda x: 'k'):
    import scipy
    from scipy import cluster
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    import matplotlib as mpl
    """

    Run hierarchical clustering on data. Creates a heatmap of cluster-ordered data
    heavy-lifting is done by:

    gets distances between rows/columns

    y_events = scipy.spatial.distance.pdist(data, distance_metric)

    calculates the closest rows/columns

    Z_events = scipy.cluster.hierarchy.linkage(y_events, linkage_method)

    genereates dendrogram (tree)

    d_events = scipy.cluster.hierarchy.dendrogram(Z_events, no_labels=True)

    set outfile == "None" to inibit saving an eps file (only show it, don't save it)

    """
    data = np.array(dataFrame)
    colLabels = dataFrame.columns
    rowLabels = dataFrame.index
    nRow, nCol = data.shape

    if clusterRows:
        print "getting row distance matrix"
        y_events = scipy.spatial.distance.pdist(data, distance_metric)
        print "calculating linkages"
        Z_events = scipy.cluster.hierarchy.linkage(y_events, linkage_method, metric=distance_metric)

    if clusterCols:
        print "getting column distance matrix"
        y_samples = scipy.spatial.distance.pdist(np.transpose(data), distance_metric)
        print "calculating linkages"
        Z_samples = scipy.cluster.hierarchy.linkage(y_samples, linkage_method, metric=distance_metric)
    else:
        if doCovar:
            raise ValueError

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(18,10)

    ax1 = plt.subplot(gs[1:, 0:2]) #row dendrogram

    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_frame_on(False)

    reordered = data
    event_order = range(nRow)
    if clusterRows:
        d_events = scipy.cluster.hierarchy.dendrogram(Z_events, orientation='right',
                                                      link_color_func=link_color_func,
                                                      labels=rowLabels)
        event_order = d_events['leaves']
        reordered = data[event_order,:]

    labels = ax1.get_yticklabels()


    ax2 = plt.subplot(gs[0:1, 2:9]) #column dendrogram

    ax2.set_yticklabels([])
    ax2.set_yticks([])
    ax2.set_frame_on(False)

    sample_order = range(nCol)
    if clusterCols:
        d_samples = scipy.cluster.hierarchy.dendrogram(Z_samples, labels=colLabels, leaf_rotation=90,
                                                       link_color_func=link_color_func)
        sample_order = d_samples['leaves']
        reordered = reordered[:,sample_order]

    axmatrix = plt.subplot(gs[1:, 2:9])
    bds = np.max(abs(reordered))
    if timeSeries:
        norm = mpl.colors.Normalize(vmin=-bds, vmax=bds)
    else:
        norm = None

    if (np.max(reordered) * np.min(reordered)) > 0:
        cmap = plt.cm.Reds
    else:
        cmap= plt.cm.RdBu_r

    im = axmatrix.matshow(reordered, aspect='auto', origin='lower', cmap=cmap, norm=norm)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    axcolor = plt.subplot(gs[1:6, -1])

    cbTicks = [np.min(data), np.mean(data), np.max(data)]
    cb = plt.colorbar(im, cax=axcolor, ticks=cbTicks, use_gridspec=True)
    plt.draw()
    [i.set_color(row_label_color_fun(i.get_text())) for i in ax1.get_yticklabels()]
    [i.set_color(col_label_color_fun(i.get_text())) for i in ax2.get_xticklabels()]

    plt.tight_layout()

    if outfile is not None:
        fig.savefig(outfile)
    return event_order, sample_order


class TwoWayScatterViz(TwoWayGeneComparisonLocal):

    def __call__(self, **kwargs):
        self.plot(**kwargs)

    def plot(self, ax=None):

        co = [] #colors container
        for label, (pVal, logratio, isSig) in self.result_.get(["pValue", "log2Ratio", "isSig"]).iterrows():
            if (pVal < self.pCut) and isSig:
                if logratio > 0:
                    co.append(red)
                elif logratio < 0:
                    co.append(green)
                else:
                    raise Exception
            else:
                co.append(blue)

        if ax == None:
            ax = plt.gca()

        ax.set_aspect('equal')
        minVal=np.min(np.c_[self.sample1, self.sample2])
        ax.scatter(self.sample1, self.sample2, c=co, alpha=0.7, edgecolor='none')
        ax.set_xlabel("%s %s" % (self.sampleNames[0], self.dtype))
        ax.set_ylabel("%s %s" % (self.sampleNames[1], self.dtype))
        ax.set_yscale('log', basey=2)
        ax.set_xscale('log', basex=2)
        ax.set_xlim(xmin=max(minVal, 0.1))
        ax.set_ylim(ymin=max(minVal, 0.1))
        if ax == None:
            plt.tight_layout()