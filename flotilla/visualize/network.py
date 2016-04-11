"""
Visualize results from :py:mod:flotilla.compute.network
"""

import sys

import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from ..compute.network import Networker
from ..util import dict_to_str
from .color import dark2, almost_black, green, red


class NetworkerViz(Networker):
    # TODO: needs to be decontaminated, as it requires methods from
    # data_object;
    # maybe this class should move to data_model.BaseData
    def __init__(self, DataModel):
        self.DataModel = DataModel
        Networker.__init__(self)

    def draw_graph(self,
                   n_pcs=5,
                   use_pc_1=True, use_pc_2=True, use_pc_3=True, use_pc_4=True,
                   degree_cut=2, cov_std_cut=1.8,
                   weight_function='no_weight',
                   featurewise=False,  # else feature_components
                   rpkms_not_events=False,  # else event features
                   feature_of_interest='RBFOX2', draw_labels=True,
                   reduction_name=None,
                   feature_ids=None,
                   sample_ids=None,
                   graph_file='',
                   compare="",
                   sample_id_to_color=None,
                   label_to_color=None,
                   label_to_marker=None, groupby=None,
                   data_type=None):

        """Draw the graph of similarities between samples or features

        Parameters
        ----------
        feature_ids : list of str, or None
            Feature ids to subset the data. If None, all features will be used.
        sample_ids : list of str, or None
            Sample ids to subset the data. If None, all features will be used.
        x_pc : str, optional
            Which component to use for the x-axis, default "pc_1"
        y_pc :
            y component for PCA, default "pc_2"
        n_pcs : int
            Number of components to use for cells' covariance calculation
        cov_std_cut : float
            Covariance cutoff for edges
        use_pc{1-4} : bool
            Use these pcs in cov calculation (default True)
        degree_cut : int
            miniumum degree for a node to be included in graph display
        weight_function : ['arctan' | 'sq' | 'abs' | 'arctan_sq']
            weight function (arctan (arctan cov), sq (sq cov), abs (abs cov),
            arctan_sq (sqared arctan of cov))
        gene_of_interest : str
            map a gradient representing this gene's data onto nodes (ENSEMBL
            id or gene symbol)

        Returns
        -------
        graph : networkx.Graph

        positions : (x,y) positions of nodes
        """
        node_color_mapper = self._default_node_color_mapper
        node_size_mapper = self._default_node_color_mapper
        settings = locals().copy()
        # not pertinent to the graph, these are what we want to be able to
        # re-apply to the same graph if it exists
        pca_settings = dict()
        pca_settings['sample_ids'] = sample_ids
        pca_settings['featurewise'] = featurewise
        pca_settings['feature_ids'] = feature_ids
        # pca_settings['obj_id'] = reduction_name

        adjacency_settings = dict((k, settings[k]) for k in
                                  ['use_pc_1', 'use_pc_2', 'use_pc_3',
                                   'use_pc_4', 'n_pcs', ])

        plt.figure(figsize=(10, 10))

        plt.axis((-0.2, 1.2, -0.2, 1.2))
        main_ax = plt.gca()
        ax_pev = plt.axes([0.1, .8, .2, .15])
        ax_cov = plt.axes([0.1, 0.1, .2, .15])
        ax_degree = plt.axes([0.9, .8, .2, .15])

        pca = self.DataModel.reduce(
            # label_to_color=label_to_color,
            # label_to_marker=label_to_marker,
            # groupby=groupby,
            **pca_settings)

        try:
            feature_id = self.DataModel.maybe_renamed_to_feature_id(
                feature_of_interest)[0]
        except (ValueError, KeyError, IndexError):
            feature_id = ''

        if featurewise:
            def node_color_mapper(x):
                if (x == feature_id):
                    return green
                else:
                    return almost_black

            def node_size_mapper(x):
                return (pca.means.ix[x] ** 2) + 10

        else:
            if sample_id_to_color is not None:
                def node_color_mapper(x):
                    return sample_id_to_color[x]

            else:
                def node_color_mapper(x):
                    return dark2[0]

            def node_size_mapper(x):
                return 95

        percent_explained_variance = pca.explained_variance_ratio_ * 100.
        ax_pev.plot(percent_explained_variance.values)
        ax_pev.axvline(n_pcs, label='cutoff', color=green)
        ax_pev.legend()
        ax_pev.set_ylabel("% explained variance")
        ax_pev.set_xlabel("component")
        ax_pev.set_title("Explained variance from dim reduction")
        sns.despine(ax=ax_pev)

        adjacency = self.adjacency(pca.reduced_space, **adjacency_settings)
        cov_dist = np.array(
            [i for i in adjacency.values.ravel() if np.abs(i) > 0])
        cov_cut = np.mean(cov_dist) + cov_std_cut * np.std(cov_dist)

        graph_settings = dict(
            (k, settings[k]) for k in ['weight_function', 'degree_cut', ])
        graph_settings['cov_cut'] = cov_cut
        this_graph_name = "_".join(map(dict_to_str,
                                       [pca_settings, adjacency_settings,
                                        graph_settings]))
        graph_settings['name'] = this_graph_name

        sns.kdeplot(cov_dist, ax=ax_cov)
        xmin, xmax = ax_cov.get_xlim()
        ax_cov.set_xlim(0, xmax)
        ax_cov.axvline(cov_cut, label='cutoff', color=green)
        ax_cov.set_title("Covariance in dim reduction space")
        ax_cov.set_ylabel("Density")
        ax_cov.legend()
        sns.despine(ax=ax_cov)

        graph, pos = self.graph(adjacency, **graph_settings)

        nx.draw_networkx_nodes(
            graph, pos,
            node_color=map(node_color_mapper, graph.nodes()),
            node_size=map(node_size_mapper, graph.nodes()),
            ax=main_ax, alpha=0.5)

        try:
            node_color = map(lambda x: pca.X[feature_id].ix[x], graph.nodes())

            nx.draw_networkx_nodes(graph, pos, node_color=node_color,
                                   cmap=mpl.cm.Greys,
                                   node_size=map(
                                       lambda x: node_size_mapper(x) * .5,
                                       graph.nodes()), ax=main_ax, alpha=1)
        except (KeyError, ValueError):
            pass

        if featurewise:
            namer = self.DataModel.feature_renamer
        else:
            def namer(x):
                return x

        labels = dict([(name, namer(name)) for name in graph.nodes()])
        if draw_labels:
            nx.draw_networkx_labels(graph, pos, labels=labels, ax=main_ax)
        nx.draw_networkx_edges(graph, pos, ax=main_ax, alpha=0.1)
        main_ax.set_axis_off()
        degree = nx.degree(graph)
        sns.kdeplot(np.array(degree.values()), ax=ax_degree)
        xmin, xmax = ax_degree.get_xlim()
        ax_degree.set_xlim(0, xmax)
        ax_degree.set_xlabel("degree")
        ax_degree.set_ylabel("density")
        try:
            ax_degree.axvline(x=degree[feature_id],
                              label=feature_of_interest,
                              color=green)
            ax_degree.legend()

        except Exception as e:
            sys.stdout.write(str(e))
            pass

        sns.despine(ax=ax_degree)
        if graph_file != '':
            try:
                nx.write_gml(graph, graph_file)
            except Exception as e:
                sys.stdout.write("error writing graph file:"
                                 "\n{}".format(str(e)))

        return graph, pos

    def draw_nonreduced_graph(self,
                              degree_cut=2, cov_std_cut=1.8,
                              wt_fun='abs',
                              featurewise=False,  # else feature_components
                              rpkms_not_events=False,  # else event features
                              feature_of_interest='RBFOX2', draw_labels=True,
                              feature_ids=None,
                              group_id=None,
                              graph_file='',
                              compare=""):

        """
        Parameters
        ----------
        feature_ids : list of str, or None
            Feature ids to subset the data. If None, all features will be used.
        sample_ids : list of str, or None
            Sample ids to subset the data. If None, all features will be used.
        x_pc : str
            x component for DataFramePCA, default "pc_1"
        y_pc :
            y component for DataFramePCA, default "pc_2"
        n_pcs : int???
            n components to use for cells' covariance calculation
        cov_std_cut : float??
            covariance cutoff for edges
        use_pc{1-4} use these pcs in cov calculation (default True)
        degree_cut : int??
            miniumum degree for a node to be included in graph display
        weight_function : ['arctan' | 'sq' | 'abs' | 'arctan_sq']
            weight function (arctan (arctan cov), sq (sq cov), abs (abs cov),
            arctan_sq (sqared arctan of cov))
        gene_of_interest : str
            map a gradient representing this gene's data onto nodes (ENSEMBL
            id or gene name???)


        Returns
        -------
        #TODO: Mike please fill these in
        graph : networkx.Graph
            ???
        positions : ???
            ???
        """
        node_color_mapper = self._default_node_color_mapper
        node_size_mapper = self._default_node_color_mapper
        settings = locals().copy()

        adjacency_settings = dict(('non_reduced', True))

        plt.figure(figsize=(10, 10))
        plt.axis((-0.2, 1.2, -0.2, 1.2))
        main_ax = plt.gca()
        ax_cov = plt.axes([0.1, 0.1, .2, .15])
        ax_degree = plt.axes([0.9, .8, .2, .15])

        data = self.DataModel.df

        try:
            feature_id = self.DataModel.maybe_renamed_to_feature_id(
                feature_of_interest)[0]
        except (ValueError, KeyError):
            feature_id = ''

        if featurewise:
            def node_color_mapper(x):
                if x == feature_id:
                    return red
                else:
                    return 'k'

            def node_size_mapper(x):
                return (data.mean().ix[x] ** 2) + 10

        else:
            def node_color_mapper(x):
                return self.DataModel.sample_metadata.color[x]

            def node_size_mapper(x):
                return 75

        adjacency_name = "_".join([dict_to_str(adjacency_settings)])
        adjacency = self.adjacency(data, name=adjacency_name,
                                   **adjacency_settings)
        cov_dist = np.array(
            [i for i in adjacency.values.ravel() if np.abs(i) > 0])
        cov_cut = np.mean(cov_dist) + cov_std_cut * np.std(cov_dist)

        graph_settings = dict(
            (k, settings[k]) for k in ['wt_fun', 'degree_cut', ])
        graph_settings['cov_cut'] = cov_cut
        this_graph_name = "_".join(
            map(dict_to_str, [adjacency_settings, graph_settings]))
        graph_settings['name'] = this_graph_name

        sns.kdeplot(cov_dist, ax=ax_cov)
        ax_cov.axvline(cov_cut, label='cutoff')
        ax_cov.set_title("covariance in original space")
        ax_cov.set_ylabel("density")
        ax_cov.legend()
        sns.despine(ax=ax_cov)
        graph, positions = self.graph(adjacency, **graph_settings)

        nx.draw_networkx_nodes(graph, positions,
                               node_color=map(node_color_mapper,
                                              graph.nodes()),
                               node_size=map(node_size_mapper, graph.nodes()),
                               ax=main_ax, alpha=0.5)
        try:
            node_color = map(lambda x: data[feature_id].ix[x],
                             graph.nodes())
            nx.draw_networkx_nodes(graph, positions, node_color=node_color,
                                   cmap=plt.cm.Greys,
                                   node_size=map(
                                       lambda x: node_size_mapper(x) * .5,
                                       graph.nodes()), ax=main_ax, alpha=1)
        except (KeyError, ValueError):
            pass

        def renamer(x):
            return x

        labels = dict([(name, renamer(name)) for name in graph.nodes()])
        if draw_labels:
            nx.draw_networkx_labels(graph, positions, labels=labels,
                                    ax=main_ax)
        nx.draw_networkx_edges(graph, positions, ax=main_ax, alpha=0.1)
        main_ax.set_axis_off()
        degree = nx.degree(graph)
        sns.kdeplot(np.array(degree.values()), ax=ax_degree)
        ax_degree.set_xlabel("degree")
        ax_degree.set_ylabel("density")
        try:
            ax_degree.axvline(x=degree[feature_of_interest],
                              label=feature_of_interest)
            ax_degree.legend()

        except Exception as e:
            sys.stdout.write(str(e))
            pass

        sns.despine(ax=ax_degree)
        if graph_file != '':
            try:
                nx.write_gml(graph, graph_file)
            except Exception as e:
                sys.stdout.write("error writing graph file:"
                                 "\n{}".format(str(e)))

        return (graph, positions)
