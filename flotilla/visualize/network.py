__author__ = 'olga'

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn


from ..compute.network import Networker
from ..util import dict_to_str
from ..visualize.decomposition import DecompositionViz


class NetworkerViz(Networker, DecompositionViz):
    #TODO.md: needs to be decontaminated, as it requires methods from data_object;
    #maybe this class should move to data_model.BaseData
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
        gene_of_interest - map a gradient representing this gene's data onto nodes
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

        f = plt.figure(figsize=(10,10))

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
        gene_of_interest - map a gradient representing this gene's data onto nodes
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