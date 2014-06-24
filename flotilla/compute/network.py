import networkx as nx

import numpy as np
import pandas as pd

from ..util import memoize

from ..visualize.color import red


class Networker(object):
    weight_funs = ['abs', 'sq', 'arctan', 'arctan_sq']

    def __init__(self):
        # self.adjacencies_ = defaultdict()
        # self.graphs_ = defaultdict()
        self._default_node_color_mapper = lambda x: red
        self._default_node_size_mapper = lambda x: 300
        # self._last_adjacency_accessed = None
        # self._last_graph_accessed = None

    def get_weight_fun(self, fun_name):
        _abs = lambda x: x
        _sq = lambda x: x ** 2
        _arctan = lambda x: np.arctan(x)
        _arctan_sq = lambda x: np.arctan(x) ** 2
        if fun_name == 'abs':
            wt = _abs
        elif fun_name == 'sq':
            wt = _sq
        elif fun_name == 'arctan':
            wt = _arctan
        elif fun_name == 'arctan_sq':
            wt = _arctan_sq
        else:
            raise ValueError
        return wt

    @memoize
    def adjacency(self, data, use_pc_1=True, use_pc_2=True,
                  use_pc_3=True, use_pc_4=True, n_pcs=5):
        total_pcs = data.shape[1]
        use_cols = np.ones(total_pcs, dtype='bool')
        use_cols[n_pcs:] = False
        use_cols = use_cols * np.array(
            [use_pc_1, use_pc_2, use_pc_3, use_pc_4] + [True, ] * (
                total_pcs - 4))
        selected_cols = data.loc[:, use_cols]
        cov = np.cov(selected_cols)
        nrow, ncol = selected_cols.shape
        return pd.DataFrame(np.tril(cov * - (np.identity(nrow) - 1)),
                            index=selected_cols.index, columns=data.index)

    @memoize
    def graph(self, adjacency, cov_cut=None, name=None,
              node_color_mapper=None,
              node_size_mapper=None,
              degree_cut=2,
              weight_function='abs'):

        if node_color_mapper is None:
            node_color_mapper = self._default_node_color_mapper
        if node_size_mapper is None:
            node_size_mapper = self._default_node_size_mapper

        weight = self.get_weight_fun(weight_function)
        graph = nx.Graph()
        for node_label in adjacency.index:
            node_color = node_color_mapper(node_label)
            node_size = node_size_mapper(node_label)
            graph.add_node(node_label, node_size=node_size,
                           node_color=node_color)
            #    g.add_nodes_from(adjacency.index) #to add without setting attributes...neater, but does same thing as above loop
        for cell1, others in adjacency.iterrows():
            for cell2, value in others.iteritems():
                if value > cov_cut:
                    #cast to floats because write_gml doesn't like numpy dtypes
                    graph.add_edge(cell1, cell2, weight=float(weight(value)),
                                   inv_weight=float(1 / weight(value)),
                                   alpha=0.05)

        graph.remove_nodes_from(
            [k for k, v in graph.degree().iteritems() if v <= degree_cut])

        positions = nx.spring_layout(graph)

        return graph, positions