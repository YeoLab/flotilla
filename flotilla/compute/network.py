from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from ..util import memoize


class Networker(object):
    weight_funs=['abs', 'sq', 'arctan', 'arctan_sq']

    def get_weight_fun(self, fun_name):
        _abs =  lambda x: x
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

    def __init__(self):
        self.adjacencies_ = defaultdict()
        self.graphs_ = defaultdict()
        self._default_node_color_mapper = lambda x: 'r'
        self._default_node_size_mapper = lambda x: 300
        self._last_adjacency_accessed = None
        self._last_graph_accessed = None

    @memoize
    def adjacency(self, data=None, name=None, use_pc_1=True, use_pc_2=True,
                  use_pc_3=True, use_pc_4=True, n_pcs=5):

        if data is None and self._last_adjacency_accessed is None:
            raise AttributeError("this hasn't been called yet")

            # if name is None:
            #     if self._last_adjacency_accessed is None:
            #         name = 'default'
            #     else:
            #         name = self._last_adjacency_accessed
            # self._last_adjacency_accessed = name
            # try:
            #     if name in self.adjacencies_:
            #         #print "returning a pre-built adjacency"
            #         return self.adjacencies_[name]
            #     else:
            #         raise ValueError("adjacency hasn't been built yet")
            # except ValueError:
            #print 'reduced space', data.shape
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
    def graph(self, adjacency=None, cov_cut=None, name=None,
              node_color_mapper=None,
                  node_size_mapper=None,
                  degree_cut = 2,
                  weight_function='abs'):

        if node_color_mapper is None:
            node_color_mapper = self._default_node_color_mapper
        if node_size_mapper is None:
            node_size_mapper = self._default_node_size_mapper

        if name is None:
            if self._last_graph_accessed is None:
                name = 'default'
            else:
                name = self._last_graph_accessed
        self._last_graph_accessed = name
        try:
            g,pos = self.graphs_[name]
        except:
            wt = self.get_weight_fun(weight_function)
            g = nx.Graph()
            for node_label in adjacency.index:

                node_color = node_color_mapper(node_label)
                node_size = node_size_mapper(node_label)
                g.add_node(node_label, node_size=node_size, node_color=node_color)
        #    g.add_nodes_from(adjacency.index) #to add without setting attributes...neater, but does same thing as above loop
            for cell1, others in adjacency.iterrows():
                for cell2, value in others.iteritems():
                    if value > cov_cut:
                        #cast to floats because write_gml doesn't like numpy dtypes
                        g.add_edge(cell1, cell2, weight=float(wt(value)),inv_weight=float(1/wt(value)), alpha=0.05)

            g.remove_nodes_from([k for k, v in g.degree().iteritems() if v <= degree_cut])

            pos = nx.spring_layout(g)
            self.graphs_[name] = (g, pos)

        return g, pos