"""
Compute networks (the kind with nodes and edges) on data. Visualize with
:py:mod:flotilla.visualize.network
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import networkx as nx
import numpy as np
import pandas as pd

from ..util import memoize
from ..visualize.color import dark2


class Networker(object):
    """Networks (the kind with nodes and edges), aka a graph

    Calculate the edges based on similarity between rows of PCA-reduced data
    """
    weight_funs = ['no_weight', 'sq', 'arctan', 'arctan_sq']

    def __init__(self):
        """Construct a Networker object with default node colors (dark teal)
        and sizes (all nodes at 300)
        """
        self._default_node_color_mapper = lambda x: dark2[0]
        self._default_node_size_mapper = lambda x: 300

    def get_weight_fun(self, fun_name='no_weight'):
        """Given a string, return the function

        Used to obtain functions that perform common transforms on distance

        Parameters
        ----------
        fun_name : 'no_weight' | 'sq' | 'arctan' | 'arctan_sq', optional
            Name of the function to obtain (default 'no_weight')

        Returns
        -------
        func : function
            A function which transforms a number in the indicated way

        Raises
        ------
        ValueError
            If `fun_name` is not one of the ones indicated above
        """
        def _noweight(x):
            return x

        def _arctan_sq(x):
            return np.arctan(x) ** 2

        if fun_name == 'no_weight':
            wt = _noweight
        elif fun_name == 'sq':
            wt = np.square
        elif fun_name == 'arctan':
            wt = np.arctan
        elif fun_name == 'arctan_sq':
            wt = _arctan_sq
        else:
            raise ValueError
        return wt

    @memoize
    def adjacency(self, data, use_pc_1=True, use_pc_2=True,
                  use_pc_3=True, use_pc_4=True, n_pcs=5):
        """Calculate the adjacency graph, i.e. connectedness between nodes

        Parameters
        ----------
        data : pandas.DataFrame
            A (n_nodes, n_pcs) sized dataframe of reduced data
        use_pc1 : bool, optional
            If True, use the first principal component of reduced data
            (default True)
        use_pc2 : bool, optional
            If True, use the second principal component of reduced data
            (default True)
        use_pc3 : bool, optional
            If True, use the third principal component of reduced data
            (default True)
        use_pc4 : bool, optional
            If True, use the fourth principal component of reduced data
            (default True)
        n_pcs : int, optional
            Total number of principal components to use (default 5)

        Returns
        -------
        adjacency : pandas.DataFrame
            A lower triangular matrix of the edge weights between the rows of
            the data
        """
        total_pcs = data.shape[1]
        use_cols = np.ones(total_pcs, dtype='bool')
        use_cols[n_pcs:] = False
        use_cols = use_cols * np.array(
            [use_pc_1, use_pc_2, use_pc_3, use_pc_4] + [True, ] * (
                total_pcs - 4))
        subset = data.loc[:, use_cols]
        cov = np.cov(subset)
        nrow, ncol = subset.shape
        return pd.DataFrame(np.tril(cov * - (np.identity(nrow) - 1)),
                            index=subset.index, columns=data.index)

    @memoize
    def graph(self, adjacency, cov_cut=0,
              node_color_mapper=None,
              node_size_mapper=None,
              degree_cut=2,
              weight_function='no_weight', name=None):
        """Create a graph based on the adjacency matrix and other inputs

        Parameters
        ----------
        adjacency : pandas.DataFrame
            A (n_nodes, n_nodes) square dataframe of edge weights between all
            nodes in the graph
        cov_cut : float, optional
            Minimum covariance between two nodes for their edge to be plotted.
            (default 0)
        node_color_mapper : function, optional
            Function to recolor the nodes for plotting, based on the node name.
            If None, defaults to a dark teal. (default None)
        node_size_mapper : function, optional
            Function to resize the nodes for plotting, based on the node name.
            If None, defaults to the same size for all nodes. (default None)
        degree_cut : int
            Minimum number of edges a node must have for it to be drawn on the
            graph
        weight_function : 'no_weight' | 'sq' | 'arctan' | 'arctan_sq', optional
            Weight function of the edges. The lower the weight, the farther
            away two nodes are drawn from each other.
        name : str, optional (default=None)
            For memoization purposes, not used in the function.

        Returns
        -------
        graph : networkx.Graph
            The graph created with all these parameters
        positions : dict
            A {node_name : [x, y]} mapping of all nodes and their x, y
            positions
        """
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
        for cell1, others in adjacency.iterrows():
            for cell2, value in others.iteritems():
                if value > cov_cut:
                    # cast to floats because write_gml doesn't like numpy
                    # dtypes
                    graph.add_edge(cell1, cell2, weight=float(weight(value)),
                                   inv_weight=float(1 / weight(value)),
                                   alpha=0.05)

        graph.remove_nodes_from(
            [k for k, v in graph.degree().iteritems() if v <= degree_cut])

        # TODO: can we output this as a (nodes, (x, y)) DataFrame instead?
        positions = nx.spring_layout(graph)

        return graph, positions
