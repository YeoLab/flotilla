import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from collections import defaultdict
import networkx as nx
from ..frigate import get_weight_fun
from ...project.project_params import min_cells, _default_group_id, _default_group_ids, _default_list_id, _default_list_ids

class Data(object):
    """Generic data model for both splicing and expression data

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, data, n_components, step=0.1, reducer=PCA):
        """Constructor for Data

        Specific implementation in the SplicingData and ExpressionData classes
        """
        raise NotImplementedError

    def calculate_distances(self, metric='euclidean'):
        """Creates a squareform distance matrix for clustering fun

        Needed for some clustering algorithms

        Parameters
        ----------
        metric : str
            One of any valid scipy.distance metric strings
        """
        raise NotImplementedError
        self.pdist = squareform(pdist(self.binned, metric=metric))
        return self

    def correlate(self, method='spearman', between='measurements'):
        """Find correlations between either splicing/expression measurements
        or cells
        """
        raise NotImplementedError

    def jsd(self):
        """Jensen-Shannon divergence showing most varying measurements within a
        celltype and between celltypes

        Returns
        -------
        fig : matplotlib.Figure
            A figure object for saving.
        """
        raise NotImplementedError

    def _echo(self, x):
        return x

    _naming_fun = _echo

    def get_naming_fun(self):
        return self._naming_fun

    def set_naming_fun(self, fun):
        self._naming_fun = fun
        try:
            fun('foo')
        except:
            raise TypeError("not a naming function")


    _default_reducer_args = {'whiten':False, 'show_point_labels':False, }
    _default_list = _default_list_id
    _default_featurewise=False
    samplewise_reduction = {}
    featurewise_reduction = {}
    pca_plotting_args = {}


    #def interactive_dim_reduction_plot(self):
    #    from IPython.html.widgets import interactive
    #    interactive(self.auto_dim_reduction_plot, x_pc=(1,10), y_pc=(1,10), featurewise=False,
    #                group_id=_default_group_ids, list_name=self.lists.keys())

#    def plot_last_dim_reduction_plot(self, x_pc=1, y_pc=2):
#        """plot_dimensionality_reduction with some params hidden"""
#        self.plot_dimensionality_reduction(x_pc, y_pc)

    def plot_dimensionality_reduction(self, x_pc=1, y_pc=2, obj_id=None, group_id=None,
                                      list_name=None, featurewise=None, **plotting_args):

        """Principal component-like analysis of measurements

        Returns
        -------
        self
        """
        local_plotting_args = self.pca_plotting_args.copy()
        local_plotting_args.update(plotting_args)
        pca = self.get_reduced(obj_id, list_name, group_id, featurewise=featurewise)
        pca(markers_size_dict=lambda x: 400,
            show_vectors=False,
            title_size=10,
            axis_label_size=10,
            x_pc = "pc_" + str(x_pc),#this only affects the plot, not the data.
            y_pc = "pc_" + str(y_pc),#this only affects the plot, not the data.
            **local_plotting_args
            )
        return self

    _last_reducer_accessed = None
    def get_reduced(self, obj_id=None, list_name=None, group_id=None, featurewise=None, **reducer_args):
        _used_default_group = False
        if group_id is None:
            group_id = _default_group_id
            _used_default_group = True

        _used_default_list = False
        if list_name is None:
            list_name = self._default_list
            _used_default_list = True

        _used_default_featurewise = False
        if featurewise is None:
            featurewise = self._default_featurewise
            _used_default_featurewise = True

        if obj_id is None:
            if self._last_reducer_accessed is None or \
                    (not _used_default_list or not _used_default_group or not _used_default_featurewise):
                #if last_reducer_accessed hasn't been set or if the user asks for specific params,
                #else return the last reducer gotten by this method

                obj_id = list_name + ":" + group_id + ":" + str(featurewise)

            else:
                obj_id = self._last_reducer_accessed

        self._last_reducer_accessed = obj_id
        if featurewise:
            rdc_dict = self.featurewise_reduction
        else:
            rdc_dict = self.samplewise_reduction
        try:
            return rdc_dict[obj_id]
        except:
            rdc_obj = self.make_reduced(list_name, group_id, featurewise=featurewise, **reducer_args)
            rdc_obj.obj_id = obj_id
            rdc_dict[obj_id] = rdc_obj

        return rdc_dict[obj_id]
