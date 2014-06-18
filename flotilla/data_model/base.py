from collections import defaultdict
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import six
import sys

from ..visualize.color import red, blue, green
from ..util import memoize

MINIMUM_SAMPLES = 12


class BaseData(object):
    """Generic study_data model for both splicing and expression study_data

    Attributes
    ----------


    Methods
    -------

    """

    #_feature_rename converts input feature names to something else. by default, just echo.
    _feature_rename = lambda x: x
    _default_reducer_kwargs = {'whiten' : False,
                               'show_point_labels': False,
                               'show_vectors': False}
    _default_plot_kwargs = {'marker': 'o', 'color': blue}

    feature_sets = {}

    def __init__(self, phenotype_data, data, feature_data=None, species=None):
        """Base class for biological data measurements

        Parameters
        ----------
        phenotype_data : pandas.DataFrame
            Metadata on the samples, with sample names as rows and columns as
            attributes. Any boolean column will be added as an option to
            interactive_pca
        data : pandas.DataFrame
            A dataframe of samples x features (samples on rows, features on
            columns) with some kind of measurements of cells,
            e.g. gene expression values such as TPM, RPKM or FPKM, alternative
            splicing "Percent-spliced-in" (PSI) values, or RNA editing scores.
        species : str, optional
            The species in which this was measured

        """
        self.data = data
        self.phenotype_data = phenotype_data
        self.feature_data = feature_data
        self.species = species
        self._set_plot_colors()
        self._set_plot_markers()

    def _set_plot_colors(self):
        """If there is a column 'color' in the sample metadata, specify this
        as the plotting color
        """
        try:
            self._default_reducer_kwargs.update(
                {'colors_dict': self.phenotype_data.color})
            self._default_plot_kwargs.update(
                {'color': self.phenotype_data.color.tolist()})
        except AttributeError:
            sys.stderr.write("There is no column named 'color' in the "
                             "metadata, defaulting to blue for all samples")
            self._default_reducer_kwargs.update(
                {'colors_dict': defaultdict(lambda : blue)})

    def _set_plot_markers(self):
        """If there is a column 'marker' in the sample metadata, specify this
        as the plotting marker (aka the plotting shape). Only valid matplotlib
        symbols are allowed. See http://matplotlib.org/api/markers_api.html
        for a more complete description.
        """
        try:
            self._default_reducer_kwargs.update(
                {'markers_dict': self.phenotype_data.marker})
            self._default_plot_kwargs.update(
                {'marker': self.phenotype_data.marker.tolist()})
        except AttributeError:
            sys.stderr.write("There is no column named 'marker' in the sample "
                             "metadata, defaulting to a circle for all samples")
            self._default_reducer_kwargs.update({'markers_dict':
                                                   defaultdict(lambda : 'o')})

    @property
    def outliers(self):
        """If there is a column called 'outliers' in the phenotype_data,
        then return the samples where this is True for them
        """
        try:
            return set(self.phenotype_data.ix[
                           self.phenotype_data.outlier.map(bool),
                           'outlier'].index)
        except AttributeError:
            return set([])

    def drop_outliers(self, df):
        # assert 'outlier' in self.phenotype_data.columns
        outliers = self.outliers.intersection(df.index)
        print "dropping ", outliers
        return df.drop(outliers)


    def calculate_distances(self, metric='euclidean'):
        """Creates a squareform distance matrix for clustering fun

        Needed for some clustering algorithms

        Parameters
        ----------
        metric : str, optional
            One of any valid scipy.distance metric strings. Default 'euclidean'
        """
        raise NotImplementedError
        self.pdist = squareform(pdist(self.binned, metric=metric))
        return self

    def correlate(self, method='spearman', between='features'):
        """Find correlations between either splicing/expression measurements
        or cells

        Parameters
        ----------
        method : str
            Specify to calculate either 'spearman' (rank-based) or 'pearson'
            (linear) correlation. Default 'spearman'
        between : str
            Either 'features' or 'samples'. Default 'features'
        """
        raise NotImplementedError
        # Stub for choosing between features or samples
        if 'features'.startswith(between):
            pass
        elif 'samples'.startswith(between):
            pass

    def jsd(self):
        """Jensen-Shannon divergence showing most varying measurements within a
        celltype and between celltypes
        """
        raise NotImplementedError


    def get_feature_renamer(self):
        return self._feature_rename

    def _set_naming_fun(self, fun, test_name='foo'):
        self._feature_rename = fun
        try:
            fun(test_name)
        except:
            pass
            #print "might not be a good naming function, failed on %s" % test_name


    # TODO: Specify dtypes in docstring
    def plot_classifier(self, gene_list_name=None, sample_list_name=None, clf_var=None,
                        predictor_args=None, plotting_args=None):
        """Principal component-like analysis of measurements

        Params
        -------
        obj_id : str
            key of the object getting plotted
        group_id : str
            ???
        categorical_trait : str
            classifier feature
        list_name : str
            subset of genes to use for building class



        Returns
        -------
        self
        """
        if predictor_args is None:
            predictor_args = {}

        if plotting_args is None:
            plotting_args = {}

        local_plotting_args = self.pca_plotting_args.copy()
        local_plotting_args.update(plotting_args)

        clf = self.classify(gene_list_name=gene_list_name,
                                 sample_list_name=sample_list_name,
                                 clf_var=clf_var,
                                 **predictor_args)
        clf(plotting_args=local_plotting_args)

        return self

    def plot_dimensionality_reduction(self, x_pc=1, y_pc=2, obj_id=None,
                                      group_id=None,
                                      list_name=None, featurewise=None,
                                      **plotting_kwargs):
        """Principal component-like analysis of measurements

        Parameters
        ----------
        x_pc : int
            Which principal component to plot on the x-axis
        y_pc : int
            Which principal component to plot on the y-axis
        obj_id : str
            Key of the object getting plotted
        group_id : str
            ???


        Returns
        -------


        Raises
        ------

        Returns
        -------
        self
        """
        local_plotting_kwargs = self.pca_plotting_kwargs.copy()
        local_plotting_kwargs.update(plotting_kwargs)
        pca = self.reduce(obj_id, list_name, group_id,
                           featurewise=featurewise)
        pca(markers_size_dict=lambda x: 400,
            show_vectors=False,
            title_size=10,
            axis_label_size=10,
            x_pc = "pc_" + str(x_pc), #this only affects the plot, not the study_data.
            y_pc = "pc_" + str(y_pc), #this only affects the plot, not the study_data.
            **local_plotting_kwargs
            )
        return self

    @memoize
    def reduce(self, *args, **kwargs):
        """Reduce the dimensionality of the data. Must be implemented for
        each specific data sub-type
        """
        raise NotImplementedError

    @memoize
    def classify(self, *args, **kwargs):
        """Run a classifier on the data. Must be implemented for each
        specific data sub-type
        """
        raise NotImplementedError

    # @memoize
    # def get_reduced(self, obj_id=None, list_name=None, group_id=None, featurewise=None, **reducer_args):
    #     _used_default_group = False
    #     if group_id is None:
    #         group_id = self._default_group_id
    #         _used_default_group = True
    #
    #     _used_default_list = False
    #     if list_name is None:
    #         list_name = self._default_list_id
    #         _used_default_list = True
    #
    #     _used_default_featurewise = False
    #     if featurewise is None:
    #         featurewise = self._default_featurewise
    #         _used_default_featurewise = True
    #
    #     if obj_id is None:
    #         if self._last_reducer_accessed is None or \
    #                 (not _used_default_list or not _used_default_group or not _used_default_featurewise):
    #             #if last_reducer_accessed hasn't been set or if the user asks for specific params,
    #             #else return the last reducer gotten by this method
    #
    #             obj_id = list_name + ":" + group_id + ":" + str(featurewise)
    #
    #         else:
    #             obj_id = self._last_reducer_accessed
    #
    #     self._last_reducer_accessed = obj_id
    #     if featurewise:
    #         rdc_dict = self.featurewise_reduction
    #     else:
    #         rdc_dict = self.samplewise_reduction
    #     try:
    #         return rdc_dict[obj_id]
    #     except:
    #         rdc_obj = self.reduced(list_name, group_id, featurewise=featurewise, **reducer_args)
    #         rdc_obj.obj_id = obj_id
    #         rdc_dict[obj_id] = rdc_obj
    #
    #     return rdc_dict[obj_id]

    # def get_classifier(self, gene_list_name=None, sample_list_name=None, clf_var=None,
    #                   obj_id=None,
    #                   **classifier_args):
    #     """
    #     list_name = list of features to use for this clf
    #     obj_id = name of this classifier
    #     clf_var = boolean or categorical pd.Series
    #     """
    #
    #     _used_default_group = False
    #     if sample_list_name is None:
    #         sample_list_name = self._default_group_id
    #         _used_default_group = True
    #
    #     _used_default_list = False
    #     if gene_list_name is None:
    #         gene_list_name = self._default_list_id
    #         _used_default_list = True
    #
    #     if obj_id is None:
    #         if self._last_predictor_accessed is None or \
    #                 (not _used_default_list or not _used_default_group):
    #             #if last_reducer_accessed hasn't been set or if the user asks for specific params,
    #             #else return the last reducer gotten by this method
    #
    #             obj_id = gene_list_name + ":" + sample_list_name + ":" + clf_var
    #         else:
    #             obj_id = self._last_predictor_accessed
    #
    #     self._last_predictor_accessed = obj_id
    #     #print "I am a %s" % type(self)
    #     #print "here are my clf_dict keys: %s" % " ".join(self.clf_dict.keys())
    #     try:
    #         return self.clf_dict[obj_id]
    #     except:
    #         clf = self.classify(gene_list_name, sample_list_name, clf_var, **classifier_args)
    #         clf.obj_id = obj_id
    #         self.clf_dict[obj_id] = clf
    #
    #     return self.clf_dict[obj_id]

    def get_min_samples(self):
        try:
            return self.min_samples
        except AttributeError:
            return MINIMUM_SAMPLES

    def set_min_samples(self, min_samples):
        self.min_samples = min_samples

