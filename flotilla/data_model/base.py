"""
Base data class for all data types. All data types in flotilla inherit from
this, or a child object (like ExpressionData).
"""
import sys

import pandas as pd
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from ..compute.clustering import Cluster
from ..compute.infotheory import binify
from ..visualize.decomposition import PCAViz
from ..visualize.generic import violinplot
from ..external import link_to_list
from ..compute.predict import PredictorConfigManager, PredictorDataSetManager
from ..util import memoize

MINIMUM_SAMPLES = 10
default_predictor_name = "ExtraTreesClassifier"


class BaseData(object):
    """Generic study_data model for both splicing and expression study_data

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, data=None, metadata=None,
                 species=None, feature_rename_col=None, outliers=None,
                 min_samples=MINIMUM_SAMPLES, pooled=None,
                 predictor_config_manager=None):
        """Base class for biological data measurements

        Parameters
        ----------
        data : pandas.DataFrame
            A dataframe of samples x features (samples on rows, features on
            columns) with some kind of measurements of cells,
            e.g. gene expression values such as TPM, RPKM or FPKM, alternative
            splicing "Percent-spliced-in" (PSI) values, or RNA editing scores.
        """
        self.data = data
        if pooled is not None:
            self.pooled = self.data.ix[pooled]

        if outliers is not None:
            self.data, self.outliers = self.drop_outliers(self.data,
                                                          outliers)
        self.feature_data = metadata
        self.feature_rename_col = feature_rename_col
        self.min_samples = min_samples
        self.default_feature_sets = []
        self.data_type = None

        self.clusterer = Cluster()

        self.species = species

        def shortener(renamer, x):
            renamed = renamer(x)
            if isinstance(renamed, float):
                return renamed
            elif len(renamed) > 20:
                return '{}...'.format(renamed[:20])
            else:
                return renamed

        if self.feature_data is not None and self.feature_rename_col is not \
                None:
            def feature_renamer(x):
                if x in self.feature_data[feature_rename_col].index:
                    rename = self.feature_data[feature_rename_col][x]
                    if isinstance(rename, pd.Series):
                        return rename.values[0]
                    elif not isinstance(rename, float) and ':' in x:
                        # Check for NaN and ":" (then it's a splicing event
                        # name)
                        return ":".join(x.split("@")[1].split(":")[:2])
                    else:
                        return rename
                else:
                    return x

            self.feature_renamer = lambda x: shortener(feature_renamer, x)
        else:
            self.feature_renamer = lambda x: shortener(lambda y: y, x)

        self.renamed_to_feature_id = pd.Series(
            self.data.columns,
            index=self.data.columns.map(self.feature_renamer))
        self.renamed_to_feature_id = self.renamed_to_feature_id.dropna()
        self.renamed_to_feature_id = self.renamed_to_feature_id[~pd.isnull(
            self.renamed_to_feature_id.index)]

        if predictor_config_manager is None:
            self.predictor_config_manager = PredictorConfigManager()
        else:
            self.predictor_config_manager = predictor_config_manager

        self.predictor_dataset_manager = PredictorDataSetManager(
            self.predictor_config_manager)

    @property
    def _var_cut(self):
        return self.data.var().dropna().mean() + 2 * self.data.var() \
            .dropna().std()

    @property
    def variant(self):
        return pd.Index([i for i, j in (self.data.var().dropna()
                                        > self._var_cut).iteritems() if j])

    def drop_outliers(self, df, outliers):
        # assert 'outlier' in self.experiment_design_data.columns
        outliers = set(outliers).intersection(df.index)
        try:
            # Remove pooled samples, if there are any
            outliers = outliers.difference(self.pooled.index)
        except AttributeError:
            pass
        sys.stdout.write("dropping {}\n".format(outliers))
        data = df.drop(outliers)
        outlier_data = df.ix[outliers]
        return data, outlier_data

    @property
    def feature_subsets(self):
        feature_subsets = {}
        if self.feature_data is not None:
            for col in self.feature_data:
                if self.feature_data[col].dtype != bool:
                    continue
                feature_set = self.feature_data.index[self.feature_data[col]]
                if len(feature_set) > 1:
                    feature_subsets[col] = feature_set
            categories = ['tag', 'gene_type', 'splice_type', 'gene_status']
            for category in categories:
                if category in self.feature_data:
                    feature_subsets.update(
                        self.feature_data.groupby(category).groups)
        feature_subsets['all_genes'] = self.data.columns
        feature_subsets['variant'] = self.variant
        return feature_subsets

    def feature_subset_to_feature_ids(self, feature_subset, rename=True):
        if feature_subset is not None:
            if feature_subset in self.feature_subsets:
                feature_ids = self.feature_subsets[feature_subset]
            elif feature_subset == self.all_features:
                feature_ids = self.data.columns
            else:
                try:
                    feature_ids = link_to_list(feature_subset)
                    self.feature_subsets[feature_subset] = feature_ids
                except:

                    raise ValueError(
                        "There are no {} features in this data: "
                        "{}".format(feature_subset, self))
            if rename:
                feature_ids = feature_ids.map(self.feature_renamer)
        else:
            feature_ids = self.data.columns
        return feature_ids

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

    @property
    def feature_renamer(self):
        return self._feature_renamer

    @feature_renamer.setter
    def feature_renamer(self, renamer):
        self._feature_renamer = renamer

    def maybe_get_renamed(self, renamed):
        try:
            return self.renamed_to_feature_id[renamed]
        except KeyError:
            if renamed in self.data.columns:
                return renamed
            else:
                raise

    # TODO.md: Specify dtypes in docstring
    def plot_classifier(self, trait, sample_ids=None, feature_ids=None,
                        predictor_name=None,
                        standardize=True, score_coefficient=None,
                        data_name=None,
                        label_to_color=None,
                        label_to_marker=None,
                        groupby=None,
                        **plotting_kwargs):
        """Principal component-like analysis of measurements

        Params
        -------
        trait - a pandas series with categorical features, indexed like self.data
        sample_ids - an iterable of row IDs to use
        feature_ids - an iterable of column IDs to use
        standardize - 0-center, 1-variance
        score_coefficient - for calculating score cutoff, default == 2
        data_name - a name (str) for this subset of the data

        Returns
        -------
        self
        """
        # print trait
        plotting_kwargs = {} if plotting_kwargs is None else plotting_kwargs

        # local_plotting_args = self.pca_plotting_args.copy()
        # local_plotting_args.update(plotting_kwargs)
        if predictor_name is None:
            predictor_name = default_predictor_name

        clf = self.classify(trait, sample_ids=sample_ids,
                            feature_ids=feature_ids,
                            data_name=data_name,
                            standardize=standardize,
                            predictor_name=predictor_name,
                            feature_renamer=self.feature_renamer,
                            data_type=self.data_type)
        clf.score_coefficient = score_coefficient
        clf(**plotting_kwargs)
        return self

    def plot_dimensionality_reduction(self, x_pc=1, y_pc=2,
                                      sample_ids=None, feature_ids=None,
                                      featurewise=False, reducer=PCAViz,
                                      label_to_color=None,
                                      label_to_marker=None,
                                      groupby=None, order=None, color=None,
                                      reduce_kwargs=None,
                                      **plotting_kwargs):
        """Principal component-like analysis of measurements

        Parameters
        ----------
        x_pc : int
            Which principal component to plot on the x-axis
        y_pc : int
            Which principal component to plot on the y-axis
        sample_ids : None or list of strings
            If None, plot all the samples. If a list of strings, must be
            valid sample ids of the data
        feature_ids : None or list of strings
            If None, plot all the features. If a list of strings
        featurewise : bool
            Whether to keep the features and reduce on the samples (default
            is to keep the samples and reduce the features)
        reducer : flotilla.visualize.DecompositionViz
            Which decomposition object to use. Must be a flotilla object,
            as this has built-in compatibility with pandas.DataFrames.


        Returns
        -------
        self

        Raises
        ------

        """
        reduce_kwargs = {} if reduce_kwargs is None else reduce_kwargs

        pca = self.reduce(sample_ids, feature_ids,
                          featurewise=featurewise, reducer=reducer,
                          label_to_color=label_to_color,
                          label_to_marker=label_to_marker,
                          groupby=groupby, order=order, color=color,
                          **reduce_kwargs)
        pca(show_vectors=True,
            x_pc="pc_" + str(x_pc),
            y_pc="pc_" + str(y_pc),
            **plotting_kwargs)
        return self

    def plot_pca(self, **kwargs):
        self.plot_dimensionality_reduction(reducer=PCAViz, **kwargs)

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, values):
        self._min_samples = values

    def _subset(self, data, sample_ids=None, feature_ids=None,
                require_min_samples=True):
        """Take only a subset of the data, and require at least the minimum
        samples observed to be not NA for each feature.

        Parameters
        ----------
        data : pandas.DataFrame
            Data to subset
        sample_ids : list of str
            Which samples to use. If None, use all.
        feature_ids : list of str
            Which features to use. If None, use all.

        Returns
        -------
        subset : pandas.DataFrame
        """
        if feature_ids is None:
            feature_ids = data.columns
        if sample_ids is None:
            sample_ids = data.index

        sample_ids = pd.Index(set(sample_ids).intersection(data.index))
        feature_ids = pd.Index(set(feature_ids).intersection(data.columns))

        if len(sample_ids) == 1:
            sample_ids = sample_ids[0]
            single_sample = True
        else:
            single_sample = False

        if len(feature_ids) == 1:
            feature_ids = feature_ids[0]
            single_feature = True
        else:
            single_feature = False

        subset = data.ix[sample_ids]
        subset = subset.T.ix[feature_ids].T

        if require_min_samples and not single_feature:
            subset = subset.ix[:, subset.count() > self.min_samples]
        return subset

    def _subset_singles_and_pooled(self, singles, pooled, sample_ids=None,
                                   feature_ids=None):
        # singles_ids = self.data.index.intersection(sample_ids)
        # pooled_ids = self.pooled.index.intersection(sample_ids)
        # # import pdb; pdb.set_trace()
        singles = self._subset(singles, sample_ids, feature_ids,
                               require_min_samples=True)

        # If the sample ids don't overlap with the pooled sample, assume you
        # want all the pooled samples
        if sample_ids is not None and sum(pooled.index.isin(sample_ids)) > 0:
            pooled_sample_ids = sample_ids
        else:
            pooled_sample_ids = None
        pooled = self._subset(pooled, pooled_sample_ids, feature_ids,
                              require_min_samples=False)
        if len(feature_ids) > 1:
            # These are DataFrames
            singles, pooled = singles.align(pooled, axis=1, join='inner')
        else:
            # These are Seriessssss
            singles = singles.dropna()
            pooled = pooled.dropna()

        return singles, pooled

    def _subset_and_standardize(self, data, sample_ids=None,
                                feature_ids=None,
                                standardize=True, return_means=False,
                                rename=False):

        """Take only the sample ids and feature ids from this data, require
        at least some minimum samples, and standardize data using
        scikit-learn. Will also fill na values with the mean of the feature
        (column)

        Parameters
        ----------
        data : pandas.DataFrame
            The data you want to standardize
        sample_ids : None or list of strings
            If None, all sample ids will be used, else only the sample ids
            specified
        feature_ids : None or list of strings
            If None, all features will be used, else only the features
            specified
        standardize : bool
            Whether or not to "whiten" (make all variables uncorrelated) and
            mean-center via sklearn.preprocessing.StandardScaler

        Returns
        -------
        subset : pandas.DataFrame
            Subset of the dataframe with the requested samples and features,
            and standardized as described
        means : pandas.DataFrame
            Mean values of the features (columns). Ignores NAs.

        """

        # fill na with mean for each event
        subset = self._subset(data, sample_ids, feature_ids)
        means = subset.mean()
        subset = subset.fillna(means).fillna(0)

        if rename:
            means = means.rename_axis(self.feature_renamer)
            subset = subset.rename_axis(self.feature_renamer, 1)

        # whiten, mean-center
        if standardize:
            data = StandardScaler().fit_transform(subset)
        else:
            data = subset

        # "data" is a matrix so need to transform it back into a convenient
        # dataframe
        subset = pd.DataFrame(data, index=subset.index,
                              columns=subset.columns)
        if return_means:
            return subset, means
        else:
            return subset

    def plot_clusteredheatmap(self, sample_ids, feature_ids,
                              metric='euclidean',
                              linkage_method='average',
                              sample_colors=None,
                              feature_colors=None, figsize=None,
                              require_min_samples=True):
        subset, row_linkage, col_linkage = self._calculate_linkage(
            sample_ids, feature_ids, linkage_method=linkage_method,
            metric=metric)

        if figsize is None:
            figsize = reversed(subset.shape)
            figsize = map(lambda x: max(.25 * x, 1000), figsize)

        col_kws = dict(linkage_matrix=col_linkage, side_colors=feature_colors,
                       label=map(self.feature_renamer, subset.columns))
        row_kws = dict(linkage_matrix=row_linkage, side_colors=sample_colors)
        return sns.clusteredheatmap(subset, row_kws=row_kws, col_kws=col_kws,
                                    pcolormesh_kws=dict(linewidth=0.01),
                                    figsize=figsize)

    @memoize
    def reduce(self, data, sample_ids, feature_ids,
               featurewise=False,
               reducer=PCAViz,
               standardize=True,
               title='',
               reducer_kwargs=None,
               color=None,
               groupby=None, label_to_color=None, label_to_marker=None,
               order=None, bins=None):
        """Make and memoize a reduced dimensionality representation of data

        Parameters
        ----------
        data : pandas.DataFrame
            samples x features data to reduce
        sample_ids : None or list of strings
            If None, all sample ids will be used, else only the sample ids
            specified
        feature_ids : None or list of strings
            If None, all features will be used, else only the features
            specified
        featurewise : bool
            Whether or not to use the features as the "samples", e.g. if you
            want to reduce the features in to "sample-space" instead of
            reducing the samples into "feature-space"
        standardize : bool
            Whether or not to "whiten" (make all variables uncorrelated) and
            mean-center via sklearn.preprocessing.StandardScaler
        title : str
            Title of the plot
        reducer_kwargs : dict
            Any additional arguments to send to the reducer

        Returns
        -------
        reducer_object : flotilla.compute.reduce.ReducerViz
            A ready-to-plot object containing the reduced space
        """

        reducer_kwargs = {} if reducer_kwargs is None else reducer_kwargs
        reducer_kwargs['title'] = title

        subset, means = self._subset_and_standardize(data,
                                                     sample_ids, feature_ids,
                                                     standardize,
                                                     return_means=True)
        if bins is not None:
            subset = self.binify(subset, bins)

        # compute reduction
        if featurewise:
            subset = subset.T

        reducer_object = reducer(subset,
                                 feature_renamer=self.feature_renamer,
                                 label_to_color=label_to_color,
                                 label_to_marker=label_to_marker,
                                 groupby=groupby, order=order,
                                 data_type=self.data_type, color=color,
                                 DataModel=self,
                                 **reducer_kwargs)
        reducer_object.means = means
        return reducer_object

    def _calculate_linkage(self, data, sample_ids, feature_ids,
                           metric='euclidean',
                           linkage_method='median', standardize=True,
                           require_min_samples=True):

        subset = self._subset_and_standardize(data, sample_ids,
                                              feature_ids,
                                              standardize=standardize)
        row_linkage, col_linkage = self.clusterer(subset, metric,
                                                  linkage_method)
        return subset, row_linkage, col_linkage

    def binify(self, data, bins=None):
        return binify(data, bins)

    def _violinplot(self, feature_id, sample_ids=None,
                    phenotype_groupby=None,
                    phenotype_order=None, ax=None, color=None,
                    label_pooled=True):

        """For compatiblity across data types, can specify _violinplot
        """
        singles, pooled = self._subset_singles_and_pooled(
            self.data, self.pooled, sample_ids, [feature_id])

        outliers = self._subset(self.outliers, feature_ids=[feature_id])

        renamed = self.feature_renamer(feature_id)
        title = '{}\n{}'.format(renamed, ':'.join(feature_id.split(':')[:2]))

        violinplot(singles, groupby=phenotype_groupby, color=color,
                   pooled_data=pooled, order=phenotype_order,
                   title=title, data_type=self.data_type, ax=ax,
                   label_pooled=label_pooled, outliers=outliers)
