"""
Base data class for all data types. All data types in flotilla inherit from
this, or a child object (like ExpressionData).
"""
from collections import defaultdict
import sys

import pandas as pd
from scipy.spatial.distance import pdist, squareform

from ..visualize.predict import ClassifierViz


MINIMUM_SAMPLES = 12


class BaseData(object):
    """Generic study_data model for both splicing and expression study_data

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, data=None, feature_data=None,
                 species=None, feature_rename_col=None, outliers=None,
                 min_samples=MINIMUM_SAMPLES):
        """Base class for biological data measurements

        Parameters
        ----------
        experiment_design_data : pandas.DataFrame
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
        if outliers is not None:
            self.data = self.drop_outliers(self.data, outliers)

        # self.experiment_design_data = experiment_design_data
        self.feature_data = feature_data
        self.feature_rename_col = feature_rename_col
        self.min_samples = min_samples
        self.default_feature_sets = []

        self.species = species
        self.feature_sets = {}
        if self.feature_data is not None and self.feature_rename_col is not \
                None:
            def feature_renamer(x):
                if x in self.feature_data[feature_rename_col]:
                    rename = self.feature_data[feature_rename_col][x]
                    if isinstance(rename, pd.Series):
                        return rename.values[0]
                    else:
                        return rename
                else:
                    return x

            self.feature_renamer = feature_renamer
        else:
            self.feature_renamer = lambda x: x

        if self.feature_data is not None:
            for col in self.feature_data:
                if self.feature_data[col].dtype != bool:
                    continue
                self.feature_sets[col] = self.feature_data.index[self
                    .feature_data[col]]

        self.all_features = 'all_genes'

    def drop_outliers(self, df, outliers):
        # assert 'outlier' in self.experiment_design_data.columns
        outliers = set(outliers).intersection(df.index)
        sys.stdout.write("dropping {}".format(outliers))
        return df.drop(outliers)

    def feature_subset_to_feature_ids(self, feature_subset, rename=True):
        if feature_subset is not None:
            if feature_subset in self.feature_sets:
                feature_ids = self.feature_sets[feature_subset]
            elif feature_subset == self.all_features:
                feature_ids = self.data.index
        else:
            feature_ids = self.data.index

        if rename:
            feature_ids = feature_ids.map(self.feature_renamer)
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

    # def _feature_renamer(self, x):
    #     return x
    #
    @property
    def feature_renamer(self):
        return self._feature_renamer

    @feature_renamer.setter
    def feature_renamer(self, renamer):
        self._feature_renamer = renamer

    # TODO.md: Specify dtypes in docstring
    def plot_classifier(self, trait, sample_ids=None, feature_ids=None,
                        standardize=True, predictor=ClassifierViz,
                        predictor_kwargs=None, predictor_scoring_fun=None,
                        score_cutoff_fun=None, **plotting_kwargs):
        """Principal component-like analysis of measurements

        Params
        -------
        obj_id : str
            key of the object getting plotted
        sample_subset : str
            ???
        categorical_trait : str
            predictor feature
        feature_subset : str
            subset of genes to use for building class

        Returns
        -------
        self
        """
        # print trait
        predictor_kwargs = {} if predictor_kwargs is None else predictor_kwargs
        plotting_kwargs = {} if plotting_kwargs is None else plotting_kwargs

        # local_plotting_args = self.pca_plotting_args.copy()
        # local_plotting_args.update(plotting_kwargs)

        clf = self.classify(trait, sample_ids=sample_ids,
                            feature_ids=feature_ids,
                            standardize=standardize, predictor=predictor,
                            predictor_kwargs=predictor_kwargs,
                            predictor_scoring_fun=predictor_scoring_fun,
                            score_cutoff_fun=score_cutoff_fun)
        clf(**plotting_kwargs)
        # clf()
        return self

    def plot_dimensionality_reduction(self, x_pc=1, y_pc=2,  #obj_id=None,
                                      sample_ids=None, feature_ids=None,
                                      featurewise=False,
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


        Returns
        -------
        self

        Raises
        ------

        """
        pca = self.reduce(sample_ids, feature_ids,
                          featurewise=featurewise)
        pca(markers_size_dict=defaultdict(lambda x: 400),
            show_vectors=False,
            title_size=10,
            axis_label_size=10,
            x_pc="pc_" + str(x_pc),
            #this only affects the plot, not the study_data.
            y_pc="pc_" + str(y_pc),
            #this only affects the plot, not the study_data.
            **plotting_kwargs)
        return self

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, values):
        self._min_samples = values