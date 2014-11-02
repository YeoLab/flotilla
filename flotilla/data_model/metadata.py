import sys

import matplotlib as mpl
import seaborn as sns

from .base import BaseData, subsets_from_metadata
from ..visualize.color import str_to_color


POOLED_COL = 'pooled'
PHENOTYPE_COL = 'phenotype'
MINIMUM_SAMPLE_SUBSET = 10

# Any informational data goes here

class MetaData(BaseData):
    def __init__(self, data, phenotype_order=None, phenotype_to_color=None,
                 phenotype_to_marker=None,
                 phenotype_col=PHENOTYPE_COL,
                 pooled_col=POOLED_COL,
                 predictor_config_manager=None):
        super(MetaData, self).__init__(data, outliers=None,
                                       predictor_config_manager=predictor_config_manager,
                                       data_type='metadata')

        self.phenotype_col = phenotype_col if phenotype_col is not None else \
            self._default_phenotype_col
        self.phenotype_order = phenotype_order
        self.phenotype_to_color = phenotype_to_color
        self.pooled_col = pooled_col

        phenotypes_not_in_order = set(self.unique_phenotypes).difference(
            set(self.phenotype_order))

        if len(phenotypes_not_in_order) > 0:
            self.phenotype_order.extend(phenotypes_not_in_order)

        if self.phenotype_col not in self.data:
            sys.stderr.write('The required column name "{}" does not exist in '
                             'the sample metadata. All samples will be '
                             'treated as the same phenotype. You may also '
                             'specify "phenotype_col" in the metadata section '
                             'of the datapackage.\n'.format(self.phenotype_col))
            self.data[self.phenotype_col] = 'phenotype'
            self.phenotype_order = None
            self.phenotype_to_color = None

        # Convert color strings to non-default matplotlib colors
        if self.phenotype_to_color is not None:
            colors = iter(
                sns.color_palette('Dark2', n_colors=self.n_phenotypes))
            for phenotype in self.unique_phenotypes:
                try:
                    color = self.phenotype_to_color[phenotype]
                except KeyError:
                    sys.stderr.write(
                        'No color was assigned to the phenotype {}, '
                        'assigning a random color'.format(phenotype))
                    color = mpl.colors.rgb2hex(colors.next())
                try:
                    color = str_to_color[color]
                except KeyError:
                    pass
                self.phenotype_to_color[phenotype] = color
        else:
            sys.stderr.write('No phenotype to color mapping was provided, '
                             'so coming up with reasonable defaults\n')
            self.phenotype_to_color = {}
            colors = sns.color_palette('Dark2', n_colors=self.n_phenotypes)
            for phenotype, color in zip(self.unique_phenotypes, colors):
                self.phenotype_to_color[phenotype] = mpl.colors.rgb2hex(color)
        # Double-make sure that all incoming colors are stored as strings and
        # not lists
        for phenotype in self.phenotype_to_color:
            color = self.phenotype_to_color[phenotype]
            if isinstance(color, list) or isinstance(color, tuple):
                color = mpl.colors.rgb2hex(color)
                self.phenotype_to_color[phenotype] = color

        self.phenotype_to_marker = phenotype_to_marker
        if self.phenotype_to_marker is not None:
            for phenotype in self.unique_phenotypes:
                try:
                    marker = self.phenotype_to_marker[phenotype]
                except KeyError:
                    sys.stderr.write(
                        '{} does not have marker style, '
                        'falling back on "o" (circle)'.format(phenotype))
                    marker = 'o'
                if marker not in mpl.markers.MarkerStyle.filled_markers:
                    sys.stderr.write(
                        '{} is not a valid matplotlib marker style, '
                        'falling back on "o" (circle)'.format(marker))
                    marker = 'o'
                self.phenotype_to_marker[phenotype] = marker

        else:
            sys.stderr.write('No phenotype to marker (matplotlib plotting '
                             'symbol) was provided, so each phenotype will be '
                             'plotted as a circle in the PCA visualizations.\n')
            self.phenotype_to_marker = dict.fromkeys(
                self.sample_id_to_phenotype.unique(), 'o')

    @property
    def n_phenotypes(self):
        return len(self.unique_phenotypes)

    @property
    def unique_phenotypes(self):
        return self.sample_id_to_phenotype.unique()

    @property
    def phenotype_order(self):
        return self._phenotype_order

    @phenotype_order.setter
    def phenotype_order(self, value):
        if value is not None:
            self._phenotype_order = value
        else:
            self._phenotype_order = list(sorted(self.unique_phenotypes))

    @property
    def phenotype_color_order(self):
        return [self.phenotype_to_color[p] for p in self.phenotype_order]

    @property
    def sample_id_to_phenotype(self):
        return self.data[self.phenotype_col]

    @property
    def sample_id_to_color(self):
        return dict((sample_id, self.phenotype_to_color[p])
                    for sample_id, p in self.sample_id_to_phenotype.iteritems())

    @property
    def phenotype_transitions(self):
        return zip(self.phenotype_order[:-1], self.phenotype_order[1:])

    @property
    def sample_subsets(self):
        return subsets_from_metadata(self.data, MINIMUM_SAMPLE_SUBSET,
                                     'samples')

    def merge_boolean_columns(self, columns):
        """Merge boolean columns in data and return logical OR

        Parameters
        ----------
        columns : list-like
            Boolean columns whose attributes to merge

        Returns
        -------
        merged : pandas.Series
            A single boolean column, with True for each row that had True in
            any column

        """
        merged = self.data[columns].any(axis=1)
        return merged

    def set_outliers_by_merging_columns(self, columns):
        """Merge boolean columns of data into "outlier" column

        Parameters
        ----------
        columns : list-like
            Column names whose boolean attributes to merge

        Returns
        -------
        is_ever_an_outlier : pandas.Series
            Boolean series indicating whether this sample was ever indicated
            as an outlier

        """
        UserWarning('Over-writing existing outlier columns\n')
        self.data['outlier'] = False
        print "using thse columns: \n{}\n".format("\n".join(columns))
        is_ever_an_outlier = self.merge_boolean_columns(
            columns)
        print "there are {} outliers".format(is_ever_an_outlier.sum())
        self.data['outlier'] = is_ever_an_outlier
        return is_ever_an_outlier