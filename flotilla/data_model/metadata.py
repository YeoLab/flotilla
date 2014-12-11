from collections import defaultdict
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
                                       predictor_config_manager=predictor_config_manager)

        self.phenotype_col = phenotype_col if phenotype_col is not None else \
            self._default_phenotype_col
        self.phenotype_order = phenotype_order
        self.phenotype_to_color = phenotype_to_color
        self.pooled_col = pooled_col

        phenotypes_not_in_order = set(self.unique_phenotypes).difference(set(self.phenotype_order))

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
            colors = iter(self._colors)
            for phenotype in sorted(self.unique_phenotypes):
                try:
                    color = self.phenotype_to_color[phenotype]
                except KeyError:
                    sys.stderr.write('No color was assigned to the phenotype {}, '
                                  'assigning a random color'.format(phenotype))
                    color = self._colors.next()
                try:
                    color = str_to_color[color]
                except KeyError:
                    pass
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
    def _colors(self):
        return map(mpl.colors.rgb2hex,
                        sns.color_palette('husl', n_colors=self.n_phenotypes))

    @property
    def unique_phenotypes(self):
        return sorted(self.sample_id_to_phenotype.unique())

    @property
    def _default_phenotype_order(self):
        return list(sorted(self.unique_phenotypes))

    @property
    def phenotype_order(self):
        if len(set(self._phenotype_order) & set(self.unique_phenotypes)) > 0:
            return [v for v in self._phenotype_order if v in self.unique_phenotypes]
        else:
            return self._default_phenotype_order

    @phenotype_order.setter
    def phenotype_order(self, value):
        if value is not None:
            self._phenotype_order = value
        else:
            self._phenotype_order = self._default_phenotype_order

    @property
    def _default_phenotype_to_color(self):
        colors = iter(self._colors)
        color_factory = lambda: colors.next()
        return defaultdict(color_factory)

    @property
    def phenotype_to_color(self):
        if len(set(self._phenotype_order) & set(self.unique_phenotypes)) > 0:
            return [v for v in self._phenotype_order if v in self.unique_phenotypes]
        else:
            return self._default_phenotype_to_color

    @phenotype_to_color.setter
    def phenotype_to_color(self, value):
        if value is not None:
            self._phenotype_to_color = value
        else:
            self._phenotype_to_color = self._default_phenotype_to_color

    @property
    def phenotype_color_order(self):
        return [self.phenotype_to_color[p] for p in self.phenotype_order]

    @property
    def sample_id_to_phenotype(self):
        return self.data[self.phenotype_col]

    @property
    def sample_id_to_color(self):
        return dict((sample_id, self.phenotype_to_color[p])
                    for sample_id, p in
                    self.sample_id_to_phenotype.iteritems())

    @property
    def phenotype_transitions(self):
        return zip(self.phenotype_order[:-1], self.phenotype_order[1:])

    @property
    def sample_subsets(self):
        return subsets_from_metadata(self.data, MINIMUM_SAMPLE_SUBSET,
                                     'samples')

    @property
    def phenotype_series(self):
        return self.data[self.phenotype_col]