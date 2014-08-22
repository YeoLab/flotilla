import itertools
import sys

import matplotlib as mpl
import seaborn as sns

from .base import BaseData
from ..visualize.color import str_to_color, set1



# Any informational data goes here

class MetaData(BaseData):
    _default_phenotype_col = 'phenotype'
    _default_pooled_col = 'pooled'

    def __init__(self, data, phenotype_order=None, phenotype_to_color=None,
                 phenotype_to_marker=None,
                 phenotype_col=_default_phenotype_col,
                 pooled_col=None,
                 predictor_config_manager=None):
        super(MetaData, self).__init__(data, outliers=None,
                                       predictor_config_manager=predictor_config_manager)

        self.phenotype_col = phenotype_col if phenotype_col is not None else \
            self._default_phenotype_col
        self.phenotype_order = phenotype_order
        self.phenotype_to_color = phenotype_to_color
        self.pooled_col = pooled_col

        if self.phenotype_col not in self.data:
            sys.stderr.write('The required column name "{}" does not exist in '
                             'the sample metadata. All samples will be '
                             'treated as the same phenotype. You may also '
                             'specify "phenotype_col" in the metadata section '
                             'of the datapackge.'.format(self.phenotype_col))
            self.data[self.phenotype_col] = 'phenotype'
            self.phenotype_order = None
            self.phenotype_to_color = None

        # Convert color strings to non-default matplotlib colors
        if self.phenotype_to_color is not None:
            for phenotype, color in self.phenotype_to_color.iteritems():
                try:
                    color = str_to_color[color]
                    self.phenotype_to_color[phenotype] = color
                except KeyError:
                    pass
        else:
            sys.stderr.write('No phenotype to color mapping was provided, '
                             'so coming up with reasonable defaults')
            self.phenotype_to_color = {}
            colors = sns.color_palette(n_colors=self.n_phenotypes)
            for phenotype, color in zip(self.unique_phenotypes, colors):
                self.phenotype_to_color[phenotype] = color

        self.phenotype_to_marker = phenotype_to_marker
        if self.phenotype_to_marker is not None:
            for phenoytpe, marker in self.phenotype_to_marker.iteritems():
                if marker not in mpl.markers.MarkerStyle.filled_markers:
                    sys.stderr.write(
                        '{} is not a valid matplotlib marker style, '
                        'falling back on "o" (circle)'.format(marker))
                    self.phenotype_to_marker[phenotype] = 'o'
        else:
            sys.stderr.write('No phenotype to marker (matplotlib plotting '
                             'symbol) was provided, so each phenotype will be '
                             'plotted as a circle in the PCA visualizations.')
            self.phenotype_to_marker = dict.fromkeys(
                self.sample_id_to_phenotype.unique(), 'o')

    @property
    def n_phenotypes(self):
        return len(self.self.unique_phenotypes)

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
    def phenotype_to_color(self):
        return self._phenotype_to_color

    @phenotype_to_color.setter
    def phenotype_to_color(self, value):
        if value is not None:
            self._phenotype_to_color = value
        else:
            self._phenotype_to_color = dict(zip(self.phenotype_order,
                                                itertools.cycle(set1)))

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