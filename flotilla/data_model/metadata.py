import itertools
import sys

import matplotlib as mpl
import pandas as pd

from .base import BaseData
from ..visualize.color import str_to_color, set1



# Any informational data goes here

class MetaData(BaseData):
    def __init__(self, data, phenotype_order=None, phenotype_to_color=None,
                 phenotype_to_marker=None, phenotype_col='phenotype',
                 predictor_config_manager=None):
        super(MetaData, self).__init__(data, outliers=None,
                                       predictor_config_manager=predictor_config_manager)

        self.phenotype_col = phenotype_col
        self.phenotype_order = phenotype_order
        self.phenotype_to_color = phenotype_to_color

        # Convert color strings to non-default matplotlib colors
        if self.phenotype_to_color is not None:
            for phenotype, color in self.phenotype_to_color.iteritems():
                try:
                    color = str_to_color[color]
                    self.phenotype_to_color[phenotype] = color
                except KeyError:
                    pass

        self.phenotype_to_marker = phenotype_to_marker
        if self.phenotype_to_marker is not None:
            for phenoytpe, marker in self.phenotype_to_marker.iteritems():
                if marker not in mpl.markers.MarkerStyle.filled_markers:
                    sys.stderr.write(
                        '{} is not a valid matplotlib marker style, '
                        'falling back on "o" (circle)'.format(marker))
                    self.phenotype_to_marker[phenotype] = 'o'

    @property
    def phenotype_order(self):
        return self._phenotype_order

    @phenotype_order.setter
    def phenotype_order(self, value):
        if value is not None:
            self._phenotype_order = value
        else:
            self._phenotype_order = list(sorted(self.sample_id_to_phenotype
                                                .unique()))

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
        if self.phenotype_col in self.data:
            return self.data[self.phenotype_col]
        else:
            return pd.Series([self.phenotype_col],
                             index=self.data.index)
