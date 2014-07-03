import numpy as np
import pandas as pd

from .infotheory import jsd


class Modalities(object):
    modalities_bins = np.array([[1, 0, 0],  # excluded
                                [0, 1, 0],  # middle
                                [0, 0, 1],  # included
                                [1, 0, 1],  # bimodal
                                [1, 1, 1]])  # uniform

    modalities_names = ['included', 'middle', 'excluded', 'bimodal', 'uniform']

    true_modalities = pd.DataFrame(modalities_bins, index=modalities_names)

    def __init__(self):
        pass

    def _col_jsd_modalities(self, col):
        return self.true_modalities.apply(lambda x: jsd(x, col), axis=0)

    def sqrt_jsd_modalities(self, binned):
        """Return the square root of the JSD of each splicing event versus
        all the modalities. Use square root of JSD because it's a metric.

        """
        return binned.apply(self._col_jsd_modalities, axis=0)

