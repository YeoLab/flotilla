import numpy as np
import pandas as pd

from .infotheory import jsd, binify


class Modalities(object):
    modalities_bins = np.array([[1, 0, 0],  # excluded
                                [0, 1, 0],  # middle
                                [0, 0, 1],  # included
                                [1, 0, 1],  # bimodal
                                [1, 1, 1]])  # uniform

    modalities_names = ['included', 'middle', 'excluded', 'bimodal', 'uniform']

    true_modalities = pd.DataFrame(modalities_bins.T, columns=modalities_names)

    def __init__(self, psi, excluded_max=0.2, included_min=0.8):
        self.bins = (0, excluded_max, included_min, 1)

        self.binned = binify(psi, self.bins)
        self.true_modalities.index = self.binned.index

    def _col_jsd_modalities(self, col):
        return self.true_modalities.apply(lambda x: jsd(x, col), axis=0)

    @property
    def sqrt_jsd_modalities(self):
        """Return the square root of the JSD of each splicing event versus
        all the modalities. Use square root of JSD because it's a metric.

        """
        return np.sqrt(self.binned.apply(self._col_jsd_modalities, axis=0))

    @property
    def modality_assignments(self):
        return self.true_modalities.columns[
            np.argmin(self.sqrt_jsd_modalities.values, axis=0)]

