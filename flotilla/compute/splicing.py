import numpy as np
import pandas as pd


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

    def jsd_modalities(self, row):
        return true_modalities.apply(lambda x: gmath.jsd(x, psi_binned_row),
                                     axis=1)

    modality_jsd = psi_binned.head().apply(jsd_modalities, axis=1)