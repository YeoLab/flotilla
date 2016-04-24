"""

"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from .base import BaseData


MIN_READS = 5e5


class MappingStatsData(BaseData):
    """Constructor for mapping statistics data from STAR

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, data, number_mapped_col, min_reads=MIN_READS,
                 predictor_config_manager=None):
        """Constructor for MappingStatsData

        Parameters
        ----------
        data, sample_descriptors

        Returns
        -------


        Raises
        ------

        """
        super(MappingStatsData, self).__init__(
            data, predictor_config_manager=predictor_config_manager)
        self.number_mapped_col = number_mapped_col
        self.min_reads = min_reads

    @property
    def too_few_mapped(self):
        return self.number_mapped.index[self.number_mapped < self.min_reads]

    @property
    def number_mapped(self):
        return self.data[self.number_mapped_col]
