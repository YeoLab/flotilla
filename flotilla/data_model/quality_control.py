from .base import BaseData


MIN_READS = 2e6


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
        super(MappingStatsData, self).__init__(data,
                                               predictor_config_manager=predictor_config_manager)
        self.number_mapped_col = number_mapped_col
        self.min_reads = min_reads

    @property
    def too_few_mapped(self):
        return self.mapped_reads.index[self.mapped_reads < self.min_reads]

    @property
    def mapped_reads(self):
        return self.data[self.number_mapped_col]
