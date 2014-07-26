from .base import BaseData


class MappingStatsData(BaseData):
    """Constructor for mapping statistics data from STAR

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, data, number_mapped_col, predictor_config_manager=None):
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

        pass

    @property
    def mapped_reads(self):
        return self.data[self.number_mapped_col]
