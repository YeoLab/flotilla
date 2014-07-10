from .base import BaseData


class MappingStatsData(BaseData):
    """Constructor for mapping statistics data from STAR

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, data, number_mapped_col):
        """Constructor for MappingStatsData

        Parameters
        ----------
        data, sample_descriptors

        Returns
        -------


        Raises
        ------

        """
        super(MappingStatsData, self).__init__(data)
        self.number_mapped_col = number_mapped_col

        pass

    @property
    def mapped_reads(self):
        return self.data[self.number_mapped_col]
