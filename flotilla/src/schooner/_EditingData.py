__author__ = 'olga'
from singlesail.data_model._Data import Data

class EditingData(Data):
    """Class for RNA-editing data

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, df):
        """Constructor for EditingData

        Parameters
        ----------
        editing_matrix_filename

        Returns
        -------


        Raises
        ------

        """
        self.df = df