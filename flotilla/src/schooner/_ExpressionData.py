from singlesail.data_model._Data import Data

class ExpressionData(Data):
    """Data model for gene expression data

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, df):
        """Constructor for ExpressionData

        Parameters
        ----------


        Returns
        -------


        Raises
        ------

        """
        self.df = df

    def coeff_var(self):
        """Plot coefficient of variance for all cells, then within

        """