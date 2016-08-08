"""

"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)


class SupplementalData(object):

    def __init__(self, name_to_df=None):
        """Container for holding any arbitrary pandas dataframes

        All attributes of type pandas.DataFrame will be saved with the study.

        Parameters
        ----------
        name_to_df : dict
            Mapping of a string of the name of the desired attribute, to the
            pandas dataframe
        """
        if name_to_df is not None:
            for name, df in name_to_df.items():
                self.__setattr__(name, df)
