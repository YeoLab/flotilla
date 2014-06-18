__author__ = 'lovci'
"""
Data model for anything informational, like metadata about samples or mapping
stats data
"""


import sys
from base import BaseData

# Any informational data goes here

class MetaData(BaseData):
    def _get(self, sample_metadata_filename=None, gene_metadata_filename=None,
                     event_metadata_filename=None):

        metadata = {'sample':None,
                   'gene':None,
                   'event':None}
        try:
            if sample_metadata_filename is not None:
                metadata['sample'] = self.load(*sample_metadata_filename)
            if event_metadata_filename is not None:
                metadata['event'] = self.load(*event_metadata_filename)
            if gene_metadata_filename is not None:
                metadata['gene'] = self.load(*gene_metadata_filename)

        except Exception as E:
            sys.stderr.write("error loading descriptors: %s, \n\n .... entering pdb ... \n\n" % E)
            raise E

        return {'sample_metadata': metadata['sample'],
                'gene_metadata': metadata['gene'],
                'event_metadata': metadata['event'],
                'expression_metadata': None}

class MappingStatsData(BaseData):
    """Constructor for mapping statistics data from STAR

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, df, sample_descriptors):
        """Constructor for MappingStatsData

        Parameters
        ----------
        df, sample_descriptors

        Returns
        -------


        Raises
        ------

        """
        super(MappingStatsData).__init__()
        pass