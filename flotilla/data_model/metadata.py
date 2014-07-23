import sys

from .base import BaseData
from ..visualize.color import str_to_color


# Any informational data goes here

class MetaData(BaseData):
    def __init__(self, data, phenotype_order=None, phenotype_to_color=None,
                 phenotype_to_marker=None):
        super(MetaData, self).__init__(data, outliers=None)
        self.phenotype_order = phenotype_order
        self.phenotype_to_color = phenotype_to_color
        for phenotype, color in self.phenotype_to_color.iteritems():
            try:
                color = str_to_color[color]
                self.phenotype_to_color[phenotype] = color
            except KeyError:
                pass

        self.phenotype_to_marker = phenotype_to_marker


    def _get(self, sample_metadata_filename=None, gene_metadata_filename=None,
             event_metadata_filename=None):

        metadata = {'sample': None,
                    'gene': None,
                    'event': None}
        try:
            if sample_metadata_filename is not None:
                metadata['sample'] = self.load(*sample_metadata_filename)
            if event_metadata_filename is not None:
                metadata['event'] = self.load(*event_metadata_filename)
            if gene_metadata_filename is not None:
                metadata['gene'] = self.load(*gene_metadata_filename)

        except Exception as E:
            sys.stderr.write("error loading descriptors: %s, \n\n .... "
                             "entering pdb ... \n\n" % E)
            raise E

        return {'experiment_design_data': metadata['sample'],
                'feature_data': metadata['gene'],
                'event_metadata': metadata['event'],
                'expression_metadata': None}
