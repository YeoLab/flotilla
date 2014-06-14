__author__ = 'lovci'
from _Data import Data
from _ExpressionData import ExpressionData
from _SplicingData import SplicingData

class Transcriptome(object):

    def _get(self, expression_data_filename, splicing_data_filename):
        try:
            splicing = self.get_splicing_data(splicing_data_filename)['splicing_df']
            expression = self.get_expression_data(expression_data_filename)['expression_df']
            sparse_expression = expression[expression > 0]

        except Exception as E:
            sys.stderr.write("error loading transcriptome data: %s, \n\n .... entering pdb ... \n\n" % E)
            raise E

        return {'splicing_df': splicing,
                'expression_df': expression,
                'sparse_expression_df': sparse_expression}