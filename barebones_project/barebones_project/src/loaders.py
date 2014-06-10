__author__ = 'lovci'

""" loads pre-made pickle files """

import sys
import pandas as pd

def load_metadata(sample_descriptors_data_dump=None, gene_descriptors_data_dump=None,
                     event_descriptors_data_dump=None):

    descrip = {'sample':None,
               'gene':None,
               'event':None}
    try:
        if sample_descriptors_data_dump is not None:
            descrip['sample'] = pd.read_pickle(sample_descriptors_data_dump)
        if event_descriptors_data_dump is not None:
            descrip['event'] = pd.read_pickle(event_descriptors_data_dump)
        if gene_descriptors_data_dump is not None:
            descrip['gene'] = pd.read_pickle(gene_descriptors_data_dump)

    except Exception as E:
        sys.stderr.write("error loading descriptors: %s, \n\n .... entering pdb ... \n\n" % E)
        import pdb
        pdb.set_trace()
        raise
    return {'sample_metadata': descrip['sample'],
            'gene_metadata': descrip['gene'],
            'event_metadata': descrip['event']}

def load_splicing_data(splicing_data_file):
    return {'splicing': pd.read_pickle(splicing_data_file)}

def load_expression_data(expression_data_file):
    return {'expression': pd.read_pickle(expression_data_file)}

def load_transcriptome_data(expression_data_dump, splicing_data_dump):
    try:
        splicing = load_splicing_data(splicing_data_dump)['splicing']
        expression = load_expression_data(expression_data_dump)['expression']
        sparse_expression = expression[expression > 0]

    except Exception as E:
        sys.stderr.write("error loading transcriptome data: %s, \n\n .... entering pdb ... \n\n" % E)
        import pdb
        pdb.set_trace()
        raise

    return {'splicing': splicing,
            'expression': expression,
            'sparse_expression': sparse_expression}
