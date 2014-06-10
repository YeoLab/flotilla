__author__ = 'lovci'

import params
from params import sample_metadata_filename, event_metadata_filename, gene_metadata_filename, \
    expression_filename, splicing_filename
from .src.loaders import load_metadata, load_expression_data, load_splicing_data
from flotilla import schooner

data_loaders = []

if sample_metadata_filename is not None and event_metadata_filename is not None:
    from .src.loaders import load_metadata

    metadata_dict = {
        'sample_metadata_filename': sample_metadata_filename,
        'gene_metadata_filename': gene_metadata_filename,
        'event_metadata_filename': event_metadata_filename,
    }
    data_loaders.append(metadata_dict, load_metadata)

if expression_filename is not None:
    from .src.loaders import load_expression_data
    expression_dict = {
        'expression_filename': expression_filename
    }
    data_loaders.append(expression_dict, load_expression_data)

if splicing_filename is not None:
    from .src.loaders import load_splicing_data
    splicing_dict = {
        'splicing_filename':splicing_filename,
    }
    data_loaders.append(splicing_dict, load_splicing_data)

study = None
def embark(load_cargo=True, drop_outliers=False, datatypes='all'):

    """return a flotilla study with the works
    datatypes = 'all' or a list of acceptable *Data schooner classes to load"""
    interactive_args = {'load_cargo': load_cargo, 'drop_outliers': drop_outliers}
    study = schooner.FlotillaStudy(data_loaders, datatypes=datatypes,
                                   params_dict=vars(params), **interactive_args)
    return study