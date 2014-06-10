__author__ = 'lovci'

import params
from params import sample_metadata_filename, event_metadata_filename, gene_metadata_filename, \
    expression_filename, splicing_filename
from .src.loaders import load_metadata, load_expression_data, load_splicing_data, DataLoaderManager
from flotilla import schooner

data_loaders = DataLoaderManager() #wrappers for all the loading-things

if sample_metadata_filename is not None and event_metadata_filename is not None:

    data_loaders.register_loader(load_metadata,
                                 sample_metadata_filename=sample_metadata_filename,
                                 gene_metadata_filename=gene_metadata_filename,
                                 event_metadata_filename=event_metadata_filename,
                                 )
else:
    raise RuntimeError("at least set metadata")

if expression_filename is not None:
    data_loaders.register_loader(load_expression_data, expression_filename=expression_filename)

if splicing_filename is not None:
    data_loaders.register_loader(load_splicing_data, splicing_filename=splicing_filename)

def embark(load_cargo=True, drop_outliers=False):

    """return a flotilla study with the works
    TODO:datatypes = 'all' or a list of acceptable *Data schooner classes to load"""
    study = schooner.FlotillaStudy(params_dict=vars(params), data_loaders=data_loaders)
    return study