__author__ = 'lovci'

import params
from params import sample_metadata_filename, event_metadata_filename, gene_metadata_filename, \
    expression_data_filename, splicing_data_filename
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

if expression_data_filename is not None:
    data_loaders.register_loader(load_expression_data, expression_data_filename=expression_data_filename)

if splicing_data_filename is not None:
    data_loaders.register_loader(load_splicing_data, splicing_data_filename=splicing_data_filename)

def embark(load_cargo=False, drop_outliers=False):

    """return a flotilla study with the works
    TODO:datatypes = 'all' or a list of acceptable *Data schooner classes to load"""
    import sys

    study = schooner.FlotillaStudy(vars(params), data_loaders, load_cargo=load_cargo, drop_outliers=drop_outliers)
    return study
