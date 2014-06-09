__author__ = 'lovci'

import os
from .study_data import study_data_dir

#minimum number of samples for analyses
min_samples = 12
#not all studies are single-cell studies... but the terms are mixed, for backwards-compatibility, keep both
min_cells = min_samples
# TODO: remove min_cells references

#metadat about samples, psi columns and genes
sample_descriptors_data_dump = os.path.join(study_data_dir, "descriptors.df")
event_descriptors_data_dump = os.path.join(study_data_dir, "miso_to_ids.df")
gene_descriptors_data_dump = None

#pickled pandas DataFrames:
splicing_data_dump = os.path.join(study_data_dir, "psi.df")
expression_data_dump = os.path.join(study_data_dir, "rpkm.df")

#fill this in if you want to use carrier
#mongoHost, mongoPort = #host, port

#default boolean column to in interactive widgets
default_group_id = 'any_cell'

#for menus items in interactive widgets, if there are several lists that should be options
#can be all boolean columns in sample_descriptors
default_group_ids = ['any_cell',]

#default list to use. 'variant' is automatically calculated for SplicingData and ExpressionData
default_list_id = 'variant'

#for menus items, if there are several lists that should be options
default_list_ids = ['variant',]

default_gene_list = default_list_id
default_event_list = default_list_id

default_gene_list_ids = ['variant', 'all_genes'] #lists of interesting gene IDs
default_event_ids = [] # lists of interesting splicing events

study_name = "barebones_project" # not used but helpful in the future, probably.

species = "hg19"