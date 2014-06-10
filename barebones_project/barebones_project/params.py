import os

#essential params
from .study_data import study_data_dir
min_samples = 12

#recommended parameters:
#minimum number of samples for analyses

sample_metadata_filename = None
#sample_metadata_filename = os.path.join(study_data_dir, "metadata.tsv")

event_metadata_filename = None
#event_metadata_filename = os.path.join(study_data_dir, "miso_to_ids.df")

gene_metadata_filename = None

expression_filename = None
#expression_filename = os.path.join(study_data_dir, "expression.pickle")

splicing_filename = None
#splicing_filename = os.path.join(study_data_dir, "splicing.pickle")

subclasses = []
#subclasses = ['expression', 'splicing'] #optional subclasses to initialize on load

#optional params
###fill this in if you want to use carrier (you don't)

#mongoHost, mongoPort = #host, port

###end carrier-optional params

###fill this in if you want to use the schooner.InteractiveStudy

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

###end schooner.InteractiveStudy-optional params
