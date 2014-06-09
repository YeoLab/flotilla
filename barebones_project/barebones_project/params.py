__author__ = 'lovci'

import os
from .src.utils import data_dir

study_data_dir = data_dir()

min_cells = 12

letters = #[... group ids ... ]
#transitions = ["PN", "NM", "MS", ] not used yet.

#not used, but this is the data source.
#original_rpkm_file = "/nas/nas0/scratch/ppliu/single_cell/analysis/final_groups/final_table"
#original_splice_file = "/nas3/obot/projects/singlecell/miso_psis_filtered.json"

sample_metadata_filename = os.path.join(study_data_dir, "descriptors.df")
event_metadata_filename = os.path.join(study_data_dir, "miso_to_ids.df")
gene_metadata_filename = None

splicing_filename = os.path.join(study_data_dir, "psi.df")
expression_filename = os.path.join(study_data_dir, "rpkm.df")

mongoHost, mongoPort = #host, port

default_group_id = 'any_cell'
default_group_ids = ['any_cell',]


default_list_id = 'variant' #default list to use
default_list_ids = ['variant',] #for menus items, if there are several lists that should be options

default_gene_list = default_list_id
default_event_list = default_list_id

default_gene_list_ids = ['variant', 'all_genes'] #lists of interesting gene IDs
default_event_ids = [] # lists of interesting splicing events

study_name = "neural differentiation" # not used

species = "hg19"