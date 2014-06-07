__author__ = 'lovci'

import os, sys
from ..src.utils import data_dir

study_data_dir = data_dir()
with open(os.path.join(study_data_dir, "README"), 'r') as f:
    for line in f.readlines():
        if not line.startswith("#"):
            sys.stderr.write(line)

from ..params import expression_data_dump, splicing_data_dump, \
    sample_descriptors_data_dump, event_descriptors_data_dump, gene_descriptors_data_dump

from ..src.loaders import load_descriptors, load_transcriptome_data


sample_info, gene_info, splicing_info = load_descriptors(sample_descriptors_data_dump, gene_descriptors_data_dump,
                                                         event_descriptors_data_dump)

expression_info = None

splicing, expression = load_transcriptome_data(expression_data_dump, splicing_data_dump)

sparse_expression = expression[expression > 0] # rpkms that we actually counted, not .1 artifacts