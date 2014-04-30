__author__ = 'lovci'
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

"""

object library

commonly used data objects

"""

# <codecell>

import sys

import pandas as pd
import os

from flotilla_params import flotilla_data

from skiff import link_to_list, hg19GO

from ..project.data import *
from ..project.project_params import *


# <codecell>

rbps_file = os.path.join(flotilla_data, "rbps_list")
confident_rbp_file = os.path.join(flotilla_data, "all_pfam_defined_rbps_uniq.txt")


try:
    rbps = pd.read_pickle(os.path.join(flotilla_data, "rbps.df"))
    confident_rbps = pd.read_pickle(os.path.join(flotilla_data, "confident_rpbs.df"))
    splicing_genes = pd.read_pickle(os.path.join(flotilla_data, "splicing_genes.df"))
    tfs = pd.read_pickle(os.path.join(flotilla_data, "tfs.df"))

except:

    sys.stderr.write("rebuilding gene list objects from text and databases...\n")

    go_tool = hg19GO()

    rbps = pd.read_table(rbps_file).set_index("Ensembl_ID")
    rbps.to_pickle(os.path.join(flotilla_data, "rbps.df"))

    with open(confident_rbp_file, 'r') as f:
        confident_rbps = set(map(str.strip, f.readlines()))
    rbps = rbps.ix[pd.Series(rbps.index.unique()).dropna()]

    confident_rbps = rbps.select(lambda x: rbps.GeneSymbol[x] in confident_rbps, 0)

    confident_rbps.to_pickle(os.path.join(flotilla_data, "confident_rpbs.df"))

    splicing_genes = set(go_tool.GO['GO:0008380']['genes']) | \
                     set(go_tool.GO['GO:0000381']['genes']) | \
                     set(go_tool.GO['GO:0006397']['genes'])
    splicing_genes = rbps.select(lambda x: x in splicing_genes)

    splicing_genes.to_pickle(os.path.join(flotilla_data, "splicing_genes.df"))

    tfs = link_to_list("http://www.bioguo.org/AnimalTFDB/download/gene_list_of_Homo_sapiens.txt")

    with open(os.path.join(flotilla_data, "gene_list_of_Homo_sapiens.txt"), 'r') as f:
        xx = f.readlines()
        tfs = pd.Series(map(lambda x: go_tool.geneNames(x.strip()), xx), index= map(str.strip, xx))
    tfs.to_pickle(os.path.join(flotilla_data, "tfs.df"))
#splicing_genes

# <codecell>



sys.stderr.write("importing GO...")
go = hg19GO()
sys.stderr.write("done.\n")