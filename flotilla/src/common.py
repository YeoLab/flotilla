__author__ = 'lovci'

"""

object library

commonly used data objects

"""

import sys

import pandas as pd
import os

from flotilla_params import flotilla_data

from gene_ontology import link_to_list, hg19GO, neuro_genes_human


rbps_file = os.path.join(flotilla_data, "rbps_list")
confident_rbp_file = os.path.join(flotilla_data, "all_pfam_defined_rbps_uniq.txt")

sys.stderr.write("importing GO...")
go = hg19GO()
sys.stderr.write("done.\n")

try:
    rbps = pd.read_pickle(os.path.join(flotilla_data, "rbps.df"))
    confident_rbps = pd.read_pickle(os.path.join(flotilla_data, "confident_rpbs.df"))
    splicing_genes = pd.read_pickle(os.path.join(flotilla_data, "splicing_genes.df"))
    tfs = pd.read_pickle(os.path.join(flotilla_data, "tfs.df"))


except:

    sys.stderr.write("rebuilding gene list objects from text and databases...\n")


    rbps = pd.read_table(rbps_file).set_index("Ensembl_ID")
    rbps.to_pickle(os.path.join(flotilla_data, "rbps.df"))

    with open(confident_rbp_file, 'r') as f:
        confident_rbps = set(map(str.strip, f.readlines()))
    rbps = rbps.ix[pd.Series(rbps.index.unique()).dropna()]

    confident_rbps = rbps.select(lambda x: rbps.GeneSymbol[x] in confident_rbps, 0)

    confident_rbps.to_pickle(os.path.join(flotilla_data, "confident_rpbs.df"))

    splicing_genes = set(go.GO['GO:0008380']['genes']) | \
                     set(go.GO['GO:0000381']['genes']) | \
                     set(go.GO['GO:0006397']['genes'])
    splicing_genes = rbps.select(lambda x: x in splicing_genes)

    splicing_genes.to_pickle(os.path.join(flotilla_data, "splicing_genes.df"))

    tfs = link_to_list("http://www.bioguo.org/AnimalTFDB/download/gene_list_of_Homo_sapiens.txt")

    with open(os.path.join(flotilla_data, "gene_list_of_Homo_sapiens.txt"), 'r') as f:
        xx = f.readlines()
        tfs = pd.Series(map(lambda x: go.geneNames(x.strip()), xx), index= map(str.strip, xx))
    tfs.to_pickle(os.path.join(flotilla_data, "tfs.df"))

gene_lists = dict([('confident_rbps', confident_rbps),
                   ('rbps', rbps),
                   ('splicing_genes', splicing_genes),
                   ('marker_genes', pd.Series(map(go.geneNames, neuro_genes_human), index = neuro_genes_human)),
                   ('tfs', tfs)
])



