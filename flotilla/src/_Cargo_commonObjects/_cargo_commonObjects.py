__author__ = 'lovci'

"""

object library

commonly used study_data objects

"""

import sys

import pandas as pd
import os

from .._skiff_external_sources import link_to_list, GO
from .cargo_data import go_file_name, data_path


class Cargo(object):
    """long loading and memory-intensive object management"""

    go = {}
    gene_lists = {}

    def get_species_cargo(self, species):
        self.get_go(species)
        self.get_lists(species)
        return self

    def get_go(self, species):
        try:
            assert(species == "hg19")
        except AssertionError:
            raise NotImplementedError("only hg19 is allowed at this point")

        if species in self.cargo.keys():
            return self.cargo[species]
        else:
            GOFile = os.path.join(data_path, species, go_file_name)
            sys.stderr.write("importing GO...")
            self.go[species] = GO(GOFile)
            sys.stderr.write("done.\n")


    def get_lists(self, species):

        try:
            assert(species == "hg19")
        except AssertionError:
            raise NotImplementedError("only hg19 is allowed at this point")

        try:
            rbps = pd.read_pickle(os.path.join(data_path, species, "rbps.df"))
            confident_rbps = pd.read_pickle(os.path.join(data_path, species, "confident_rpbs.df"))
            splicing_genes = pd.read_pickle(os.path.join(data_path, species, "splicing_genes.df"))
            tfs = pd.read_pickle(os.path.join(data_path, species, "tfs.df"))


        except:

            sys.stderr.write("rebuilding gene list objects from text and databases...\n")

            rbps_file = os.path.join(data_path, "rbps_list")
            confident_rbp_file = os.path.join(data_path, "all_pfam_defined_rbps_uniq.txt")

            rbps = pd.read_table(rbps_file).set_index("Ensembl_ID")
            rbps.to_pickle(os.path.join(data_path, species, "rbps.df"))

            with open(confident_rbp_file, 'r') as f:
                confident_rbps = set(map(str.strip, f.readlines()))
            rbps = rbps.ix[pd.Series(rbps.index.unique()).dropna()]

            confident_rbps = rbps.select(lambda x: rbps.GeneSymbol[x] in confident_rbps, 0)

            confident_rbps.to_pickle(os.path.join(data_path, species, "confident_rpbs.df"))

            splicing_genes = set(self.go[species].GO['GO:0008380']['genes']) | \
                             set(self.go[species].GO['GO:0000381']['genes']) | \
                             set(self.go[species].GO['GO:0006397']['genes'])
            splicing_genes = rbps.select(lambda x: x in splicing_genes)

            splicing_genes.to_pickle(os.path.join(data_path, species, "splicing_genes.df"))

            tfs = link_to_list("http://www.bioguo.org/AnimalTFDB/download/gene_list_of_Homo_sapiens.txt")

            with open(os.path.join(data_path, species, "gene_list_of_Homo_sapiens.txt"), 'r') as f:
                xx = f.readlines()
                tfs = pd.Series(map(lambda x: go.geneNames(x.strip()), xx), index= map(str.strip, xx))
            tfs.to_pickle(os.path.join(data_path, species, "tfs.df"))

        self.gene_lists = dict([('confident_rbps', confident_rbps),
                           ('rbps', rbps),
                           ('splicing_genes', splicing_genes),
                           ('tfs', tfs)
        ])



