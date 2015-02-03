""" interface with external data sources i.e. GO files, web"""
from __future__ import division

from collections import defaultdict
import gzip

import numpy as np
import pandas as pd

from flotilla.util import link_to_list


class GO(object):
    """
    gene ontology tool

    >>> go = hg19GO()
    >>> go.geneXref['ENSG00000100320']
    'RBFOX2'
    >>> data = go.enrichment(list, background)

    """

    def __init__(self, GOFile):
        with gzip.open(GOFile) as file_handle:
            GO_to_ENSG = pd.read_table(file_handle)
        geneXref = defaultdict()
        for k in np.array(
                GO_to_ENSG.get(["Ensembl Gene ID", "Associated Gene Name"])):
            ensg = k[0]
            gene = k[1]
            geneXref[ensg] = gene

        GO, allGenes = generateOntology(GO_to_ENSG)
        self.GO = GO
        self.allGenes = allGenes
        self.geneXref = geneXref

    def enrichment(self, geneList, background=None, **kwargs):
        if background is None:
            background = self.allGenes
        return GO_enrichment(geneList, self.GO, expressedGenes=background,
                             xRef=self.geneXref)

    def geneNames(self, x):
        try:
            return self.geneXref[x]
        except:
            return x

    def link_to_geneNames(self, list_link):
        list = link_to_list(list_link)
        pd.DataFrame(map(self.geneNames, list), index=list)
