""" interface with external data sources i.e. GO files, web"""
from __future__ import division

from collections import defaultdict
import gzip

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from flotilla.util import link_to_list


def generateOntology(df):
    from collections import defaultdict
    import itertools

    ontology = defaultdict(
        lambda: {'genes': set(), 'name': set(), 'domain': set()})
    allGenesInOntologies = set(df.get('Ensembl Gene ID'))
    for GO, gene, domain, name in itertools.izip(df.get('GO Term Accession'),
                                                 df.get('Ensembl Gene ID'),
                                                 df.get('GO domain'),
                                                 df.get('GO Term Name')):
        ontology[GO]['genes'].add(gene)
        ontology[GO]['name'].add(name)
        ontology[GO]['domain'].add(domain)
        ontology[GO]['n_genes'] = len(ontology[GO]['genes'])
    return ontology, allGenesInOntologies


def GO_enrichment(geneList, ontology, expressedGenes=None, printIt=False,
                  pCut=1000000, xRef={}):
    lenAllGenes, lenTheseGenes = len(expressedGenes), len(geneList)
    pValues = defaultdict()
    nCmps = 0

    for GOTerm, GOGenes in ontology.items():
        inBoth = GOGenes['genes'].intersection(geneList)
        expressedGOGenes = GOGenes['genes'].intersection(expressedGenes)
        if len(inBoth) <= 3 or len(expressedGOGenes) < 5:
            pValues[GOTerm] = 'notest'
            continue
        # survival function is more accurate on small p-values...
        pVal = hypergeom.sf(len(inBoth), lenAllGenes, len(expressedGOGenes),
                            lenTheseGenes)
        if pVal < 0:
            pVal = 0
        symbols = []
        for ensg in inBoth:
            if ensg in xRef:
                symbols.append(xRef[ensg])
            else:
                symbols.append(ensg)
        pValues[GOTerm] = (
            pVal, len(inBoth), len(expressedGOGenes), len(GOGenes['genes']),
            inBoth,
            symbols)

    for k, v in pValues.items():
        try:
            pValues[k][0] = v * float(nCmps)  # bonferroni correction
        except:
            pass
    import operator

    y = []

    sorted_x = sorted(pValues.iteritems(), key=operator.itemgetter(1))

    for k, v in sorted_x:
        if v == "notest":
            continue
        if not type(k) == str:
            continue
        try:
            if v[0] > pCut:
                continue
            if printIt:
                print k, "|".join(ontology[k]['name']), "%.3e" % v[0], v[1], \
                    v[2], v[3], "|".join(v[3])
                pass
            y.append([k, "|".join(ontology[k]['name']), v[0], v[1], v[2], v[3],
                      ",".join(v[4]), ",".join(v[5])])

        except:
            pass

    try:
        df = pd.DataFrame(y, columns=['GO Term ID', 'GO Term Description',
                                      'Bonferroni-corrected Hypergeometric '
                                      'p-Value',
                                      'N Genes in List and GO Category',
                                      'N Expressed Genes in GO Category',
                                      'N Genes in GO category',
                                      'Ensembl Gene IDs in List',
                                      'Gene symbols in List'])
        df.set_index('GO Term ID', inplace=True)
    except:
        df = pd.DataFrame(None, columns=['GO Term ID', 'GO Term Description',
                                         'Bonferroni-corrected Hypergeometric '
                                         'p-Value',
                                         'N Genes in List and GO Category',
                                         'N Expressed Genes in GO Category',
                                         'N Genes in GO category',
                                         'Ensembl Gene IDs in List',
                                         'Gene symbols in List'])

    return df


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
