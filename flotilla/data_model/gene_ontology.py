from collections import defaultdict
from itertools import izip
import operator
import sys

import pandas as pd
from scipy.stats import hypergeom

from ..util import timestamp


class GeneOntologyData(object):

    def __init__(self, data):

        self.data = data
        sys.stdout.write('{}\tBuilding Gene Ontology '
                         'database...\n'.format(timestamp()))
        ontology, all_genes = self._generate_ontology(self.data)
        sys.stdout.write('{}\t\tDone.'.format(timestamp()))
        self.ontology = ontology
        self.all_genes = all_genes

    def _generate_ontology(self, df):

        ontology = defaultdict(
            lambda: {'genes': set(), 'name': set(), 'domain': set()})
        allGenesInOntologies = set(df.get('Ensembl Gene ID'))
        for GO, gene, domain, name in izip(
                df.get('GO Term Accession'),
                df.get('Ensembl Gene ID'),
                df.get('GO domain'),
                df.get('GO Term Name')):
            ontology[GO]['genes'].add(gene)
            ontology[GO]['name'].add(name)
            ontology[GO]['domain'].add(domain)
            ontology[GO]['n_genes'] = len(ontology[GO]['genes'])
        return ontology, allGenesInOntologies


    def enrichment(self, features_of_interest, background=None,
                   p_value_cutoff=1000000, xRef={}):
        background = self.all_genes if background is None else background
        n_all_genes = len(background)
        n_features_of_interest = len(features_of_interest)
        pValues = defaultdict()
        n_comparisons = 0

        for go_term, go_genes in self.ontology.items():
            feature_ids_go = go_genes['genes'].intersection(
                features_of_interest)
            background_go = go_genes['genes'].intersection(background)
            if len(feature_ids_go) <= 3 or len(background_go) < 5:
                pValues[go_term] = 'notest'
                continue

            # Survival function is more accurate on small p-values
            pVal = hypergeom.sf(len(feature_ids_go), n_all_genes,
                                len(background_go),
                                n_features_of_interest)
            if pVal < 0:
                pVal = 0
            symbols = []
            for ensg in feature_ids_go:
                if ensg in xRef:
                    symbols.append(xRef[ensg])
                else:
                    symbols.append(ensg)
            pValues[go_term] = (
                pVal, len(feature_ids_go), len(background_go),
                len(go_genes['genes']),
                feature_ids_go,
                symbols)

        for k, v in pValues.items():
            try:
                # Bonferroni correction
                pValues[k][0] = v * float(n_comparisons)
            except TypeError:
                pass

        y = []

        sorted_x = sorted(pValues.iteritems(),
                          key=operator.itemgetter(1))

        for k, v in sorted_x:
            if v == "notest":
                continue
            if not type(k) == str:
                continue
            try:
                if v[0] > p_value_cutoff:
                    continue
                y.append(
                    [k, "|".join(self.ontology[k]['name']), v[0], v[1],
                     v[2], v[3], ",".join(v[4]), ",".join(v[5])])

            except:
                pass
        columns = ['go_term_id', 'go_term_description',
                   'bonferroni_corrected_hypergeometric_'
                   'p_value',
                   'n_genes_in_list_and_go_category',
                   'n_expressed_genes_in_go_category',
                   'n_genes in_go_category',
                   'ensembl_gene_ids_in_list',
                   'gene_symbols_in_ist']

        try:
            df = pd.DataFrame(y, columns=columns)
            df.set_index('go_term_id', inplace=True)
        except:
            df = pd.DataFrame(None, columns=columns)

        return df

    def plot_enrichment(self, feature_ids, background, ):
        pass