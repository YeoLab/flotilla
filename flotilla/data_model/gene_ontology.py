from collections import defaultdict, Iterable
import sys
import warnings

import pandas as pd
from scipy.stats import hypergeom

from ..util import timestamp


class GeneOntologyData(object):

    domains = frozenset(['biological_process', 'molecular_function',
                         'cellular_component'])

    def __init__(self, data):
        """Object to calculate enrichment of Gene Ontology terms

        Acceptable Gene Ontology tables can be downloaded from ENSEMBL's
        BioMart tool: http://www.ensembl.org/biomart

        1. Choose "Ensembl Genes ##" (## = version number, for me it's 78)
        2. Click "Attributes"
        3. Expand "EXTERNAL"
        4. Check the boxes for 'GO Term Accession', 'Ensembl Gene ID',
           'GO Term Name', and 'GO domain'

        Parameters
        ----------
        data : pandas.DataFrame
            A dataframe with at least the following columns:
            'GO Term Accession', 'Ensembl Gene ID', 'GO Term Name', 'GO domain'
        """

        self.data = data.dropna()

        # Need "data_original" to be consistent with other datatypes
        self.data_original = self.data
        sys.stdout.write('{}\tBuilding Gene Ontology '
                         'database...\n'.format(timestamp()))
        self.ontology = defaultdict(dict)
        for go, df in data.groupby('GO Term Accession'):
            self.ontology[go]['genes'] = set(df['Ensembl Gene ID'])
            self.ontology[go]['name'] = df['GO Term Name'].values[0]
            self.ontology[go]['domain'] = df['GO domain'].values[0]
            self.ontology[go]['n_genes'] = len(self.ontology[go]['genes'])
        sys.stdout.write('{}\t\tDone.\n'.format(timestamp()))

        self.all_genes = self.data['Ensembl Gene ID'].unique()

    def enrichment(self, features_of_interest, background=None,
                   p_value_cutoff=1000000, cross_reference=None,
                   min_feature_size=3, min_background_size=5,
                   domain=None):
        """Bonferroni-corrected hypergeometric p-values of GO enrichment

        Calculates hypergeometric enrichment of the features of interest, in
        each GO category.

        Parameters
        ----------
        features_of_interest : list-like
            List of features. Must match the identifiers in the ontology
            database exactly, i.e. if your ontology database is ENSEMBL ids,
            then you can only provide those and not common names like "RBFOX2"
        background : list-like, optional
            Background genes to use. It is best to use a relevant background
            such as all expressed genes. If None, defaults to all genes.
        p_value_cutoff : float, optional
            Maximum accepted Bonferroni-corrected p-value
        cross_reference : dict-like, optional
            A mapping of gene ids to gene symbols, e.g. a pandas Series of
            ENSEMBL genes e.g. ENSG00000139675 to gene symbols e.g HNRNPA1L2
        min_feature_size : int, optional
            Minimum number of features of interest overlapping in a GO Term,
            to calculate enrichment
        min_background_size : int, optional
            Minimum number of features in the background overlapping a GO Term
        domain : str or list, optional
            Only calculate GO enrichment for a particular GO category or
            subset of categories. Valid domains:
            'biological_process', 'molecular_function', 'cellular_component'

        Returns
        -------
        enrichment_df : pandas.DataFrame
            A (n_go_categories, columns) DataFrame of the enrichment scores

        Raises
        ------
        ValueError
            If features of interest and background do not overlap, or invalid
            GO domains are given
        """
        cross_reference = {} if cross_reference is None else cross_reference
        background = self.all_genes if background is None else background
        if len(set(background) & set(features_of_interest)) == 0:
            raise ValueError('Features of interest and background do not '
                             'overlap! Not calculating GO enrichment')
        if len(set(features_of_interest) & set(self.all_genes)) == 0:
            raise ValueError('Features of interest do not overlap with GO term'
                             'gene ids. Not calculating GO enrichment.')
        domains = self.domains
        valid_domains = ",".join("'{}'".format(x) for x in self.domains)

        if isinstance(domain, str):
            if domain not in self.domains:
                raise ValueError(
                    "'{}' is not a valid GO domain. "
                    "Only {} are acceptable".format(domain, valid_domains))
            domains = frozenset([domain])
        elif isinstance(domain, Iterable):
            if len(set(domain) & self.domains) == 0:
                raise ValueError(
                    "'{}' are not a valid GO domains. "
                    "Only {} are acceptable".format(
                        ",".join("'{}'".format(x) for x in domain),
                        valid_domains))
            domains = frozenset(domain)

        n_all_genes = len(background)
        n_features_of_interest = len(features_of_interest)
        enrichment = defaultdict(dict)

        for go_term, go_genes in self.ontology.items():
            if go_genes['domain'] not in domains:
                continue

            features_in_go = go_genes['genes'].intersection(
                features_of_interest)
            background_in_go = go_genes['genes'].intersection(background)
            too_few_features = len(features_in_go) < min_feature_size
            too_few_background = len(background_in_go) < min_background_size
            if too_few_features or too_few_background:
                continue

            # Survival function is more accurate on small p-values
            p_value = hypergeom.sf(len(features_in_go), n_all_genes,
                                   len(background_in_go),
                                   n_features_of_interest)
            p_value = 0 if p_value < 0 else p_value
            symbols = [cross_reference[f] if f in cross_reference else f for f
                       in features_in_go]
            enrichment['p_value'][go_term] = p_value
            enrichment['n_features_of_interest_in_go_term'][go_term] = len(
                features_in_go)
            enrichment['n_background_in_go_term'][go_term] = len(
                background_in_go)
            enrichment['n_features_total_in_go_term'][go_term] = len(
                go_genes['genes'])
            enrichment['features_of_interest_in_go_term'][
                go_term] = ','.join(features_in_go)
            enrichment['features_of_interest_in_go_term_gene_symbols'][
                go_term] = ','.join(symbols)
            enrichment['go_domain'][go_term] = go_genes['domain']
            enrichment['go_name'][go_term] = go_genes['name']
        enrichment_df = pd.DataFrame(enrichment)

        if enrichment_df.empty:
            warnings.warn('No GO categories enriched in provided features')
            return

        # Bonferonni correction
        enrichment_df['bonferonni_corrected_p_value'] = \
            enrichment_df.p_value * enrichment_df.shape[0]
        ind = enrichment_df['bonferonni_corrected_p_value'] < p_value_cutoff
        enrichment_df = enrichment_df.ix[ind]
        enrichment_df = enrichment_df.sort(columns=['p_value'])

        return enrichment_df
