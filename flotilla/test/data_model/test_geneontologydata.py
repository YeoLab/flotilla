from collections import defaultdict

import pandas as pd
import pandas.util.testing as pdt
from scipy.stats import hypergeom

class TestGeneOntologyData(object):
    def test_init(self, gene_ontology_data):
        from flotilla import GeneOntologyData

        gene_ontology = GeneOntologyData(gene_ontology_data)

        true_data = gene_ontology_data.dropna()
        true_all_genes = gene_ontology_data['Ensembl Gene ID'].unique()
        true_ontology = defaultdict(dict)
        for go, df in true_data.groupby('GO Term Accession'):
            true_ontology[go]['genes'] = set(df['Ensembl Gene ID'])
            true_ontology[go]['name'] = df['GO Term Name'].values[0]
            true_ontology[go]['domain'] = df['GO domain'].values[0]
            true_ontology[go]['n_genes'] = len(true_ontology[go]['genes'])

        pdt.assert_frame_equal(true_data, gene_ontology.data)
        pdt.assert_array_equal(true_all_genes, gene_ontology.all_genes)
        pdt.assert_dict_equal(true_ontology, true_data)

    def test_enrichment(self, gene_ontology_data):
        from flotilla import GeneOntologyData

        gene_ontology = GeneOntologyData(gene_ontology_data)

        features_of_interest = gene_ontology.all_genes[:10]
        test_enrichment_df = gene_ontology.enrichment(features_of_interest)

        min_feature_size = 3
        min_background_size = 5
        cross_reference = {}
        domains = gene_ontology.domains
        background = gene_ontology.all_genes
        n_all_genes = len(background)
        n_features_of_interest = len(features_of_interest)
        enrichment = defaultdict(dict)

        for go_term, go_genes in gene_ontology.ontology.items():
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


        # Bonferonni correction
        enrichment_df['bonferonni_corrected_p_value'] = \
            enrichment_df.p_value * enrichment_df.shape[0]
        ind = enrichment_df['bonferonni_corrected_p_value'] < p_value_cutoff
        enrichment_df = enrichment_df.ix[ind]
        true_enrichment_df = enrichment_df.sort(columns=['p_value'])

        pdt.assert_frame_equal(test_enrichment_df, true_enrichment_df)
