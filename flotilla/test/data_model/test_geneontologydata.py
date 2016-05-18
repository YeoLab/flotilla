from collections import defaultdict

import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pytest
from scipy.stats import hypergeom


class TestGeneOntologyData(object):

    @pytest.fixture
    def gene_ontology(self, gene_ontology_data):
        from flotilla import GeneOntologyData

        return GeneOntologyData(gene_ontology_data)

    def test_init(self, gene_ontology_data, gene_ontology):
        true_data = gene_ontology_data.dropna()
        true_all_genes = true_data['Ensembl Gene ID'].unique()
        true_ontology = defaultdict(dict)

        for go, df in true_data.groupby('GO Term Accession'):
            true_ontology[go]['genes'] = set(df['Ensembl Gene ID'])
            true_ontology[go]['name'] = df['GO Term Name'].values[0]
            true_ontology[go]['domain'] = df['GO domain'].values[0]
            true_ontology[go]['n_genes'] = len(true_ontology[go]['genes'])

        pdt.assert_frame_equal(true_data, gene_ontology.data)
        pdt.assert_numpy_array_equal(sorted(true_all_genes),
                                     sorted(gene_ontology.all_genes))

        pdt.assert_contains_all(true_ontology.keys(), gene_ontology.ontology)
        pdt.assert_contains_all(gene_ontology.ontology.keys(), true_ontology)

        for go, true_attributes in true_ontology.items():
            test_attributes = gene_ontology.ontology[go]
            true_genes = sorted(true_attributes['genes'])
            test_genes = sorted(test_attributes['genes'])
            pdt.assert_numpy_array_equal(true_genes, test_genes)
            pdt.assert_equal(true_attributes['name'], test_attributes['name'])
            pdt.assert_equal(true_attributes['domain'],
                             test_attributes['domain'])
            pdt.assert_equal(true_attributes['n_genes'],
                             test_attributes['n_genes'])

    def test_enrichment(self, gene_ontology):
        features_of_interest = gene_ontology.all_genes[:10]
        test_enrichment_df = gene_ontology.enrichment(features_of_interest)

        p_value_cutoff = 1000000
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

            # TODO D.R.Y. this
            # Survival function is more accurate on small p-values
            log_p_value = hypergeom.logsf(len(features_in_go), n_all_genes,
                                          len(background_in_go),
                                          n_features_of_interest)
            # p_value = 0 if p_value < 0 else p_value
            symbols = [cross_reference[f] if f in cross_reference else f for f
                       in features_in_go]
            enrichment['negative_log_p_value'][go_term] = -log_p_value
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

        # TODO D.R.Y. this
        # Bonferonni correction
        enrichment_df['bonferonni_corrected_negative_log_p_value'] = \
            enrichment_df['negative_log_p_value'] \
            - np.log(enrichment_df.shape[0])
        ind = enrichment_df['bonferonni_corrected_negative_log_p_value'
                            ] < np.log(p_value_cutoff)
        enrichment_df = enrichment_df.ix[ind]
        true_enrichment_df = enrichment_df.sort(
            columns=['negative_log_p_value'], ascending=False)

        pdt.assert_frame_equal(test_enrichment_df, true_enrichment_df)

    @pytest.mark.xfail
    def test_invalid_background(self, gene_ontology):
        features_of_interest = gene_ontology.all_genes[:10]
        background = [f + '_asdf' for f in features_of_interest]
        gene_ontology.enrichment(features_of_interest,
                                 background=background)

    @pytest.mark.xfail
    def test_invalid_features(self, gene_ontology):
        features_of_interest = gene_ontology.all_genes[:10]
        features_of_interest = [f + '_asdf' for f in features_of_interest]
        background = [f + '_asdf' for f in gene_ontology.all_genes[:20]]
        gene_ontology.enrichment(features_of_interest,
                                 background=background)

    @pytest.mark.xfail
    def test_invalid_domain_str(self, gene_ontology):
        features_of_interest = gene_ontology.all_genes[:10]
        gene_ontology.enrichment(features_of_interest,
                                 domain='fake_domain')

    @pytest.mark.xfail
    def test_invalid_domain_iterable(self, gene_ontology):
        features_of_interest = gene_ontology.all_genes[:10]
        gene_ontology.enrichment(features_of_interest,
                                 domain=['fake_domain1', 'fake_domain2'])

    def test_custom_domain_str(self, gene_ontology):
        features_of_interest = gene_ontology.all_genes[:10]

        domain = 'cellular_component'

        test_enrichment_df = gene_ontology.enrichment(features_of_interest,
                                                      domain=domain)
        domains = frozenset([domain])
        p_value_cutoff = 1000000
        min_feature_size = 3
        min_background_size = 5
        cross_reference = {}
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

            # TODO D.R.Y. this
            # Survival function is more accurate on small p-values
            log_p_value = hypergeom.logsf(len(features_in_go), n_all_genes,
                                          len(background_in_go),
                                          n_features_of_interest)
            # p_value = 0 if p_value < 0 else p_value
            symbols = [cross_reference[f] if f in cross_reference else f for f
                       in features_in_go]
            enrichment['negative_log_p_value'][go_term] = -log_p_value
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

        # TODO D.R.Y. this
        # Bonferonni correction
        enrichment_df['bonferonni_corrected_negative_log_p_value'] = \
            enrichment_df['negative_log_p_value'] \
            - np.log(enrichment_df.shape[0])
        ind = enrichment_df['bonferonni_corrected_negative_log_p_value'
                            ] < np.log(p_value_cutoff)
        enrichment_df = enrichment_df.ix[ind]

        true_enrichment_df = enrichment_df.sort(
            columns=['negative_log_p_value'], ascending=False)

        pdt.assert_frame_equal(test_enrichment_df, true_enrichment_df)

    def test_custom_domain_iterable(self, gene_ontology):
        features_of_interest = gene_ontology.all_genes[:10]

        domain = ['cellular_component', 'molecular_function']

        test_enrichment_df = gene_ontology.enrichment(features_of_interest,
                                                      domain=domain)

        domains = frozenset(domain)
        p_value_cutoff = 1000000
        min_feature_size = 3
        min_background_size = 5
        cross_reference = {}
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

            # TODO D.R.Y this
            # Survival function is more accurate on small p-values
            log_p_value = hypergeom.logsf(len(features_in_go), n_all_genes,
                                          len(background_in_go),
                                          n_features_of_interest)
            # p_value = 0 if p_value < 0 else p_value
            symbols = [cross_reference[f] if f in cross_reference else f for f
                       in features_in_go]
            enrichment['negative_log_p_value'][go_term] = - log_p_value
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

        # TODO D.R.Y. this
        # Bonferonni correction
        enrichment_df['bonferonni_corrected_negative_log_p_value'] = \
            enrichment_df['negative_log_p_value'] \
            - np.log(enrichment_df.shape[0])
        ind = enrichment_df['bonferonni_corrected_negative_log_p_value'
                            ] < np.log(p_value_cutoff)
        enrichment_df = enrichment_df.ix[ind]
        true_enrichment_df = enrichment_df.sort(
            columns=['negative_log_p_value'], ascending=False)

        pdt.assert_frame_equal(test_enrichment_df, true_enrichment_df)

    def test_too_few_features(self, gene_ontology):
        features_of_interest = gene_ontology.all_genes[:3]
        test_enrichment_df = gene_ontology.enrichment(features_of_interest)

        domains = gene_ontology.domains
        p_value_cutoff = 1000000
        min_feature_size = 3
        min_background_size = 5
        cross_reference = {}
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

            # TODO D.R.Y. this
            # Survival function is more accurate on small p-values
            log_p_value = hypergeom.logsf(len(features_in_go), n_all_genes,
                                          len(background_in_go),
                                          n_features_of_interest)
            # p_value = 0 if p_value < 0 else p_value
            symbols = [cross_reference[f] if f in cross_reference else f for f
                       in features_in_go]
            enrichment['negative_log_p_value'][go_term] = -log_p_value
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

        # TODO D.R.Y. this
        # Bonferonni correction
        enrichment_df['bonferonni_corrected_negative_log_p_value'] = \
            enrichment_df['negative_log_p_value'] \
            - np.log(enrichment_df.shape[0])
        ind = enrichment_df['bonferonni_corrected_negative_log_p_value'
                            ] < np.log(p_value_cutoff)
        enrichment_df = enrichment_df.ix[ind]
        true_enrichment_df = enrichment_df.sort(
            columns=['negative_log_p_value'], ascending=False)

        pdt.assert_frame_equal(test_enrichment_df, true_enrichment_df)

    def test_no_enrichment(self, gene_ontology):
        features_of_interest = gene_ontology.all_genes[:2]
        test_enrichment_df = gene_ontology.enrichment(features_of_interest)

        domains = gene_ontology.domains
        min_feature_size = 3
        min_background_size = 5
        cross_reference = {}
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
        true_enrichment_df = pd.DataFrame(enrichment)

        assert true_enrichment_df.empty
        assert test_enrichment_df is None
