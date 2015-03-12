"""
This tests whether the Study object was created correctly. No
computation or visualization tests yet.
"""
from collections import Iterable
import json

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import pytest
import semantic_version


@pytest.fixture(params=['expression', 'splicing'])
def data_type(request):
    return request.param


@pytest.fixture(params=[None, 'subset1'],
                ids=['color_samples_by_none', 'color_samples_by_subset1'])
def color_samples_by(request, metadata_phenotype_col):
    if request.param == 'phenotype':
        return metadata_phenotype_col
    else:
        return request.param


class TestStudy(object):
    # @pytest.fixture
    # def n_groups(self):
    #     return 3

    @pytest.fixture
    def study(self, metadata_data, metadata_kws_fixed,
              mapping_stats_data, mapping_stats_kws,
              expression_data, expression_kws,
              splicing_data, splicing_kws,
              gene_ontology_data):
        from flotilla.data_model import Study

        kwargs = {}
        metadata = metadata_data.copy()
        splicing = splicing_data.copy()
        expression = expression_data.copy()
        mapping_stats = mapping_stats_data.copy() 
        
        kw_pairs = (('metadata', metadata_kws_fixed),
                    ('mapping_stats', mapping_stats_kws),
                    ('expression', expression_kws),
                    ('splicing', splicing_kws))
        for data_type, kws in kw_pairs:
            for kw_name, kw_value in kws.iteritems():
                kwargs['{}_{}'.format(data_type, kw_name)] = kw_value

        return Study(metadata_data,
                     mapping_stats_data=mapping_stats_data,
                     expression_data=expression_data,
                     splicing_data=splicing_data,
                     gene_ontology_data=gene_ontology_data, **kwargs)
    def test_init(self, metadata_data):
        from flotilla import Study

        metadata = metadata_data.copy()
        study = Study(metadata)

        metadata['outlier'] = False

        true_default_sample_subsets = list(sorted(list(set(
            study.metadata.sample_subsets.keys()).difference(
            set(study.default_sample_subset)))))
        true_default_sample_subsets.insert(0, study.default_sample_subset)

        pdt.assert_frame_equal(study.metadata.data, metadata)
        pdt.assert_equal(study.version, '0.1.0')
        pdt.assert_equal(study.pooled, None)
        pdt.assert_equal(study.technical_outliers, None)
        pdt.assert_equal(study.phenotype_col, study.metadata.phenotype_col)
        pdt.assert_equal(study.phenotype_order, study.metadata.phenotype_order)
        pdt.assert_equal(study.phenotype_to_color,
                         study.metadata.phenotype_to_color)
        pdt.assert_equal(study.phenotype_to_marker,
                         study.metadata.phenotype_to_marker)
        pdt.assert_series_equal(study.sample_id_to_phenotype,
                                study.metadata.sample_id_to_phenotype)
        pdt.assert_series_equal(study.sample_id_to_color,
                                study.metadata.sample_id_to_color)
        pdt.assert_array_equal(study.phenotype_transitions,
                               study.metadata.phenotype_transitions)
        pdt.assert_array_equal(study.phenotype_color_ordered,
                               study.metadata.phenotype_color_order)
        pdt.assert_equal(study.default_sample_subset, 'all_samples')
        pdt.assert_equal(study.default_feature_subset, 'variant')
        pdt.assert_array_equal(study.default_sample_subsets,
                               true_default_sample_subsets)
        pdt.assert_dict_equal(study.default_feature_subsets, {})

    @pytest.mark.xfail
    def test_setattr(self, metadata_data):
        # warnings.simplefilter("error")

        from flotilla import Study

        study = Study(metadata_data.copy())

        study.pooled = 'asdf'
        # warnings.simplefilter('default')

    def test_init_metdadata_kws(self, metadata_data, metadata_kws):
        # Also need to check for when these are NAs
        from flotilla import Study

        kws = dict(('metadata_'+k, v) for k, v in metadata_kws.items())
        study = Study(metadata_data, **kws)

        pdt.assert_frame_equal(study.metadata.data,
                               metadata_data)
        pdt.assert_equal(study.version, '0.1.0')
        npt.assert_equal(study.pooled, None)
        # npt.assert_equal(study.outliers, None)

    def test_init_pooled(self, metadata_data,
                         metadata_kws,
                         pooled):
        from flotilla import Study
        metadata = metadata_data.copy()

        kws = dict(('metadata_'+k, v) for k, v in metadata_kws.items())
        metadata['pooled'] = metadata.index.isin(pooled)

        study = Study(metadata, **kws)

        npt.assert_array_equal(sorted(study.pooled), sorted(pooled))

    def test_init_bad_pooled(self, metadata_data, metadata_kws, pooled):
        from flotilla import Study

        metadata = metadata_data.copy()

        kws = dict(('metadata_' + k, v) for k, v in metadata_kws.items())
        metadata['pooled_asdf'] = metadata.index.isin(pooled)

        study = Study(metadata, **kws)

        true_pooled = None
        if study.metadata.pooled_col is not None:
            if study.metadata.pooled_col in study.metadata.data:
                try:
                    true_pooled = study.metadata.data.index[
                        study.metadata.data[
                            study.metadata.pooled_col].astype(bool)]
                except KeyError:
                    true_pooled = None

        npt.assert_equal(study.pooled, true_pooled)

    def test_init_outlier(self, metadata_data, metadata_kws, outliers):
        from flotilla import Study

        metadata = metadata_data.copy()

        kws = dict(('metadata_' + k, v) for k, v in metadata_kws.items())
        metadata['outlier'] = metadata.index.isin(outliers)

        study = Study(metadata, **kws)

        npt.assert_array_equal(study.metadata.data, metadata)

    def test_init_technical_outlier(self, metadata_data, metadata_kws,
                                    technical_outliers, mapping_stats_data,
                                    mapping_stats_kws):
        from flotilla import Study

        metadata = metadata_data.copy()

        kw_pairs = (('metadata', metadata_kws),
                    ('mapping_stats', mapping_stats_kws))
        kwargs = {}
        for name, kws in kw_pairs:
            for k, v in kws.items():
                kwargs['{}_{}'.format(name, k)] = v
        study = Study(metadata, mapping_stats_data=mapping_stats_data,
                      **kwargs)
        pdt.assert_array_equal(sorted(study.technical_outliers),
                               sorted(technical_outliers))

    def test_init_expression(self, metadata_data, metadata_kws,
                             expression_data, expression_kws):
        from flotilla import Study

        metadata = metadata_data.copy()
        expression = expression_data.copy()

        kw_pairs = (('metadata', metadata_kws),
                    ('expression', expression_kws))
        kwargs = {}
        for name, kws in kw_pairs:
            for k, v in kws.items():
                kwargs['{}_{}'.format(name, k)] = v
        study = Study(metadata, expression_data=expression,
                      **kwargs)
        pdt.assert_array_equal(study.expression.data_original,
                               expression_data)

    def test_init_splicing(self, metadata_data, metadata_kws,
                           splicing_data, splicing_kws):
        from flotilla import Study

        metadata = metadata_data.copy()
        splicing = splicing_data.copy()

        kw_pairs = (('metadata', metadata_kws),
                    ('splicing', splicing_kws))
        kwargs = {}
        for name, kws in kw_pairs:
            for k, v in kws.items():
                kwargs['{}_{}'.format(name, k)] = v
        study = Study(metadata, splicing_data=splicing,
                      **kwargs)
        pdt.assert_array_equal(study.splicing.data_original,
                               splicing_data)

    def test_feature_subset_to_feature_ids(self, study, data_type,
                                           feature_subset):
        test_feature_subset = study.feature_subset_to_feature_ids(
            data_type, feature_subset)
        if 'expression'.startswith(data_type):
            true_feature_subset = \
                study.expression.feature_subset_to_feature_ids(feature_subset,
                                                               rename=False)
        elif 'splicing'.startswith(data_type):
            true_feature_subset = study.splicing.feature_subset_to_feature_ids(
                feature_subset, rename=False)
        pdt.assert_array_equal(test_feature_subset, true_feature_subset)

    def test_sample_subset_to_sample_ids(self, study, sample_subset):
        test_sample_subset = study.sample_subset_to_sample_ids(sample_subset)

        try:
            true_sample_subset = study.metadata.sample_subsets[sample_subset]
        except (KeyError, TypeError):
            try:
                ind = study.metadata.sample_id_to_phenotype == sample_subset
                if ind.sum() > 0:
                    true_sample_subset = \
                        study.metadata.sample_id_to_phenotype.index[ind]
                else:
                    if sample_subset is None or 'all_samples'.startswith(
                            sample_subset):
                        sample_ind = np.ones(study.metadata.data.shape[0],
                                             dtype=bool)
                    elif sample_subset.startswith("~"):
                        sample_ind = ~pd.Series(
                            study.metadata.data[sample_subset.lstrip("~")],
                            dtype='bool')

                    else:
                        sample_ind = pd.Series(
                            study.metadata.data[sample_subset], dtype='bool')
                    true_sample_subset = study.metadata.data.index[sample_ind]
            except (AttributeError, ValueError):
                true_sample_subset = sample_subset

        pdt.assert_array_equal(true_sample_subset, test_sample_subset)

    def test_filter_splicing_on_expression(self, study):
        expression_thresh = 5
        sample_subset = None
        test_filtered_splicing = study.filter_splicing_on_expression(
            expression_thresh)
        columns = study._maybe_get_axis_name(study.splicing.data, axis=1,
                                            alt_name=study._event_name)

        index = study._maybe_get_axis_name(study.splicing.data, axis=0,
                                          alt_name=study._sample_id)
    
        sample_ids = study.sample_subset_to_sample_ids(sample_subset)
        splicing_with_expression = \
            study.tidy_splicing_with_expression.ix[
                study.tidy_splicing_with_expression.sample_id.isin(
                    sample_ids)]
        ind = splicing_with_expression.expression >= expression_thresh
        splicing_high_expression = splicing_with_expression.ix[ind]
        splicing_high_expression = \
            splicing_high_expression.reset_index().dropna()
    
        if isinstance(columns, list) or isinstance(index, list):
            true_filtered_splicing = splicing_high_expression.pivot_table(
                columns=columns, index=index, values='psi')
        else:
            true_filtered_splicing = splicing_high_expression.pivot(
                columns=columns, index=index, values='psi')
        pdt.assert_frame_equal(true_filtered_splicing, test_filtered_splicing)
        

    def test_plot_pca(self, study, data_type):
        study.plot_pca(feature_subset='all', data_type=data_type)
        plt.close('all')

    # Too few features to test graph or classifier
    # def test_plot_graph(self, study, data_type):
    #     study.plot_graph(feature_subset='all', data_type=data_type)
    #     plt.close('all')

    # def test_plot_classifier(self, study, data_type):
    #     trait = study.metadata.phenotype_col
    #     study.plot_classifier(trait, feature_subset='all', data_type=data_type)
    #     plt.close('all')

    def test_plot_clustermap(self, study, data_type):
        study.plot_clustermap(feature_subset='all', data_type=data_type)
        plt.close('all')

    def test_plot_correlations(self, study, featurewise, data_type):
        study.plot_correlations(feature_subset='all', featurewise=featurewise,
                                data_type=data_type)
        plt.close('all')

    def test_plot_lavalamps(self, study):
        study.plot_lavalamps()
        plt.close('all')

    def test_plot_big_nmf_space_transitions(self, study):
        study.plot_big_nmf_space_transitions('splicing')
        plt.close('all')

    def test_plot_two_samples(self, study, data_type):
        sample1 = study.expression.data.index[0]
        sample2 = study.expression.data.index[-1]
        study.plot_two_samples(sample1, sample2, data_type=data_type)

    def test_plot_two_features(self, study, data_type):
        if data_type == 'expression':
            feature1 = study.expression.data.columns[0]
            feature2 = study.expression.data.columns[-1]
        elif data_type == 'splicing':
            feature1 = study.splicing.data.columns[0]
            feature2 = study.splicing.data.columns[-1]
        study.plot_two_features(feature1, feature2, data_type=data_type)

    @pytest.fixture(params=[None, 'gene'])
    def gene_of_interest(self, request, genes):
        if request is not None:
            return genes[0]
        else:
            return request.param




    # def test_plot_graph(self, study, gene_of_interest, featurewise):
    #     study.plot_graph(feature_of_interest=gene_of_interest,
    #                      feature_subset='all', featurewise=featurewise)
    #     plt.close('all')
    #
    # # def test_plot_classifier(self, study):
    # #     study.plot_classifier(study.metadata.phenotype_col,
    # #                           feature_subset='all')
    # #     plt.close('all')
    # #
    # # def test_plot_classifier_splicing(self, study):
    # #     study.plot_classifier(study.metadata.phenotype_col,
    # #                           feature_subset='all',
    # #                           data_type='splicing')
    # #     plt.close('all')
    #
    # def test_plot_clustermap(self, study):
    #     study.plot_clustermap(feature_subset='all')
    #     plt.close('all')
    #
    # def test_plot_clustermap_splicing(self, study):
    #     study.plot_clustermap(feature_subset='all',
    #                           data_type='splicing')
    #     plt.close('all')
    #
    # def test_plot_correlations(self, study, featurewise):
    #     study.plot_correlations(featurewise=featurewise,
    #                             feature_subset='all')
    #     plt.close('all')
    #
    # def test_plot_correlations_splicing(self, study, featurewise):
    #     study.plot_correlations(featurewise=featurewise,
    #                             data_type='splicing',
    #                             feature_subset='all')
    #     plt.close('all')
    #
    # def test_tidy_splicing_with_expression(self, study):
    #     test = study.tidy_splicing_with_expression
    #
    #     common_id = 'common_id'
    #     sample_id = 'sample_id'
    #     event_name = 'event_name'
    #
    #     splicing_common_id = study.splicing.feature_data[
    #         study.splicing.feature_expression_id_col]
    #
    #     # Tidify splicing
    #     splicing = study.splicing.data
    #     splicing_index_name = study._maybe_get_axis_name(splicing, axis=0)
    #     splicing_columns_name = study._maybe_get_axis_name(splicing, axis=1)
    #
    #     splicing_tidy = pd.melt(splicing.reset_index(),
    #                             id_vars=splicing_index_name,
    #                             value_name='psi',
    #                             var_name=splicing_columns_name)
    #     rename_columns = {}
    #     if splicing_index_name == 'index':
    #         rename_columns[splicing_index_name] = sample_id
    #     if splicing_columns_name == 'columns':
    #         rename_columns[splicing_columns_name] = event_name
    #         splicing_columns_name = event_name
    #     splicing_tidy = splicing_tidy.rename(columns=rename_columns)
    #
    #     # Create a column of the common id on which to join splicing
    #     # and expression
    #     splicing_names = splicing_tidy[splicing_columns_name]
    #     if isinstance(splicing_names, pd.Series):
    #         splicing_tidy[common_id] = splicing_tidy[
    #             splicing_columns_name].map(splicing_common_id)
    #     else:
    #         splicing_tidy[common_id] = [
    #             study.splicing.feature_renamer(x)
    #             for x in splicing_names.itertuples(index=False)]
    #
    #     splicing_tidy = splicing_tidy.dropna()
    #
    #     # Tidify expression
    #     expression = study.expression.data_original
    #     expression_index_name = study._maybe_get_axis_name(expression,
    #                                                        axis=0)
    #     expression_columns_name = study._maybe_get_axis_name(expression,
    #                                                          axis=1)
    #
    #     expression_tidy = pd.melt(expression.reset_index(),
    #                               id_vars=expression_index_name,
    #                               value_name='expression',
    #                               var_name=common_id)
    #     # This will only do anything if there is a column named "index" so
    #     # no need to check anything
    #     expression_tidy = expression_tidy.rename(
    #         columns={'index': sample_id})
    #     expression_tidy = expression_tidy.dropna()
    #
    #     splicing_tidy.set_index([sample_id, common_id], inplace=True)
    #     expression_tidy.set_index([sample_id, common_id], inplace=True)
    #
    #     true = splicing_tidy.join(expression_tidy, how='inner').reset_index()
    #
    #     pdt.assert_frame_equal(test, true)

    #
    #
    # @pytest.fixture(params=[None, 'pooled_col', 'phenotype_col'])
    #     def metadata_none_key(self, request):
    #         return request.param
    #
    #     @pytest.fixture(params=[None])
    #     def expression_none_key(self, request):
    #         return request.param
    #
    #     @pytest.fixture(params=[None,
    #                             pytest.mark.xfail('feature_rename_col')])
    #     def splicing_none_key(self, request):
    #         return request.param
    #
    #     @pytest.fixture
    #     def datapackage(self, shalek2013_datapackage, metadata_none_key,
    #                     expression_none_key, splicing_none_key, monkeypatch):
    #         datapackage = copy.deepcopy(shalek2013_datapackage)
    #         datatype_to_key = {'metadata': metadata_none_key,
    #                            'expression': expression_none_key,
    #                            'splicing': splicing_none_key}
    #         for datatype, key in datatype_to_key.iteritems():
    #             if key is not None:
    #                 resource = name_to_resource(datapackage, datatype)
    #                 if key in resource:
    #                     monkeypatch.delitem(resource, key, raising=False)
    #         return datapackage
    #
    #     @pytest.fixture
    #     def datapackage_dir(self, shalek2013_datapackage_path):
    #         return os.path.dirname(shalek2013_datapackage_path)
    #
    #     # def test_from_datapackage(self, datapackage, datapackage_dir):
    #     #     import flotilla
    #     #
    #     #     study = flotilla.Study.from_datapackage(
    #     #         datapackage, datapackage_dir, load_species_data=False)
    #     #
    #     #     metadata_resource = get_resource_from_name(
    #     #         datapackage, 'metadata')
    #     #     expression_resource = get_resource_from_name(datapackage,
    #     #                                                  'expression')
    #     #     splicing_resource = get_resource_from_name(datapackage,
    #     #                                                'splicing')
    #     #
    #     #     phenotype_col = 'phenotype' if 'phenotype_col' \
    #     #                                    not in metadata_resource else \
    #     #     metadata_resource['phenotype_col']
    #     #     pooled_col = 'pooled' if 'pooled_col' not in
    #     #         metadata_resource else \
    #     #         metadata_resource['pooled_col']
    #     #     expression_feature_rename_col = None if \
    #     #         'feature_rename_col' not in expression_resource \
    #     #         else expression_resource['feature_rename_col']
    #     #     splicing_feature_rename_col = 'gene_name' if \
    #     #         'feature_rename_col' not in splicing_resource \
    #     #         else splicing_resource['feature_rename_col']
    #     #
    #     #     assert study.metadata.phenotype_col == phenotype_col
    #     #     assert study.metadata.pooled_col == pooled_col
    #     #     assert study.expression.feature_rename_col \
    #     #            == expression_feature_rename_col
    #     #     assert study.splicing.feature_rename_col \
    #     #            == splicing_feature_rename_col

    @staticmethod
    def get_data_eval_command(data_type, attribute):
        if 'feature' in data_type:
            # Feature data doesn't have "data_original", only "data"
            if attribute == 'data_original':
                attribute = 'data'
            command = 'study.{}.feature_{}'.format(
                data_type.split('_feature')[0], attribute)
        else:
            command = 'study.{}.{}'.format(data_type, attribute)
        return command

    def test_save(self, study, tmpdir):
        from flotilla.datapackage import name_to_resource

        study_name = 'test_save'

        study.save(study_name, flotilla_dir=tmpdir)

        assert len(tmpdir.listdir()) == 1
        save_dir = tmpdir.listdir()[0]

        with open('{}/datapackage.json'.format(save_dir)) as f:
            test_datapackage = json.load(f)

        assert study_name == save_dir.purebasename

        # resource_keys_to_ignore = ('compression', 'format', 'path',
        #                            'url')
        keys_from_study = {'splicing': [],
                           'expression': ['thresh',
                                          'log_base',
                                          'plus_one'],
                           'metadata': ['phenotype_order',
                                        'phenotype_to_color',
                                        'phenotype_col',
                                        'phenotype_to_marker',
                                        'pooled_col',
                                        'minimum_samples'],
                           'mapping_stats': ['number_mapped_col',
                                             'min_reads'],
                           'expression_feature': ['rename_col',
                                                  'ignore_subset_cols'],
                           'splicing_feature': ['rename_col',
                                                'ignore_subset_cols',
                                                'expression_id_col'],
                           'gene_ontology': []}
        resource_names = keys_from_study.keys()

        # Add auto-generated attributes into the true datapackage
        for name, keys in keys_from_study.iteritems():
            resource = name_to_resource(test_datapackage, name)
            for key in keys:
                command = self.get_data_eval_command(name, key)
                test_value = resource[key]
                true_value = eval(command)
                if isinstance(test_value, dict):
                    pdt.assert_dict_equal(test_value, true_value)
                elif isinstance(test_value, Iterable):
                    pdt.assert_array_equal(test_value, true_value)

        for name in resource_names:
            resource = name_to_resource(test_datapackage, name)
            path = '{}.csv.gz'.format(name)
            assert resource['path'] == path
            test_df = pd.read_csv('{}/{}/{}'.format(tmpdir, study_name, path),
                                  index_col=0, compression='gzip')
            command = self.get_data_eval_command(name, 'data_original')
            true_df = eval(command)
            pdt.assert_frame_equal(test_df, true_df)

        version = semantic_version.Version(study.version)
        version.patch += 1
        assert str(version) == test_datapackage['datapackage_version']
        assert study_name == test_datapackage['name']

        # datapackage_keys_to_ignore = ['name', 'datapackage_version',
        #                               'resources']
        # datapackages = (true_datapackage, test_datapackage)

        # for name in resource_names:
        #     for datapackage in datapackages:
        #         resource = name_to_resource(datapackage, name)
        #         for key in resource_keys_to_ignore:
        #             monkeypatch.delitem(resource, key, raising=False)

        # # Have to check for resources separately because they could be
        # # in any order, it just matters that the contents are equal
        # # sorted_true = sorted(true_datapackage['resources'],
        # #                      key=lambda x: x['name'])
        # sorted_test = sorted(test_datapackage['resources'],
        #                      key=lambda x: x['name'])
        # for i in range(len(sorted_true)):
        #     pdt.assert_equal(sorted(sorted_true[i].items()),
        #                      sorted(sorted_test[i].items()))
        #
        # for key in datapackage_keys_to_ignore:
        #     for datapackage in datapackages:
        #         monkeypatch.delitem(datapackage, key)

        # pdt.assert_dict_equal(test_datapackage)

    # Temporary commenting out while chr22 dataset is down
    # def test_nmf_space_positions(self, chr22):
    #     test_positions = chr22.nmf_space_positions()
    #
    #     true_positions = chr22.splicing.nmf_space_positions(
    #         groupby=chr22.sample_id_to_phenotype)
    #
    #     pdt.assert_frame_equal(test_positions, true_positions)

# def test_write_package(tmpdir):
# from flotilla.data_model import StudyFactory
#
# new_study = StudyFactory()
#     new_study.experiment_design_data = None
#     new_study.event_metadata = None
#     new_study.expression_metadata = None
#     new_study.expression_df = None
#     new_study.splicing_df = None
#     new_study.event_metadata = None
#     new_study.write_package('test_package', where=tmpdir, install=False)
