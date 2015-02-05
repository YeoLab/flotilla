"""
This tests whether the Study object was created correctly. No
computation or visualization tests yet.
"""
import matplotlib.pyplot as plt
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import pytest


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
    @pytest.fixture
    def n_groups(self):
        return 3

    @pytest.fixture
    def study(self, metadata_data_groups_fixed, metadata_kws_fixed,
              mapping_stats_data, mapping_stats_kws,
              expression_data_no_na, expression_kws,
              splicing_data_fixed, splicing_kws):
        from flotilla.data_model import Study

        kwargs = {}
        kw_pairs = (('metadata', metadata_kws_fixed),
                    ('mapping_stats', mapping_stats_kws),
                    ('expression', expression_kws),
                    ('splicing', splicing_kws))
        for data_type, kws in kw_pairs:
            for kw_name, kw_value in kws.iteritems():
                kwargs['{}_{}'.format(data_type, kw_name)] = kw_value

        return Study(metadata_data_groups_fixed,
                     mapping_stats_data=mapping_stats_data,
                     expression_data=expression_data_no_na,
                     splicing_data=splicing_data_fixed, **kwargs)

    @pytest.fixture
    def study_no_mapping_stats(self, metadata_data_groups_fixed,
                               metadata_kws_fixed,
                               expression_data_no_na, expression_kws,
                               splicing_data_fixed, splicing_kws):
        from flotilla.data_model import Study

        kwargs = {}
        kw_pairs = (('metadata', metadata_kws_fixed),
                    ('expression', expression_kws),
                    ('splicing', splicing_kws))
        for data_type, kws in kw_pairs:
            for kw_name, kw_value in kws.iteritems():
                kwargs['{}_{}'.format(data_type, kw_name)] = kw_value

        return Study(metadata_data_groups_fixed,
                     expression_data=expression_data_no_na,
                     splicing_data=splicing_data_fixed, **kwargs)

    # @pytest.mark.parameterize('n_groups', '3_groups')
    def test__init(self, study, pooled, technical_outliers):
        # Also need to check for when these are NAs

        if pooled is None:
            npt.assert_equal(study.pooled, None)
        else:
            npt.assert_array_equal(sorted(study.pooled), sorted(pooled))
        if technical_outliers is None:
            pdt.assert_array_equal(study.technical_outliers, pd.Index([]))
        else:
            pdt.assert_array_equal(sorted(study.technical_outliers),
                                   sorted(technical_outliers))

    def test_plot_pca(self, study_no_mapping_stats, color_samples_by):
        study_no_mapping_stats.plot_pca(color_samples_by=color_samples_by,
                                        feature_subset='all')
        plt.close('all')

    def test_plot_pca_splicing(self, study_no_mapping_stats, color_samples_by):
        study_no_mapping_stats.plot_pca(color_samples_by=color_samples_by,
                                        data_type='splicing',
                                        feature_subset='all')
        plt.close('all')

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
    #
    #     def test_save(self, shalek2013_datapackage_path,
    #     #             shalek2013_datapackage,
    #                   tmpdir, monkeypatch):
    #         import flotilla
    #         from flotilla.datapackage import name_to_resource
    #
    #         study = flotilla.embark(shalek2013_datapackage_path,
    #                                 load_species_data=False)
    #         study_name = 'test_save'
    #         study.save(study_name, flotilla_dir=tmpdir)
    #
    #         assert len(tmpdir.listdir()) == 1
    #         save_dir = tmpdir.listdir()[0]
    #
    #         with open('{}/datapackage.json'.format(save_dir)) as f:
    #             test_datapackage = json.load(f)
    #         true_datapackage = copy.deepcopy(shalek2013_datapackage)
    #
    #         assert study_name == save_dir.purebasename
    #
    #         resource_keys_to_ignore = ('compression', 'format', 'path',
    #                                    'url')
    #         keys_from_study = {'splicing': [],
    #                            'expression': ['thresh',
    #                                           'log_base',
    #                                           'plus_one'],
    #                            'metadata': ['phenotype_order',
    #                                         'phenotype_to_color',
    #                                         'phenotype_col',
    #                                         'phenotype_to_marker',
    #                                         'pooled_col',
    #                                         'minimum_samples'],
    #                            'mapping_stats': ['number_mapped_col'],
    #                            'expression_feature': ['rename_col',
    #                                                   'ignore_subset_cols'],
    #                            'splicing_feature': ['rename_col',
    #                                                 'ignore_subset_cols',
    #                                                 'expression_id_col']}
    #         resource_names = keys_from_study.keys()
    #
    #         # Add auto-generated attributes into the true datapackage
    #         for name, keys in keys_from_study.iteritems():
    #             resource = name_to_resource(true_datapackage, name)
    #             for key in keys:
    #                 if 'feature' in name:
    #                     command = 'study.{}.feature_{}'.format(name.rstrip(
    #                         '_feature'), key)
    #                 else:
    #                     command = 'study.{}.{}'.format(name, key)
    #                 monkeypatch.setitem(resource, key, eval(command))
    #
    #         for name in resource_names:
    #             resource = name_to_resource(test_datapackage, name)
    #             assert resource['path'] == '{}.csv.gz'.format(name)
    #
    #         version = semantic_version.Version(study.version)
    #         version.patch += 1
    #         assert str(version) == test_datapackage['datapackage_version']
    #         assert study_name == test_datapackage['name']
    #
    #         datapackage_keys_to_ignore = ['name', 'datapackage_version',
    #                                       'resources']
    #         datapackages = (true_datapackage, test_datapackage)
    #
    #         for name in resource_names:
    #             for datapackage in datapackages:
    #                 resource = name_to_resource(datapackage, name)
    #                 for key in resource_keys_to_ignore:
    #                     monkeypatch.delitem(resource, key, raising=False)
    #
    #         # Have to check for resources separately because they could be
    #         # in any
    #         # order, it just matters that the contents are equal
    #         sorted_true = sorted(true_datapackage['resources'],
    #                              key=lambda x: x['name'])
    #         sorted_test = sorted(test_datapackage['resources'],
    #                              key=lambda x: x['name'])
    #         for i in range(len(sorted_true)):
    #             pdt.assert_equal(sorted(sorted_true[i].items()),
    #                              sorted(sorted_test[i].items()))
    #
    #         for key in datapackage_keys_to_ignore:
    #             for datapackage in datapackages:
    #                 monkeypatch.delitem(datapackage, key)
    #
    #         pdt.assert_dict_equal(test_datapackage,
    #                               true_datapackage)

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
