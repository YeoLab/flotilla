"""
This tests whether the Study object was created correctly. No
computation or visualization tests yet.
"""
import copy
import json
import os

import matplotlib.pyplot as plt
import pandas.util.testing as pdt
import pytest
import semantic_version

from flotilla.datapackage import name_to_resource


@pytest.fixture
def toy_study(shalek2013_data):
    from flotilla import Study

    return Study(sample_metadata=shalek2013_data.metadata,
                 version='0.1.0',
                 expression_data=shalek2013_data.expression,
                 splicing_data=shalek2013_data.splicing)


@pytest.fixture(params=['expression', 'splicing'])
def data_type(request):
    return request.param

class TestStudy(object):
    def test_toy_init(self, toy_study, shalek2013_data):
        from flotilla.data_model import ExpressionData, SplicingData

        outliers = shalek2013_data.metadata.index[
            shalek2013_data.metadata.outlier.astype(bool)]
        expression = ExpressionData(data=shalek2013_data.expression,
                                    outliers=outliers)
        splicing = SplicingData(data=shalek2013_data.splicing,
                                outliers=outliers)

        pdt.assert_frame_equal(toy_study.metadata.data,
                               shalek2013_data.metadata)
        pdt.assert_frame_equal(toy_study.expression.data,
                               expression.data)
        pdt.assert_frame_equal(toy_study.splicing.data, splicing.data)
        # There's more to test for correct initialization but this is barebones
        # for now

    def test_real_init(self, shalek2013_datapackage_path):
        import flotilla

        flotilla.embark(shalek2013_datapackage_path, load_species_data=False)

    def test_plot_pca(self, shalek2013, feature_subset, data_type,
                      featurewise, sample_subset):
        shalek2013.plot_pca(feature_subset=feature_subset, data_type=data_type,
                            featurewise=featurewise, sample_subset=sample_subset)

    def test_plot_graph(self, shalek2013, data_type, feature_subset,
                        sample_subset):
        shalek2013.plot_graph(feature_of_interest=None, data_type=data_type,
                              feature_subset=feature_subset,
                              sample_subset=sample_subset)
        plt.close('all')

    def test_plot_classifier(self, shalek2013, data_type, feature_subset,
                             sample_subset):
        shalek2013.plot_classifier('pooled', data_type=data_type,
                                   feature_subset=feature_subset,
                                   sample_subset=sample_subset)
        plt.close('all')

    def test_plot_clustermap(self, shalek2013, data_type, feature_subset,
                             sample_subset):
        shalek2013.plot_clustermap(data_type=data_type,
                                   feature_subset=feature_subset,
                                   sample_subset=sample_subset)
        plt.close('all')

    def test_plot_correlations(self, shalek2013, data_type, feature_subset,
                               featurewise, sample_subset):
        shalek2013.plot_correlations(data_type=data_type,
                                     feature_subset=feature_subset,
                                     featurewise=featurewise,
                                     sample_subset=sample_subset)
        plt.close('all')

    @pytest.fixture(params=[None, 'pooled_col', 'phenotype_col'])
    def metadata_none_key(self, request):
        return request.param

    @pytest.fixture(params=[None])
    def expression_none_key(self, request):
        return request.param

    @pytest.fixture(params=[None, pytest.mark.xfail('feature_rename_col')])
    def splicing_none_key(self, request):
        return request.param

    @pytest.fixture
    def datapackage(self, shalek2013_datapackage, metadata_none_key,
                    expression_none_key, splicing_none_key, monkeypatch):
        datapackage = copy.deepcopy(shalek2013_datapackage)
        datatype_to_key = {'metadata': metadata_none_key,
                           'expression': expression_none_key,
                           'splicing': splicing_none_key}
        for datatype, key in datatype_to_key.iteritems():
            if key is not None:
                resource = name_to_resource(datapackage, datatype)
                if key in resource:
                    monkeypatch.delitem(resource, key, raising=False)
        return datapackage

    @pytest.fixture
    def datapackage_dir(self, shalek2013_datapackage_path):
        return os.path.dirname(shalek2013_datapackage_path)

    # def test_from_datapackage(self, datapackage, datapackage_dir):
    #     import flotilla
    #
    #     study = flotilla.Study.from_datapackage(datapackage, datapackage_dir,
    #                                             load_species_data=False)
    #
    #     metadata_resource = get_resource_from_name(datapackage, 'metadata')
    #     expression_resource = get_resource_from_name(datapackage,
    #                                                  'expression')
    #     splicing_resource = get_resource_from_name(datapackage, 'splicing')
    #
    #     phenotype_col = 'phenotype' if 'phenotype_col' \
    #                                    not in metadata_resource else \
    #     metadata_resource['phenotype_col']
    #     pooled_col = 'pooled' if 'pooled_col' not in metadata_resource else \
    #         metadata_resource['pooled_col']
    #     expression_feature_rename_col = None if \
    #         'feature_rename_col' not in expression_resource \
    #         else expression_resource['feature_rename_col']
    #     splicing_feature_rename_col = 'gene_name' if \
    #         'feature_rename_col' not in splicing_resource \
    #         else splicing_resource['feature_rename_col']
    #
    #     assert study.metadata.phenotype_col == phenotype_col
    #     assert study.metadata.pooled_col == pooled_col
    #     assert study.expression.feature_rename_col \
    #            == expression_feature_rename_col
    #     assert study.splicing.feature_rename_col == splicing_feature_rename_col

    def test_save(self, shalek2013_datapackage_path, shalek2013_datapackage,
                  tmpdir, monkeypatch):
        import flotilla
        from flotilla.datapackage import name_to_resource

        study = flotilla.embark(shalek2013_datapackage_path,
                                load_species_data=False)
        study_name = 'test_save'
        study.save(study_name, flotilla_dir=tmpdir)

        assert len(tmpdir.listdir()) == 1
        save_dir = tmpdir.listdir()[0]

        with open('{}/datapackage.json'.format(save_dir)) as f:
            test_datapackage = json.load(f)
        true_datapackage = copy.deepcopy(shalek2013_datapackage)

        assert study_name == save_dir.purebasename

        resource_keys_to_ignore = ('compression', 'format', 'path', 'url')
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
                           'mapping_stats': ['number_mapped_col'],
                           'expression_feature': ['rename_col',
                                                  'ignore_subset_cols'],
                           'splicing_feature': ['rename_col',
                                                'ignore_subset_cols',
                                                'expression_id_col']}
        resource_names = keys_from_study.keys()

        # Add auto-generated attributes into the true datapackage
        for name, keys in keys_from_study.iteritems():
            resource = name_to_resource(true_datapackage, name)
            for key in keys:
                if 'feature' in name:
                    command = 'study.{}.feature_{}'.format(name.rstrip(
                        '_feature'), key)
                else:
                    command = 'study.{}.{}'.format(name, key)
                monkeypatch.setitem(resource, key, eval(command))

        for name in resource_names:
            resource = name_to_resource(test_datapackage, name)
            assert resource['path'] == '{}.csv.gz'.format(name)

        version = semantic_version.Version(study.version)
        version.patch += 1
        assert str(version) == test_datapackage['datapackage_version']
        assert study_name == test_datapackage['name']

        datapackage_keys_to_ignore = ['name', 'datapackage_version',
                                      'resources']
        datapackages = (true_datapackage, test_datapackage)

        for name in resource_names:
            for datapackage in datapackages:
                resource = name_to_resource(datapackage, name)
                for key in resource_keys_to_ignore:
                    monkeypatch.delitem(resource, key, raising=False)

        # Have to check for resources separately because they could be in any
        # order, it just matters that the contents are equal
        sorted_true = sorted(true_datapackage['resources'],
                             key=lambda x: x['name'])
        sorted_test = sorted(test_datapackage['resources'],
                             key=lambda x: x['name'])
        for i in range(len(sorted_true)):
            pdt.assert_equal(sorted(sorted_true[i].items()),
                             sorted(sorted_test[i].items()))

        for key in datapackage_keys_to_ignore:
            for datapackage in datapackages:
                monkeypatch.delitem(datapackage, key)

        pdt.assert_dict_equal(test_datapackage,
                              true_datapackage)

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
#     new_study = StudyFactory()
#     new_study.experiment_design_data = None
#     new_study.event_metadata = None
#     new_study.expression_metadata = None
#     new_study.expression_df = None
#     new_study.splicing_df = None
#     new_study.event_metadata = None
#     new_study.write_package('test_package', where=tmpdir, install=False)
