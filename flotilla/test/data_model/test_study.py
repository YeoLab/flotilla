"""
This tests whether the Study object was created correctly. No
computation or visualization tests yet.
"""
import json
import os

import matplotlib.pyplot as plt
import pandas.util.testing as pdt
import pytest
import semantic_version

from flotilla.datapackage import get_resource_from_name, \
    data_package_url_to_dict


def name_to_resource(datapackage, name):
    """
    Given the name of a resource, search through a datapackage's "resource"
    list and return that dictionary
    """
    for resource in datapackage['resources']:
        if resource['name'] == name:
            return resource
    raise ValueError('No resource named {} in this datapackage'.format(name))


class TestStudy(object):
    @pytest.fixture
    def toy_study(self, example_data):
        from flotilla import Study

        return Study(sample_metadata=example_data.metadata,
                     version='0.1.0',
                     expression_data=example_data.expression,
                     splicing_data=example_data.splicing)

    def test_toy_init(self, toy_study, example_data):
        from flotilla.data_model import ExpressionData, SplicingData

        outliers = example_data.metadata.index[
            example_data.metadata.outlier.astype(bool)]
        expression = ExpressionData(data=example_data.expression,
                                    outliers=outliers)
        splicing = SplicingData(data=example_data.splicing, outliers=outliers)

        pdt.assert_frame_equal(toy_study.metadata.data,
                               example_data.metadata)
        pdt.assert_frame_equal(toy_study.expression.data,
                               expression.data)
        pdt.assert_frame_equal(toy_study.splicing.data, splicing.data)
        # There's more to test for correct initialization but this is barebones
        # for now

    def test_real_init(self, example_datapackage_path):
        import flotilla

        flotilla.embark(example_datapackage_path, load_species_data=False)

    def test_plot_pca(self, study, feature_subset):
        study.plot_pca(feature_subset=feature_subset)

    def test_plot_graph(self, study):
        study.plot_graph(feature_of_interest=None)
        plt.close('all')

    def test_plot_classifier(self, study):
        study.plot_classifier('pooled')
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
    def datapackage(self, example_datapackage_path, metadata_none_key,
                    expression_none_key, splicing_none_key):
        with open(example_datapackage_path) as f:
            datapackage = json.load(f)
        datatype_to_key = {'metadata': metadata_none_key,
                           'expression': expression_none_key,
                           'splicing': splicing_none_key}
        for datatype, key in datatype_to_key.iteritems():
            if key is not None:
                resource = name_to_resource(datapackage, datatype)
                if key in resource:
                    resource.pop(key)
        return datapackage

    @pytest.fixture
    def datapackage_dir(self, example_datapackage_path):
        return os.path.dirname(example_datapackage_path)

    def test_from_datapackage(self, datapackage, datapackage_dir):
        import flotilla

        study = flotilla.Study.from_datapackage(datapackage, datapackage_dir,
                                                load_species_data=False)

        metadata_resource = get_resource_from_name(datapackage, 'metadata')
        expression_resource = get_resource_from_name(datapackage,
                                                     'expression')
        splicing_resource = get_resource_from_name(datapackage, 'splicing')

        phenotype_col = 'phenotype' if 'phenotype_col' \
                                       not in metadata_resource else \
        metadata_resource['phenotype_col']
        pooled_col = 'pooled' if 'pooled_col' not in metadata_resource else \
            metadata_resource['pooled_col']
        expression_feature_rename_col = None if \
            'feature_rename_col' not in expression_resource \
            else expression_resource['feature_rename_col']
        splicing_feature_rename_col = 'gene_name' if \
            'feature_rename_col' not in splicing_resource \
            else splicing_resource['feature_rename_col']

        assert study.metadata.phenotype_col == phenotype_col
        assert study.metadata.pooled_col == pooled_col
        assert study.expression.feature_rename_col \
               == expression_feature_rename_col
        assert study.splicing.feature_rename_col == splicing_feature_rename_col

    def test_save(self, example_datapackage_path, tmpdir, monkeypatch):
        import flotilla
        from flotilla.datapackage import get_resource_from_name

        study = flotilla.embark(example_datapackage_path,
                                load_species_data=False)
        study_name = 'test_save'
        study.save(study_name, flotilla_dir=tmpdir)

        assert len(tmpdir.listdir()) == 1
        save_dir = tmpdir.listdir()[0]

        with open('{}/datapackage.json'.format(save_dir)) as f:
            test_datapackage = json.load(f)
        true_datapackage = data_package_url_to_dict(example_datapackage_path)

        assert study_name == save_dir.purebasename

        resource_keys_to_ignore = ('compression', 'format', 'path', 'url')
        keys_from_study = {'splicing': [],
                           'expression': ['thresh',
                                          'log_base'],
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
                                                'ignore_subset_cols']}
        resource_names = keys_from_study.keys()

        # Add auto-generated attributes into the true datapackage
        for name, keys in keys_from_study.iteritems():
            resource = get_resource_from_name(true_datapackage, name)
            for key in keys:
                if 'feature' in name:
                    command = 'study.{}.feature_{}'.format(name.rstrip(
                        '_feature'), key)
                else:
                    command = 'study.{}.{}'.format(name, key)
                monkeypatch.setitem(resource, key, eval(command))

        version = semantic_version.Version(study.version)
        version.patch += 1
        assert str(version) == test_datapackage['datapackage_version']
        assert study_name == test_datapackage['name']

        datapackage_keys_to_ignore = ['name', 'datapackage_version',
                                      'resources']
        datapackages = (true_datapackage, test_datapackage)

        for name in resource_names:
            for datapackage in datapackages:
                resource = get_resource_from_name(datapackage, name)
                for key in resource_keys_to_ignore:
                    monkeypatch.delitem(resource, key, raising=False)

        # Have to check for resources separately because they could be in any
        # order, it just matters that the contents are equal
        assert sorted(true_datapackage['resources']) == sorted(
            test_datapackage['resources'])

        for key in datapackage_keys_to_ignore:
            for datapackage in datapackages:
                monkeypatch.delitem(datapackage, key)

        pdt.assert_dict_equal(test_datapackage,
                              true_datapackage)


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
