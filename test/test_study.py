"""
This tests whether the Study object was created correctly. No
computation or visualization tests yet.
"""
import json
import os

import pandas.util.testing as pdt
import pytest


class TestStudy(object):
    @pytest.fixture
    def toy_study(self, example_data):
        from flotilla import Study

        return Study(sample_metadata=example_data.metadata,
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

        flotilla.embark(example_datapackage_path)

    def test_plot_pca(self, study):
        study.plot_pca()

    def test_plot_graph(self, study):
        study.plot_graph(feature_of_interest=None)

    def test_plot_classifier(self, study):
        study.plot_classifier('P_cell')

    @pytest.fixture(scope="module", params=[None, 'pooled_col',
                                            'phenotype_col'])
    def metadata_key(self, request):
        return request.param

    @pytest.fixture(params=[None, 'feature_rename_col'])
    def expression_key(self, request):
        return request.param

    @pytest.fixture(params=[None, 'feature_rename_col'])
    def splicing_key(self, request):
        return request.param

    @pytest.fixture
    def datapackage(self, example_datapackage_path, metadata_key,
                    expression_key, splicing_key):
        with open(example_datapackage_path) as f:
            datapackage = json.load(f)
        datatype_to_key = {'metadata': metadata_key,
                           'expression': expression_key,
                           'splicing': splicing_key}
        for datatype, key in datatype_to_key.iteritems():
            if key is not None:
                resources = datapackage['resources']
                resource = [r for r in resources
                            if r['name'] == datatype][0]
                if key in resource:
                    resource.pop(key)
        return datapackage

    @pytest.fixture
    def datapackage_dir(self, example_datapackage_path):
        return os.path.dirname(example_datapackage_path)

    def test_from_datapackage(self, datapackage, datapackage_dir):
        import flotilla

        study = flotilla.Study.from_datapackage(datapackage, datapackage_dir)

        metadata_resource = [r for r in datapackage['resources']
                             if r['name'] == 'metadata'][0]
        expression_resource = [r for r in datapackage['resources']
                               if r['name'] == 'expression'][0]
        splicing_resource = [r for r in datapackage['resources']
                             if r['name'] == 'splicing'][0]

        phenotype_col = 'phenotype' if 'phenotype_col' \
            not in metadata_resource else metadata_resource['phenotype_col']
        pooled_col = None if 'pooled_col' not in metadata_resource else \
            metadata_resource['pooled_col']
        expression_feature_rename_col = 'gene_name' if \
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


# def test_write_package(tmpdir):
#     from flotilla.data_model import StudyFactory
#
#     new_study = StudyFactory()
#     new_study.experiment_design_data = None
#     new_study.event_metadata = None
#     new_study.expression_metadata = None
#     new_study.expression_df = None
#     new_study.splicing_df = None
#     new_study.event_metadata = None
#     new_study.write_package('test_package', where=tmpdir, install=False)
