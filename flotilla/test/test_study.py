"""
This tests whether the Study object was created correctly.
No computation or visualization tests yet.
"""

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six import iteritems

from collections import Iterable
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import pytest
import semantic_version


##############################################################################
# FIXTURES


@pytest.fixture(params=['expression', 'splicing'])
def data_type(request):
    """data_type fixture"""
    return request.param


@pytest.fixture(params=[None, 'subset1'],
                ids=['color_samples_by_none', 'color_samples_by_subset1'])
def color_samples_by(request, metadata_phenotype_col):
    """color_samples_by fixture"""
    if request.param == 'phenotype':
        return metadata_phenotype_col
    else:
        return request.param


class TestStudy(object):
    # @pytest.fixture
    # def n_groups(self):
    #     return 3

    ##########################################################################
    @pytest.fixture
    def study(self,
              metadata_data, metadata_kws,
              mapping_stats_data, mapping_stats_kws,
              expression_data, expression_kws,
              splicing_data, splicing_kws,
              gene_ontology_data):
        """study fixture"""
        from flotilla import Study

        kwargs = {}
        metadata = metadata_data.copy()
        splicing = splicing_data.copy()
        expression = expression_data.copy()
        mapping_stats = mapping_stats_data.copy()
        gene_ontology = gene_ontology_data.copy()

        kw_pairs = (('metadata', metadata_kws),
                    ('mapping_stats', mapping_stats_kws),
                    ('expression', expression_kws),
                    ('splicing', splicing_kws))
        for data_type, kws in kw_pairs:
            for kw_name, kw_value in iteritems(kws):
                kwargs['{}_{}'.format(data_type, kw_name)] = kw_value

        return Study(metadata,
                     mapping_stats_data=mapping_stats,
                     expression_data=expression,
                     splicing_data=splicing,
                     gene_ontology_data=gene_ontology,
                     **kwargs)

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
        pdt.assert_numpy_array_equal(study.phenotype_transitions,
                                     study.metadata.phenotype_transitions)
        pdt.assert_numpy_array_equal(study.phenotype_color_ordered,
                                     study.metadata.phenotype_color_order)
        pdt.assert_equal(study.default_sample_subset, 'all_samples')
        pdt.assert_equal(study.default_feature_subset, 'variant')
        pdt.assert_numpy_array_equal(study.default_sample_subsets,
                                     true_default_sample_subsets)
        pdt.assert_dict_equal(study.default_feature_subsets, {})

    #########################################################################
    @pytest.mark.xfail
    def test_setattr(self, metadata_data):
        # warnings.simplefilter("error")
        from flotilla import Study
        study = Study(metadata_data.copy())
        study.pooled = 'asdf'
        # warnings.simplefilter('default')
    #########################################################################

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
        pdt.assert_numpy_array_equal(sorted(study.technical_outliers),
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
        pdt.assert_numpy_array_equal(study.expression.data_original,
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
        pdt.assert_numpy_array_equal(study.splicing.data_original,
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
            true_feature_subset = \
                study.splicing.feature_subset_to_feature_ids(feature_subset,
                                                             rename=False)
        pdt.assert_numpy_array_equal(test_feature_subset, true_feature_subset)

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

        pdt.assert_numpy_array_equal(true_sample_subset, test_sample_subset)
    ##########################################################################

    @pytest.fixture(params=[True, False])
    def multiple_genes_per_event(self, request):
        """multiple_genes_per_event fixture"""
        return request.param

    def test_tidy_splicing_with_expression(self, study, monkeypatch,
                                           multiple_genes_per_event):
        if multiple_genes_per_event:
            df = study.splicing.feature_data.copy()
            events = df.index[:5]
            column = study.splicing.feature_expression_id_col

            # fixed for unicode issue
            # when multiple_genes_per_event == True,
            # was getting this kind of value in gene_name column:
            # "b'gene_1',b'gene_2'"
            # df.ix[events, column] = '{},{}'.format(
            #     *study.expression.data.columns[:2])
            df.ix[events, column] = u','.join(
                study.expression.data.columns[:2])

            monkeypatch.setattr(study.splicing, 'feature_data', df)
        test = study.tidy_splicing_with_expression

        splicing_common_id = study.splicing.feature_data[
            study.splicing.feature_expression_id_col]

        # Tidify splicing
        splicing = study.splicing.data
        splicing_index_name = study._maybe_get_axis_name(splicing, axis=0)
        splicing_columns_name = study._maybe_get_axis_name(splicing, axis=1)

        splicing_tidy = pd.melt(splicing.reset_index(),
                                id_vars=splicing_index_name,
                                value_name='psi',
                                var_name=splicing_columns_name)

        s = splicing_common_id.dropna()
        event_name_to_ensembl_ids = list(
            itertools.chain(*[zip([k] * len(v.split(u',')), v.split(u','))
                              for k, v in iteritems(s)])
        )
        index, data = zip(*event_name_to_ensembl_ids)
        event_name_to_ensembl_ids = pd.Series(data, index=index,
                                              name=study._common_id)

        rename_columns = {}
        if splicing_index_name == 'index':
            rename_columns[splicing_index_name] = study._sample_id
        if splicing_columns_name == 'columns':
            rename_columns[splicing_columns_name] = study._event_name
            splicing_columns_name = study._event_name
        splicing_tidy = splicing_tidy.rename(columns=rename_columns)

        splicing_tidy = splicing_tidy.set_index(splicing_columns_name)
        splicing_tidy = splicing_tidy.ix[event_name_to_ensembl_ids.index]
        splicing_tidy = splicing_tidy.join(event_name_to_ensembl_ids)

        splicing_tidy = splicing_tidy.dropna().reset_index()
        splicing_tidy = splicing_tidy.rename(
            columns={'index': study._event_name})

        # Tidify expression
        expression = study.expression.data_original
        expression_index_name = study._maybe_get_axis_name(expression, axis=0)

        expression_tidy = pd.melt(expression.reset_index(),
                                  id_vars=expression_index_name,
                                  value_name='expression',
                                  var_name=study._common_id)
        # This will only do anything if there is a column named "index" so
        # no need to check anything
        expression_tidy = expression_tidy.rename(
            columns={'index': study._sample_id})
        expression_tidy = expression_tidy.dropna()

        true = splicing_tidy.merge(
            expression_tidy, left_on=[study._sample_id, study._common_id],
            right_on=[study._sample_id, study._common_id])
        pdt.assert_frame_equal(test, true)
        assert 'event_name' in test
        assert 'event_name' in true
        assert 'common_id' in true
        assert 'common_id' in test

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

    def test_plot_gene(self, study):
        feature_id = study.expression.data.columns[0]
        study.plot_gene(feature_id)

        fig = plt.gcf()
        test_figsize = fig.get_size_inches()

        feature_ids = [feature_id]
        groupby = study.sample_id_to_phenotype
        grouped = groupby.groupby(groupby)
        single_violin_width = 0.5
        ax_width = max(4, single_violin_width*grouped.size().shape[0])
        nrows = len(feature_ids)
        ncols = 1
        true_figsize = ax_width * ncols, 4 * nrows
        npt.assert_array_equal(true_figsize, test_figsize)

    def test_plot_event(self, study):
        feature_id = study.splicing.data.columns[0]
        col_wrap = 4
        study.plot_event(feature_id, col_wrap=col_wrap)

        fig = plt.gcf()
        test_figsize = fig.get_size_inches()

        feature_ids = [feature_id]
        groupby = study.sample_id_to_phenotype
        grouped = groupby.groupby(groupby)
        single_violin_width = 0.5
        ax_width = max(4, single_violin_width*grouped.size().shape[0])
        nrows = 1
        ncols = 1
        while nrows * ncols < len(feature_ids):
            if ncols > col_wrap:
                nrows += 1
            else:
                ncols += 1

        true_figsize = ax_width * ncols, 4 * nrows
        npt.assert_array_equal(true_figsize, test_figsize)

    def test_plot_event_multiple_events_per_id(self, study):
        grouped = study.splicing.feature_data.groupby(
            study.splicing.feature_rename_col)
        ids_with_multiple_genes = grouped.filter(lambda x: len(x) > 1)
        feature_id = ids_with_multiple_genes[
            study.splicing.feature_rename_col].values[0]
        col_wrap = 4
        study.plot_event(feature_id, col_wrap=col_wrap)

        fig = plt.gcf()
        test_figsize = fig.get_size_inches()

        feature_ids = study.splicing.maybe_renamed_to_feature_id(feature_id)
        groupby = study.sample_id_to_phenotype
        grouped = groupby.groupby(groupby)
        single_violin_width = 0.5
        ax_width = max(4, single_violin_width*grouped.size().shape[0])
        nrows = 1
        ncols = 1
        while nrows * ncols < len(feature_ids):
            if ncols > col_wrap:
                nrows += 1
            else:
                ncols += 1
        true_figsize = ax_width * ncols, 4 * nrows
        npt.assert_array_equal(true_figsize, test_figsize)

    ##########################################################################
    @pytest.fixture(params=[True, False])
    def plot_violins(self, request):
        """plot_violins fixture"""
        return request.param

    def test_plot_pca(self, study, data_type, plot_violins):
        study.plot_pca(feature_subset='all', data_type=data_type,
                       plot_violins=plot_violins)
        plt.close('all')

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

    ##########################################################################
    @pytest.fixture(params=[None, 'gene'])
    def gene_of_interest(self, request, genes):
        """gene_of_interest feature"""
        if request is not None:
            return genes[0]
        else:
            return request.param

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
        print("command :", command)
        return command

    def test_save(self, study, tmpdir):
        from flotilla.datapackage import name_to_resource

        study_name = 'test_save'
        study.supplemental.expression_corr = study.expression.data.corr()
        ###########################################
        study.save(study_name, flotilla_dir=tmpdir)
        ###########################################

        assert len(tmpdir.listdir()) == 1
        save_dir = tmpdir.listdir()[0]

        with open('{}/datapackage.json'.format(save_dir)) as f:
            test_datapackage = json.load(f)

        assert study_name == save_dir.purebasename

        # resource_keys_to_ignore = ('compression', 'format', 'path', 'url')
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
        for name, keys in iteritems(keys_from_study):
            resource = name_to_resource(test_datapackage, name)
            for key in keys:
                command = self.get_data_eval_command(name, key)
                test_value = resource[key]
                true_value = eval(command)
                if isinstance(test_value, dict):
                    #############################################
                    pdt.assert_dict_equal(test_value, true_value)
                elif isinstance(test_value, Iterable):
                    ####################################################
                    pdt.assert_numpy_array_equal(test_value, true_value)

        for name in resource_names:
            resource = name_to_resource(test_datapackage, name)
            # TODO compression
            # path = '{}.csv.gz'.format(name)
            path = '{}.csv'.format(name)
            ###############################
            assert resource['path'] == path
            test_df = pd.read_csv(
                '{}/{}/{}'.format(tmpdir, study_name, path), index_col=0
                # TODO compressiom
                # , compression='gzip'
                )
            command = self.get_data_eval_command(name, 'data_original')
            true_df = eval(command)

            pdt.assert_frame_equal(test_df, true_df)

        version = semantic_version.Version(study.version)
        version.patch += 1
        ##############################################################
        assert str(version) == test_datapackage['datapackage_version']
        assert study_name == test_datapackage['name']

    # TODO D.R.Y. this with the above
    def test_save_supplemental(self, study, tmpdir):
        from flotilla.datapackage import name_to_resource

        study_name = 'test_save_supplemental'
        study.supplemental.expression_corr = study.expression.data.corr()
        ###########################################
        study.save(study_name, flotilla_dir=tmpdir)
        ###########################################

        assert len(tmpdir.listdir()) == 1
        save_dir = tmpdir.listdir()[0]

        with open('{}/datapackage.json'.format(save_dir)) as f:
            test_datapackage = json.load(f)

        supplemental = name_to_resource(test_datapackage, 'supplemental')
        for resource in supplemental['resources']:
            name = resource['name']
            # TODO compression
            # path = '{}.csv.gz'.format(name)
            path = '{}.csv'.format(name)
            ###############################
            assert resource['path'] == path
            full_path = '{}/{}/{}'.format(tmpdir, study_name, path)
            test_df = pd.read_csv(full_path,
                                  index_col=0
                                  # TODO compressiom
                                  # , compression='gzip'
                                  )
            command = self.get_data_eval_command('supplemental', name)
            true_df = eval(command)
            pdt.assert_frame_equal(test_df, true_df)

        version = semantic_version.Version(study.version)
        version.patch += 1
        assert str(version) == test_datapackage['datapackage_version']
        assert study_name == test_datapackage['name']

    def test_embark_supplemental(self, study, tmpdir):
        import flotilla

        study_name = 'test_save_supplemental'
        study.supplemental.expression_corr = study.expression.data.corr()
        study.save(study_name, flotilla_dir=tmpdir)

        study2 = flotilla.embark(study_name, flotilla_dir=tmpdir)
        pdt.assert_frame_equal(study2.supplemental.expression_corr,
                               study.supplemental.expression_corr)
