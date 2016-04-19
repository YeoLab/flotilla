import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
from sklearn.preprocessing import StandardScaler
import pytest

# @pytest.fixture(params=['expression', 'splicing'])
# def data_type(request):
# return request.param
#

# @pytest.fixture
# def data(data_type, expression_data, splicing_data):
# if data_type == 'expression':
# return expression_data
# elif data_type == 'splicing':
#         return splicing_data
#
# @pytest.fixture
# def feature_data(data_type, expression_feature_data, splicing_feature_data):
#     if data_type == 'expression':
#         return expression_feature_data
#     elif data_type == 'splicing':
#         return splicing_feature_data
#
# @pytest.fixture
# def thresh(data_type, expression_thresh):
#     if data_type == 'expression':
#         return expression_thresh
#     else:
#         return -np.inf
#
# @pytest.fixture
# def feature_rename_col(data_type, )


class TestBaseData:
    def test__init(self, expression_data_no_na, outliers):
        from flotilla.data_model.base import BaseData
        from flotilla.compute.predict import PredictorConfigManager, \
            PredictorDataSetManager

        base_data = BaseData(expression_data_no_na, outliers=outliers)
        outlier_samples = outliers.copy() if outliers is not None else []
        outliers_df = expression_data_no_na.ix[outlier_samples]

        feature_renamer_series = pd.Series(expression_data_no_na.columns,
                                           index=expression_data_no_na.columns)

        pdt.assert_frame_equal(base_data.data_original, expression_data_no_na)
        pdt.assert_equal(base_data.feature_data, None)
        pdt.assert_frame_equal(base_data.data, expression_data_no_na)
        pdt.assert_series_equal(base_data.feature_renamer_series,
                                feature_renamer_series)
        pdt.assert_frame_equal(base_data.outliers, outliers_df)
        pdt.assert_numpy_array_equal(base_data.outlier_samples,
                                     outlier_samples)
        assert isinstance(base_data.predictor_config_manager,
                          PredictorConfigManager)
        assert isinstance(base_data.predictor_dataset_manager,
                          PredictorDataSetManager)

    def test_feature_renamer_series_change_col(self, expression_data_no_na,
                                               expression_feature_data,
                                               expression_feature_rename_col,
                                               n_genes):
        from flotilla.data_model.base import BaseData

        expression_feature_data = expression_feature_data.copy()
        gene_numbers = np.arange(n_genes)
        new_renamer = 'new_renamer'
        expression_feature_data[new_renamer] = \
            expression_feature_data.index.map(
                lambda x: 'new_renamed{}'.format(
                    np.random.choice(gene_numbers)))

        base_data = BaseData(expression_data_no_na,
                             feature_data=expression_feature_data,
                             feature_rename_col=expression_feature_rename_col)
        base_data.feature_rename_col = new_renamer
        pdt.assert_series_equal(base_data.feature_renamer_series,
                                expression_feature_data[new_renamer])

    def test__init_technical_outliers(self, expression_data_no_na,
                                      technical_outliers):
        from flotilla.data_model.base import BaseData

        base_data = BaseData(expression_data_no_na,
                             technical_outliers=technical_outliers)

        data = expression_data_no_na.copy()
        if technical_outliers is not None:
            good_samples = ~data.index.isin(technical_outliers)
            data = data.ix[good_samples]
        pdt.assert_frame_equal(base_data.data, data)
        pdt.assert_frame_equal(base_data.data_original,
                               expression_data_no_na)

    def test__init_sample_thresholds(self, expression_data,
                                     expression_thresh,
                                     metadata_minimum_samples,
                                     pooled):
        from flotilla.data_model.base import BaseData

        base_data = BaseData(expression_data,
                             thresh=expression_thresh,
                             minimum_samples=metadata_minimum_samples,
                             pooled=pooled)
        data = expression_data.copy()
        pooled_samples = pooled.copy() if pooled is not None else []
        single_samples = data.index[~data.index.isin(pooled_samples)]
        singles_df = data.ix[single_samples]

        if expression_thresh > -np.inf or metadata_minimum_samples > 0:
            if not singles_df.empty:
                data = base_data._threshold(data, singles_df)
            else:
                data = base_data._threshold(data)

        singles_df = data.ix[single_samples]
        pooled_df = data.ix[pooled_samples]

        pdt.assert_frame_equal(base_data.data_original, expression_data)
        pdt.assert_frame_equal(base_data.data, data)
        pdt.assert_equal(base_data.thresh, expression_thresh)
        pdt.assert_equal(base_data.minimum_samples, metadata_minimum_samples)
        pdt.assert_frame_equal(base_data.pooled, pooled_df)
        pdt.assert_frame_equal(base_data.singles, singles_df)

    def test__init__featuredata(self, expression_data_no_na,
                                expression_feature_data,
                                expression_feature_rename_col):
        from flotilla.data_model.base import BaseData, \
            subsets_from_metadata, MINIMUM_FEATURE_SUBSET

        base_data = BaseData(expression_data_no_na,
                             feature_data=expression_feature_data,
                             feature_rename_col=expression_feature_rename_col)

        if expression_feature_rename_col is not None:
            feature_renamer_series = expression_feature_data[
                expression_feature_rename_col]
        else:
            feature_renamer_series = pd.Series(
                expression_feature_data.index,
                index=expression_feature_data.index)
        feature_subsets = subsets_from_metadata(expression_feature_data,
                                                MINIMUM_FEATURE_SUBSET,
                                                'features')
        feature_subsets['variant'] = base_data.variant

        pdt.assert_frame_equal(base_data.data_original, expression_data_no_na)
        pdt.assert_frame_equal(base_data.feature_data, expression_feature_data)
        pdt.assert_frame_equal(base_data.data, expression_data_no_na)
        pdt.assert_series_equal(base_data.feature_renamer_series,
                                feature_renamer_series)
        pdt.assert_dict_equal(base_data.feature_subsets, feature_subsets)

    @pytest.mark.xfail
    def test__init_multiindex(self, df_norm):
        from flotilla.data_model.base import BaseData

        data = df_norm.copy()
        level1 = data.columns.map(lambda x: 'level1_{}'.format(x))
        data.columns = pd.MultiIndex.from_arrays([data.columns, level1])
        BaseData(data)

    def test__variant(self, expression_data):
        from flotilla.data_model.base import BaseData

        base_data = BaseData(expression_data)

        var = expression_data.var()
        var_cut = var.mean() + 2 * var.std()
        variant = expression_data.columns[var > var_cut]

        pdt.assert_equal(base_data._var_cut, var_cut)
        pdt.assert_numpy_array_equal(base_data.variant, variant)

    def test__subset(self, expression_data_no_na, sample_ids, feature_ids):
        from flotilla.data_model.base import BaseData

        base_data = BaseData(expression_data_no_na)
        subset = base_data._subset(base_data.data, sample_ids=sample_ids,
                                   feature_ids=feature_ids)
        data = base_data.data
        if feature_ids is None:
            feature_ids = data.columns
        else:
            feature_ids = pd.Index(set(feature_ids).intersection(data.columns))
        if sample_ids is None:
            sample_ids = data.index
        else:
            sample_ids = pd.Index(set(sample_ids).intersection(data.index))

        true_subset = data.ix[sample_ids, feature_ids]

        pdt.assert_frame_equal(subset, true_subset)

    def test__subset_and_standardize(self, expression_data_no_na,
                                     standardize, feature_ids,
                                     sample_ids):
        from flotilla.data_model.base import BaseData

        base_data = BaseData(expression_data_no_na)
        base_data.subset, base_data.means = \
            base_data._subset_and_standardize(base_data.data,
                                              sample_ids=sample_ids,
                                              feature_ids=feature_ids,
                                              return_means=True,
                                              standardize=standardize)

        subset = base_data._subset(base_data.data, sample_ids=sample_ids,
                                   feature_ids=feature_ids)
        means = subset.mean().rename_axis(base_data.feature_renamer)
        subset = subset.fillna(means).fillna(0)
        subset = subset.rename_axis(base_data.feature_renamer, 1)

        if standardize:
            data = StandardScaler().fit_transform(subset)
        else:
            data = subset

        subset_standardized = pd.DataFrame(data, index=subset.index,
                                           columns=subset.columns)

        pdt.assert_frame_equal(subset_standardized, base_data.subset)
        pdt.assert_series_equal(means, base_data.means)

    def test__threshold(self, expression_data_no_na, pooled):
        from flotilla.data_model.base import BaseData

        thresh = 0.5
        minimum_samples = 5
        base_data = BaseData(expression_data_no_na, thresh=thresh,
                             minimum_samples=minimum_samples, pooled=pooled)
        data = expression_data_no_na.copy()
        if pooled is not None:
            other = base_data.singles
        else:
            other = data

        filtered = data.ix[:, other[other > thresh].count() >= minimum_samples]
        pdt.assert_frame_equal(base_data.data, filtered)

    def test_reduce(self, expression_data_no_na, featurewise):
        # TODO: parameterize and test with featurewise and subsets
        from flotilla.compute.decomposition import DataFramePCA
        from flotilla.data_model.base import BaseData

        expression = BaseData(expression_data_no_na)
        test_reduced = expression.reduce(featurewise=featurewise)

        subset, means = expression._subset_and_standardize(
            expression.data, return_means=True, standardize=True)

        if featurewise:
            subset = subset.T

        true_reduced = DataFramePCA(subset)
        true_reduced.means = means

        pdt.assert_frame_equal(test_reduced.X, subset)
        npt.assert_array_equal(test_reduced.reduced_space,
                               true_reduced.reduced_space)
        pdt.assert_series_equal(test_reduced.means,
                                true_reduced.means)

    def test_feature_subset_to_feature_ids(self, expression_data_no_na,
                                           expression_feature_data,
                                           feature_subset):
        from flotilla.data_model.base import BaseData

        expression = BaseData(expression_data_no_na,
                              feature_data=expression_feature_data)
        test_feature_ids = expression.feature_subset_to_feature_ids(
            feature_subset, rename=False)

        true_feature_ids = pd.Index([])
        if feature_subset is not None:
            try:
                if feature_subset in expression.feature_subsets:
                    true_feature_ids = expression.feature_subsets[
                        feature_subset]
                elif feature_subset.startswith('all'):
                    true_feature_ids = expression.data.columns
            except TypeError:
                if not isinstance(feature_subset, str):
                    feature_ids = feature_subset
                    n_custom = expression.feature_data.columns.map(
                        lambda x: x.startswith('custom')).sum()
                    ind = 'custom_{}'.format(n_custom + 1)
                    expression.feature_data[ind] = \
                        expression.feature_data.index.isin(feature_ids)
                else:
                    raise ValueError(
                        "There are no {} features in this data: "
                        "{}".format(feature_subset, self))
        else:
            true_feature_ids = expression.data.columns
        pdt.assert_numpy_array_equal(test_feature_ids, true_feature_ids)


def test_subsets_from_metadata(expression_feature_data):
    from flotilla.data_model.base import subsets_from_metadata

    minimum = 5
    subset_type = 'expression'
    metadata = expression_feature_data.copy()

    test_subsets = subsets_from_metadata(metadata,
                                         5, 'expression')

    true_subsets = {}
    sorted_bool = (False, True)
    if true_subsets is not None:
        for col in expression_feature_data:
            if tuple(sorted(metadata[col].dropna().unique())) == sorted_bool:
                series = metadata[col].dropna()
                sample_subset = series[series].index
                true_subsets[col] = sample_subset
            else:
                grouped = metadata.groupby(col)
                sizes = grouped.size()
                filtered_sizes = sizes[sizes >= minimum]
                for group in filtered_sizes.keys():
                    if isinstance(group, bool):
                        continue
                    name = '{}: {}'.format(col, group)
                    true_subsets[name] = grouped.groups[group]
        for sample_subset in true_subsets.keys():
            name = 'not ({})'.format(sample_subset)
            if 'False' in name or 'True' in name:
                continue
            if name not in true_subsets:
                in_features = metadata.index.isin(true_subsets[
                    sample_subset])
                true_subsets[name] = metadata.index[~in_features]
        true_subsets['all {}'.format(subset_type)] = metadata.index

    pdt.assert_dict_equal(true_subsets, test_subsets)

    # Make sure every column is in the subsets
    for col in expression_feature_data:
        if tuple(sorted(metadata[col].dropna().unique())) == sorted_bool:
            assert col in test_subsets
