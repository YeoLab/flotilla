import numpy.testing as npt
import pandas.util.testing as pdt
import pytest

@pytest.fixture
def base_data(example_data):
    from flotilla.data_model.base import BaseData
    return BaseData(example_data.expression)


def test_basedata_init(example_data):
    from flotilla.data_model.base import BaseData
    base_data = BaseData(example_data.expression)
    pdt.assert_frame_equal(base_data.data, example_data.expression)


@pytest.fixture(params=[None, 'half', 'all'])
def sample_ids(request, base_data):
    if request.param is None:
        return request.param
    elif request.param == 'some':
        half = base_data.data.shape[0] / 2
        return base_data.data.index[:half]
    elif request.param == 'all':
        return base_data.data.index


@pytest.fixture(params=[None, 'half', 'all'])
def feature_ids(request, base_data):
    if request.param is None:
        return request.param
    elif request.param == 'some':
        half = base_data.data.shape[1] / 2
        return base_data.data.columns[:half]
    elif request.param == 'all':
        return base_data.data.columns


def test_subset(base_data, sample_ids, feature_ids):
    import pandas as pd

    subset = base_data._subset(base_data.data, sample_ids=sample_ids,
                               feature_ids=feature_ids)

    data = base_data.data
    if feature_ids is None:
        feature_ids = data.columns
    if sample_ids is None:
        sample_ids = data.index

    sample_ids = pd.Index(set(sample_ids).intersection(data.index))
    feature_ids = pd.Index(set(feature_ids).intersection(data.columns))

    true_subset = data.ix[sample_ids]
    true_subset = true_subset.T.ix[feature_ids].T

    pdt.assert_frame_equal(subset, true_subset)


@pytest.fixture(params=[True, False])
def standardize(request):
    return request.param


def test__subset_and_standardize(base_data, standardize, feature_ids,
                                 sample_ids):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

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

@pytest.fixture(params=[True, False])
def featurewise(request):
    return request.param

def test_reduce(example_data, featurewise):
    # TODO: parameterize and test with featurewise and subsets
    from flotilla.compute.decomposition import DataFramePCA
    from flotilla.data_model.base import BaseData
    expression = BaseData(example_data.expression)
    test_reduced = expression.reduce(featurewise=featurewise)

    if featurewise:
        subset, means = expression._subset_and_standardize(
            expression.data.T, return_means=True)
    else:
        subset, means = expression._subset_and_standardize(
            expression.data, return_means=True)
    true_reduced = DataFramePCA(subset)
    true_reduced.means = means

    pdt.assert_frame_equal(test_reduced.X, subset)
    npt.assert_array_equal(test_reduced.reduced_space,
                           true_reduced.reduced_space)
    pdt.assert_series_equal(test_reduced.means,
                            true_reduced.means)
