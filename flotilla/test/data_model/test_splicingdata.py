"""
This tests whether the SplicingData object was created correctly. No
computation or visualization tests yet.
"""
import copy

import numpy as np
import pandas.util.testing as pdt
import pytest


@pytest.fixture
def splicing_data(shalek2013_data, minimum_samples):
    from flotilla.data_model import SplicingData

    return SplicingData(shalek2013_data.splicing, minimum_samples=minimum_samples)

@pytest.fixture(params=[None, 100])
def data_for_binned_nmf_reduced(request, splicing_data):
    if request.param is None:
        return None
    else:
        psi = copy.deepcopy(splicing_data.data)
        max_index = np.prod(map(lambda x: x-1, psi.shape))
        random_flat_indices = np.random.randint(0, max_index, 100)
        psi.values[np.unravel_index(random_flat_indices, psi.shape)] = np.nan
        return psi

class TestSplicingData:
    def test_init(self, splicing_data, shalek2013_data):

        if splicing_data.minimum_samples > 0:
            if not splicing_data.singles.empty:
                singles = shalek2013_data.splicing.ix[splicing_data.singles.index]
                data = splicing_data._threshold(shalek2013_data.splicing,
                                                singles)
            else:
                data = splicing_data._threshold(shalek2013_data.splicing)
        else:
            data = splicing_data.data_original

        pdt.assert_frame_equal(splicing_data.data_original,
                               shalek2013_data.splicing)
        pdt.assert_frame_equal(splicing_data.data, data)

    def test_binify(self, splicing_data):
        from flotilla.compute.infotheory import binify
        test_binned = splicing_data.binify(splicing_data.data)

        true_binned = binify(splicing_data.data, splicing_data.bins)
        true_binned = true_binned.dropna(how='all', axis=1)

        pdt.assert_frame_equal(test_binned, true_binned)

    def test_binned_nmf_reduced(self, splicing_data,
                                data_for_binned_nmf_reduced):
        test_binned_nmf_reduced = splicing_data.binned_nmf_reduced(
            data=data_for_binned_nmf_reduced)

        if data_for_binned_nmf_reduced is None:
            data = splicing_data.data
        else:
            data = data_for_binned_nmf_reduced

        binned = splicing_data.binify(data)
        true_binned_nmf_reduced = splicing_data.nmf.transform(binned.T)

        pdt.assert_frame_equal(
            test_binned_nmf_reduced.sort_index(axis=0).sort_index(axis=1),
            true_binned_nmf_reduced.sort_index(axis=0).sort_index(axis=1))


    def test_nmf_space_positions(self, chr22):
        """Use chr22 dataset because need multiple phenotypes"""
        pass