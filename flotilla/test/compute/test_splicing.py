import itertools

import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd


@pytest.fixture(params=[None, (0.3, 0.7)])
def kwargs(request):
    if request.param is None:
        return {}
    else:
        return {'excluded_max': request.param[0],
                'included_min': request.param[1]}

@pytest.fixture
def binned(kwargs):
    from flotilla.compute.infotheory import bin_range_strings
    binned = np.array(list(itertools.product(np.arange(0, 1.1, .1), repeat=3)))
    binned = binned[binned.sum(axis=1) == 1]

    if kwargs is None:
        index = bin_range_strings((0, 0.2, 0.8, 1))
    else:
        index = bin_range_strings((0, kwargs['excluded_max'],
                                  kwargs['included_min'], 1))
    columns = ['event_{}'.format(i+1) for i in np.arange(binned.shape[0])]
    return pd.DataFrame(binned.T, index=index, columns=columns)

@pytest.fixture
def psi(binned):
    n_samples = 10
    unbinned = (binned * n_samples).astype(int).cumsum()
    psi = pd.DataFrame(index=np.arange(n_samples), columns=binned.columns)

    values = (0, 0.5, 1)

    for col in unbinned:
        for i, row in enumerate(unbinned[col]):
            if i == 0:
                row_i = 0
            else:
                row_i = unbinned.ix[i - 1, col]
            row_j = unbinned.ix[i, col]
            psi.ix[row_i:row_j, col] = values[i]
    return psi


class TestModalities:

    def test_init(self, kwargs):
        from flotilla.compute.splicing import Modalities

        test_modalities = Modalities(**kwargs)

        if kwargs == {}:
            npt.assert_equal(test_modalities.bins, (0, 0.2, 0.8, 1))
        else:
            npt.assert_equal(test_modalities.bins, (0, kwargs['excluded_max'],
                                                    kwargs['included_min'], 1))

    # def test_col_jsd_modalities(self, ):