from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import pytest
import numpy as np
import numpy.testing as npt


@pytest.fixture(params=['list', 'array', 'nan'])
def array(request):
    x = np.arange(0, 1.1, .1)
    if request.param == 'list':
        return list(x)
    elif request.param == 'array':
        return x
    elif request.param == 'nan':
        x[x > .8] = np.nan
        return x


def test_switchy_score(array):

    from flotilla.compute.splicing import switchy_score

    test_switchy_score = switchy_score(array)

    true_array = np.array(array)
    variance = 1 - np.std(np.sin(true_array[~np.isnan(true_array)] * np.pi))
    mean_value = -np.mean(np.cos(true_array[~np.isnan(true_array)] * np.pi))
    true_switchy_score = variance * mean_value

    npt.assert_array_equal(test_switchy_score, true_switchy_score)


def test_get_switchy_score_order(splicing_data):

    from flotilla.compute.splicing import switchy_score
    from flotilla.compute.splicing import get_switchy_score_order

    test_score_order = get_switchy_score_order(splicing_data)

    switchy_scores = np.apply_along_axis(switchy_score, axis=0,
                                         arr=splicing_data)
    true_score_order = np.argsort(switchy_scores)

    npt.assert_array_equal(test_score_order, true_score_order)
